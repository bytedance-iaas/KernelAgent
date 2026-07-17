"""SM90 MXFP8-FP8 contiguous m-grouped GEMM (CuTe DSL, TMA + WGMMA pipeline).

Implements the semantics of problem.py::run for
mxfp8_fp8_group_gemm_contiguous: A [M,K] fp8_e4m3 with per-(row, 32-K-block)
UE8M0 scales, B [G,N,K] fp8_e4m3 with per-(col, 32-K-block) UE8M0 scales,
rows of A partitioned into contiguous groups (grouped_layout), output
row i = dequant(A[i]) @ dequant(B[group(i)]).T in fp32, cast to bf16.

Architecture (adapted from the repo's SM90 fp8xfp4 grouped kernel):
  - TMA bulk-tensor loads for the A (BMxBK) and B (BNxBK) fp8 tiles into
    WGMMA-ready swizzled smem, arriving on a PipelineTmaAsync transaction
    mbarrier (3 stages). Warp 0 issues; zero per-thread LSU traffic.
  - UE8M0 scales are loaded as packed 32-bit quads (4 bytes = the 4 scale
    chunks of one 128-K tile; valid because K % 128 == 0) and staged in
    smem via 32-bit cp.async, one word per A row / B col per stage.
  - MXFP8 promotes each 32-K chunk with its own scale, so unlike plain FP8
    the 4 WGMMA chunks of a 128-K tile cannot share one accumulator: each
    chunk is drained and promoted as
        final += 2^(ea + eb - 254) * partial
    where ea/eb are the UE8M0 exponent bytes. A 2-deep ping-pong overlaps
    chunk kk+1's WGMMA with chunk kk's FP32 promotion (DeepGEMM's
    kTwoDeepPipeline). The scale product is computed exactly in the integer
    exponent domain (add + clamp + shift + bitcast), matching the CUDA
    kernel's e8m0_mul_to_float and the fp32 reference bit-for-bit over the
    normal exponent range.
  - Epilogue: bf16 packed 32-bit stores with row (group end) and col (N)
    guards. Out-of-group rows of a straddling tile are computed on
    TMA-loaded neighbor data but discarded by the row guard.
"""

from __future__ import annotations

import weakref

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack

_BM = 128
_BN = 64
_BK = 128              # k-tile; holds 4 UE8M0 scale chunks of 32 K each
_STAGES = 3
_NUM_THREADS = 256
_TMA_TX_BYTES = _BM * _BK + _BN * _BK   # A tile + B tile, both fp8


@cute.kernel
def _mxfp8_tma_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,        # (M, K) fp8 TMA coord tensor
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,        # (N, K, G) fp8 TMA coord tensor
    a_sfq: cute.Tensor,     # (M, K//128) int32: packed UE8M0 quads for A
    b_sfq: cute.Tensor,     # (G, N, K//128) int32: packed UE8M0 quads for B
    m_indptr: cute.Tensor,  # (G + 1,) int32
    out: cute.Tensor,       # (M, N) bf16
    N: cutlass.Int32,
    K: cutlass.Int32,
    sA_layout: cute.ComposedLayout,
    sB_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, bz = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    start = m_indptr[bz]
    end = m_indptr[bz + 1]
    bm = start + bx * _BM
    bn = by * _BN

    if bm < end:
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        smem = cutlass.utils.SmemAllocator()
        mbar_ptr = smem.allocate_array(cutlass.Int64, 2 * _STAGES)
        sA_lin = smem.allocate_tensor(
            cutlass.Float8E4M3FN, cute.make_layout(cute.cosize(sA_layout)),
            byte_alignment=1024)
        sB_lin = smem.allocate_tensor(
            cutlass.Float8E4M3FN, cute.make_layout(cute.cosize(sB_layout)),
            byte_alignment=1024)
        sSFA = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(_STAGES * _BM), byte_alignment=16)
        sSFB = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(_STAGES * _BN), byte_alignment=16)

        sA = cute.make_tensor(
            cute.recast_ptr(sA_lin.iterator, sA_layout.inner, dtype=cutlass.Float8E4M3FN),
            sA_layout.outer)
        sB = cute.make_tensor(
            cute.recast_ptr(sB_lin.iterator, sB_layout.inner, dtype=cutlass.Float8E4M3FN),
            sB_layout.outer)

        # ---- TMA pipeline (full/empty transaction mbarriers) ---- #
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, _NUM_THREADS // 32)
        mainloop_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr,
            num_stages=_STAGES,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=_TMA_TX_BYTES,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )

        # ---- global tiles + TMA partitions ---- #
        mA_off = cute.domain_offset((start, 0), mA)
        gA = cute.local_tile(mA_off, (_BM, _BK), (bx, None))    # (BM,BK,RestK)
        gB = cute.local_tile(mB, (_BN, _BK), (by, None, bz))    # (BN,BK,RestK)

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))

        atom_g2s4 = cute.make_copy_atom(
            cpasync.CopyG2SOp(), cutlass.Int32, num_bits_per_copy=32)
        atom_32 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=32)

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        cC = cute.make_identity_tensor((_BM, _BN))
        tCcC = thr_mma.partition_C(cC)
        p0 = cute.make_rmem_tensor(tCcC.shape, cutlass.Float32)
        p1 = cute.make_rmem_tensor(tCcC.shape, cutlass.Float32)
        final = cute.make_rmem_tensor(tCcC.shape, cutlass.Float32)
        final.fill(0.0)

        num_acc4 = cute.size(tCcC) // 4
        # per-thread copies of this stage's B-scale quads (2 cols per group of 4)
        qb = cute.make_rmem_tensor(2 * num_acc4, cutlass.Int32)
        # int->float bitcast scratch for the 4 chunk scales of one 4-group
        si = cute.make_rmem_tensor(4, cutlass.Int32)
        sfv = cute.make_tensor(
            cute.recast_ptr(si.iterator, dtype=cutlass.Float32),
            cute.make_layout(4))

        # ---- scale staging roles: threads 0..BM-1 fetch A quads, ---- #
        # ---- threads BM..BM+BN-1 fetch B quads, rest idle          ---- #
        sfkw = K >> 7                    # int32 words per scale row
        sf_is_a = tidx < _BM
        sf_do_b = (tidx >= _BM) & (tidx < _BM + _BN)
        sf_i = tidx
        if not sf_is_a:
            sf_i = tidx - _BM
        if (not sf_is_a) and (not sf_do_b):
            sf_i = 0                     # idle lanes: harmless duplicate
        sf_r = bm + sf_i
        if sf_r >= end:
            sf_r = end - 1
        sf_c = bn + sf_i
        if sf_c >= N:
            sf_c = N - 1
        sfa_base = sf_r * sfkw           # + k-tile index
        sfb_base = (bz * N + sf_c) * sfkw

        num_k_blocks = cute.size(tCrA, mode=[2])   # 4 chunks of 32 K
        k_tiles = K >> 7

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, _STAGES)
        read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, _STAGES)
        release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, _STAGES)

        # every WGMMA chunk overwrites its partial buffer; accumulation
        # happens in fp32 during scale promotion
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

        # ---- prologue: TMA-prefetch stages 0..STAGES-2 + their scales ---- #
        # Always emit STAGES-1 commit groups (empty ones are legal) so the
        # mainloop's cp_async_wait_group(STAGES-2) accounting holds when
        # k_tiles < STAGES-1.
        for _ in cutlass.range_constexpr(_STAGES - 1):
            if producer_state.count < k_tiles:
                if warp_idx == 0:
                    mainloop_pipe.producer_acquire(producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, producer_state.count)],
                        tAsA[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipe.producer_get_barrier(producer_state))
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, producer_state.count)],
                        tBsB[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipe.producer_get_barrier(producer_state))
                    mainloop_pipe.producer_commit(producer_state)
                kb0 = producer_state.count
                sslot = producer_state.index
                if sf_is_a:
                    ssrc = cute.make_tensor(
                        (a_sfq.iterator + (sfa_base + kb0)).align(4),
                        cute.make_layout(1))
                    sdst = cute.make_tensor(
                        (sSFA.iterator + (sslot * _BM + sf_i)).align(4),
                        cute.make_layout(1))
                    cute.copy(atom_g2s4, ssrc, sdst)
                if sf_do_b:
                    ssrc = cute.make_tensor(
                        (b_sfq.iterator + (sfb_base + kb0)).align(4),
                        cute.make_layout(1))
                    sdst = cute.make_tensor(
                        (sSFB.iterator + (sslot * _BN + sf_i)).align(4),
                        cute.make_layout(1))
                    cute.copy(atom_g2s4, ssrc, sdst)
            cute.arch.cp_async_commit_group()
            producer_state.advance()

        # ---- mainloop over 128-K tiles ---- #
        for s in cutlass.range(k_tiles, unroll=1):
            # scales for tile s landed; make all threads' staging visible
            cute.arch.cp_async_wait_group(_STAGES - 2)
            cute.arch.sync_threads()
            # TMA data for tile s landed
            mainloop_pipe.consumer_wait(read_state)
            buf = read_state.index
            read_state.advance()

            # this thread's A-scale quads (2 accumulator rows) and B-scale
            # quads (2 cols per 4-group) for the 4 chunks of this tile
            qa0 = sSFA[buf * _BM + tCcC[0][0]]
            qa1 = sSFA[buf * _BM + tCcC[2][0]]
            for n4 in cutlass.range_constexpr(num_acc4):
                c0 = tCcC[4 * n4][1]
                qb[2 * n4] = sSFB[buf * _BN + c0]
                qb[2 * n4 + 1] = sSFB[buf * _BN + c0 + 1]

            # ---- 2-deep chunk pipeline: WGMMA(kk+1) overlaps promote(kk) --- #
            cute.nvgpu.warpgroup.fence()
            cute.gemm(tiled_mma, p0,
                      tCrA[(None, None, 0, buf)],
                      tCrB[(None, None, 0, buf)], p0)
            cute.nvgpu.warpgroup.commit_group()

            for kk in cutlass.range_constexpr(num_k_blocks):
                if cutlass.const_expr(kk + 1 < num_k_blocks):
                    if cutlass.const_expr((kk + 1) % 2 == 0):
                        cute.nvgpu.warpgroup.fence()
                        cute.gemm(tiled_mma, p0,
                                  tCrA[(None, None, kk + 1, buf)],
                                  tCrB[(None, None, kk + 1, buf)], p0)
                        cute.nvgpu.warpgroup.commit_group()
                    else:
                        cute.nvgpu.warpgroup.fence()
                        cute.gemm(tiled_mma, p1,
                                  tCrA[(None, None, kk + 1, buf)],
                                  tCrB[(None, None, kk + 1, buf)], p1)
                        cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(1)
                else:
                    cute.nvgpu.warpgroup.wait_group(0)

                # ---- promote drained chunk kk:                        ---- #
                # ---- final += bitcast((ea+eb-127)<<23) * partial      ---- #
                ea0 = (qa0 >> (8 * kk)) & 0xFF
                ea1 = (qa1 >> (8 * kk)) & 0xFF
                for n4 in cutlass.range_constexpr(num_acc4):
                    i0 = 4 * n4
                    eb0 = (qb[2 * n4] >> (8 * kk)) & 0xFF
                    eb1 = (qb[2 * n4 + 1] >> (8 * kk)) & 0xFF
                    si[0] = cutlass.max(cutlass.min(ea0 + eb0 - 127, 255), 0) << 23
                    si[1] = cutlass.max(cutlass.min(ea0 + eb1 - 127, 255), 0) << 23
                    si[2] = cutlass.max(cutlass.min(ea1 + eb0 - 127, 255), 0) << 23
                    si[3] = cutlass.max(cutlass.min(ea1 + eb1 - 127, 255), 0) << 23
                    if cutlass.const_expr(kk % 2 == 0):
                        final[i0] = final[i0] + sfv[0] * p0[i0]
                        final[i0 + 1] = final[i0 + 1] + sfv[1] * p0[i0 + 1]
                        final[i0 + 2] = final[i0 + 2] + sfv[2] * p0[i0 + 2]
                        final[i0 + 3] = final[i0 + 3] + sfv[3] * p0[i0 + 3]
                    else:
                        final[i0] = final[i0] + sfv[0] * p1[i0]
                        final[i0 + 1] = final[i0 + 1] + sfv[1] * p1[i0 + 1]
                        final[i0 + 2] = final[i0 + 2] + sfv[2] * p1[i0 + 2]
                        final[i0 + 3] = final[i0 + 3] + sfv[3] * p1[i0 + 3]

            mainloop_pipe.consumer_release(release_state)
            release_state.advance()

            # ---- TMA-refill stage s+STAGES-1 + its scales ---- #
            if producer_state.count < k_tiles:
                if warp_idx == 0:
                    mainloop_pipe.producer_acquire(producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, producer_state.count)],
                        tAsA[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipe.producer_get_barrier(producer_state))
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, producer_state.count)],
                        tBsB[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipe.producer_get_barrier(producer_state))
                    mainloop_pipe.producer_commit(producer_state)
                kbn = producer_state.count
                sslot = producer_state.index
                if sf_is_a:
                    ssrc = cute.make_tensor(
                        (a_sfq.iterator + (sfa_base + kbn)).align(4),
                        cute.make_layout(1))
                    sdst = cute.make_tensor(
                        (sSFA.iterator + (sslot * _BM + sf_i)).align(4),
                        cute.make_layout(1))
                    cute.copy(atom_g2s4, ssrc, sdst)
                if sf_do_b:
                    ssrc = cute.make_tensor(
                        (b_sfq.iterator + (sfb_base + kbn)).align(4),
                        cute.make_layout(1))
                    sdst = cute.make_tensor(
                        (sSFB.iterator + (sslot * _BN + sf_i)).align(4),
                        cute.make_layout(1))
                    cute.copy(atom_g2s4, ssrc, sdst)
                producer_state.advance()
            cute.arch.cp_async_commit_group()

        # ---- epilogue: pack adjacent col pairs into 32-bit stores ---- #
        ov = cute.make_rmem_tensor(2, cutlass.BFloat16)
        for i2 in cutlass.range_constexpr(cute.size(final) // 2):
            i = 2 * i2
            coord = tCcC[i]
            orow = bm + coord[0]
            ocol = bn + coord[1]
            if orow < end and ocol + 1 < N:
                ov[0] = final[i].to(cutlass.BFloat16)
                ov[1] = final[i + 1].to(cutlass.BFloat16)
                dstv = cute.make_tensor(
                    (out.iterator + (orow * N + ocol)).align(4),
                    cute.make_layout(2))
                cute.copy(atom_32, ov, dstv)


@cute.jit
def _launch_jit(
    a8: cute.Tensor,        # (M, K) fp8
    b8: cute.Tensor,        # (N, K, G) fp8 (permuted view of (G, N, K))
    a_sfq: cute.Tensor,     # (M, K//128) int32
    b_sfq: cute.Tensor,     # (G, N, K//128) int32
    m_indptr: cute.Tensor,  # (G + 1,) int32
    out: cute.Tensor,       # (M, N) bf16
    N: cutlass.Int32,
    K: cutlass.Int32,
    stream: cuda.CUstream,
    GRID_M: cutlass.Constexpr,
    GRID_N: cutlass.Constexpr,
    G: cutlass.Constexpr,
):
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        cute.nvgpu.warpgroup.OperandMajorMode.K,
        cute.nvgpu.warpgroup.OperandMajorMode.K,
        cutlass.Float32,
        (2, 1, 1),
        tiler_mn=(64, _BN),
    )
    sA_layout = sm90_utils.make_smem_layout_a(
        utils.LayoutEnum.ROW_MAJOR, (_BM, _BN, _BK), cutlass.Float8E4M3FN, _STAGES)
    sB_layout = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, (_BM, _BN, _BK), cutlass.Float8E4M3FN, _STAGES)

    tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        a8,
        cute.slice_(sA_layout, (None, None, 0)),
        (_BM, _BK),
    )
    tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        b8,
        cute.slice_(sB_layout, (None, None, 0)),
        (_BN, _BK),
    )

    _mxfp8_tma_kernel(
        tiled_mma,
        tma_atom_a, tma_tensor_a,
        tma_atom_b, tma_tensor_b,
        a_sfq, b_sfq, m_indptr, out, N, K,
        sA_layout, sB_layout,
    ).launch(
        grid=[GRID_M, GRID_N, G],
        block=[_NUM_THREADS, 1, 1],
        stream=stream,
    )


_COMPILE_CACHE: dict = {}
_INDPTR_CACHE: dict = {}


def _group_offsets(grouped_layout: torch.Tensor, G: int):
    # Key on data_ptr but validate OBJECT IDENTITY via weakref: a freed
    # tensor's address can be reused by a different same-shape tensor.
    key = (grouped_layout.data_ptr(), grouped_layout.numel(), G)
    hit = _INDPTR_CACHE.get(key)
    if hit is not None:
        src_ref, m_indptr, max_m = hit
        if src_ref() is grouped_layout:
            return m_indptr, max_m
    gl = grouped_layout.to(torch.int64)
    valid = gl >= 0                     # rows with group -1 are padding
    counts = torch.bincount(gl[valid], minlength=G)
    m_indptr = torch.zeros(G + 1, dtype=torch.int32, device=grouped_layout.device)
    m_indptr[1:] = counts.cumsum(0).to(torch.int32)
    max_m = int(counts.max().item())
    _INDPTR_CACHE[key] = (weakref.ref(grouped_layout), m_indptr, max_m)
    return m_indptr, max_m


def _launch_grouped(a8, b8_nkg, a_sfq, b_sfq, m_indptr, out, grid_m):
    M, K = a8.shape
    N, _, G = b8_nkg.shape
    grid_n = (N + _BN - 1) // _BN

    # Launch on torch's CURRENT stream (fresh each call): required for CUDA
    # graph capture and correctness under stream contexts.
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def _dyn_args():
        return (
            from_dlpack(a8, assumed_align=16).mark_layout_dynamic(1),
            from_dlpack(b8_nkg, assumed_align=16).mark_layout_dynamic(1),
            from_dlpack(a_sfq).mark_layout_dynamic(1),
            from_dlpack(b_sfq).mark_layout_dynamic(2),
            from_dlpack(m_indptr).mark_layout_dynamic(0),
            from_dlpack(out).mark_layout_dynamic(1),
            cutlass.Int32(N), cutlass.Int32(K), stream,
        )

    constexpr_args = (grid_m, grid_n, G)
    key = (M, N, K, G, grid_m)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_launch_jit, *_dyn_args(), *constexpr_args)
        _COMPILE_CACHE[key] = compiled
    compiled(*_dyn_args())


def kernel_function(
    a_data: torch.Tensor,          # (M, K) fp8_e4m3
    a_scale: torch.Tensor,         # (M, K // 32) uint8 UE8M0
    b_data: torch.Tensor,          # (G, N, K) fp8_e4m3
    b_scale: torch.Tensor,         # (G, N, K // 32) uint8 UE8M0
    grouped_layout: torch.Tensor,  # (M,) int32, contiguous groups
) -> torch.Tensor:
    """Contiguous m-grouped MXFP8-FP8 GEMM (NT); returns (M, N) bf16."""
    assert a_data.dtype == torch.float8_e4m3fn
    assert b_data.dtype == torch.float8_e4m3fn
    assert a_scale.dtype == torch.uint8 and b_scale.dtype == torch.uint8
    M, K = a_data.shape
    G, N, Kb = b_data.shape
    assert Kb == K
    assert K % _BK == 0, "kernel assumes 128-K tiles (4 UE8M0 chunks each)"
    assert N % 2 == 0 or N == 1, "paired epilogue stores need even N"
    assert a_scale.shape == (M, K // 32)
    assert b_scale.shape == (G, N, K // 32)

    # zero-init: rows whose group id is negative (padding) stay zero
    out = torch.zeros(M, N, device=a_data.device, dtype=torch.bfloat16)
    if M == 0 or N == 0 or K == 0:
        return out

    m_indptr, max_m = _group_offsets(grouped_layout, G)
    if max_m == 0:
        return out
    grid_m = (max_m + _BM - 1) // _BM

    # UE8M0 bytes -> packed int32 quads (byte kk = scale of 32-K chunk kk);
    # a pure reinterpret view, valid because K % 128 == 0.
    a_sfq = a_scale.contiguous().view(torch.int32)          # (M, K//128)
    b_sfq = b_scale.contiguous().view(torch.int32)          # (G, N, K//128)

    # (G, N, K) viewed as (N, K, G) for the 3-D TMA coord tensor
    b8_nkg = b_data.contiguous().permute(1, 2, 0)

    _launch_grouped(
        a_data.contiguous(), b8_nkg, a_sfq, b_sfq, m_indptr, out, grid_m)
    return out
