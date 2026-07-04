"""SM90 FP8xFP4 contiguous m-grouped GEMM (CuTe DSL, TMA + mbarrier pipeline).

Round-20: decode results written straight into an Int32[4] rmem tensor
(viewed as Int64[2] for the 128-bit STS) - drops the Int64 packing ops.

Round-19: direct k-order prmt decode - the even/odd nibble split +
re-interleave was unnecessary (LE nibble order == k order); 8 codes now
cost ~16 ALU ops vs ~30 (kernel is issue-bound: 59% issue-active).

Round-18: coalesced scale loads - b_sf transposed to (G, SFK, N) once
and cached (weights); a_sf transposed per call when M >= 4096. The
per-tile scale gathers were 4B loads strided 224B (one L2 sector each,
long-scoreboard 10.4% at m1024).

Round-17: BN 128->64, STAGES 5->3 -> ~86 KiB smem, 2 CTAs/SM (4
warps/scheduler for latency hiding at the issue-bound profile).

Round-16: round-14 + LDS.64 paired scale loads + packed 32-bit epilogue
stores (both register-neutral).

Round-11 rewrite. Round-10 NCU: l1tex throughput 91.8% active with the tensor
pipe at 15.4% -> the shared/LSU pipe is the wall (per-thread cp.async stores,
decode LDS/STS, scalar promotion loads), all serialized against wgmma by
full-CTA syncs. This round adopts DeepGEMM's load architecture:

  - TMA bulk-tensor loads (warp 0 issues; zero per-thread LSU traffic):
      A fp8 tile      -> wgmma-ready swizzled smem (TMA does the swizzle)
      packed-FP4 B    -> linear raw smem
    both arrive on a PipelineTmaAsync transaction mbarrier (5 stages).
  - BN 64 -> 128: halves A-side smem traffic per FLOP, grid 256 CTAs on the
    canonical shape (1.94 waves vs 3.88).
  - ACCUMULATE=False on the first wgmma of each k-tile: `partial.fill(0)`
    disappears.
  - Dynamic group offsets go through cute.domain_offset on the TMA coord
    tensor; TMA zero-fills out-of-bounds rows (no more row clamping).
  - Scales stay on 32-bit cp.async (TMA boxes need 16-byte inner extents);
    1 KiB/stage is noise next to the 24 KiB the TMA moves.

Decode-ahead is kept: tile s+1's raw B is prmt-decoded while tile s's wgmma
runs. Scale promotion per 128-K block keeps the math exact vs the reference.
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
_BK = 128          # == gran_k: one scale block per k-tile
_BKH = _BK // 2    # packed FP4 bytes per k-tile row
_STAGES = 3
_NUM_THREADS = 256
_RAW_STAGE_BYTES = _BN * _BKH   # 8 KiB packed-FP4 per stage
_TMA_TX_BYTES = _BM * _BK + _RAW_STAGE_BYTES   # A tile + raw B tile


@cute.kernel
def _w4a8_tma_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,       # (M, K) fp8 TMA coord tensor
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,       # (N, K//2, G) fp8-typed packed FP4 TMA coord tensor
    a_sf: cute.Tensor,     # (M, SFK) fp32, or (SFK, M) when A_T
    b_sf: cute.Tensor,     # (G, SFK, N) fp32 (transposed for coalescing)
    m_indptr: cute.Tensor,  # (G + 1,)            int32
    out: cute.Tensor,      # (M, N)               bf16
    N: cutlass.Int32,
    K: cutlass.Int32,
    MTOT: cutlass.Int32,
    sA_layout: cute.ComposedLayout,
    sBop_layout: cute.ComposedLayout,
    A_T: cutlass.Constexpr,
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
        sBop_lin = smem.allocate_tensor(
            cutlass.Float8E4M3FN, cute.make_layout(cute.cosize(sBop_layout)),
            byte_alignment=1024)
        sBraw = smem.allocate_tensor(
            cutlass.Float8E4M3FN,
            cute.make_layout((_BN, _BKH, _STAGES),
                             stride=(_BKH, 1, _RAW_STAGE_BYTES)),
            byte_alignment=1024)
        sSFA = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout(_STAGES * _BM), byte_alignment=16)
        sSFB = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout(_STAGES * _BN), byte_alignment=16)

        sA = cute.make_tensor(
            cute.recast_ptr(sA_lin.iterator, sA_layout.inner, dtype=cutlass.Float8E4M3FN),
            sA_layout.outer)
        sBop = cute.make_tensor(
            cute.recast_ptr(sBop_lin.iterator, sBop_layout.inner, dtype=cutlass.Float8E4M3FN),
            sBop_layout.outer)
        sBraw64 = cute.make_tensor(
            cute.recast_ptr(sBraw.iterator, dtype=cutlass.Int64),
            cute.make_layout(_STAGES * _RAW_STAGE_BYTES // 8))

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
        gA = cute.local_tile(mA_off, (_BM, _BK), (bx, None))       # (BM,BK,RestK)
        gB = cute.local_tile(mB, (_BN, _BKH), (by, None, bz))      # (BN,BKH,RestK)

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sBraw, 0, 2), cute.group_modes(gB, 0, 2))

        atom_g2s4 = cute.make_copy_atom(
            cpasync.CopyG2SOp(), cutlass.Float32, num_bits_per_copy=32)
        atom_128 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Int64, num_bits_per_copy=128)
        atom_64 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=64)

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sBop)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        cC = cute.make_identity_tensor((_BM, _BN))
        tCcC = thr_mma.partition_C(cC)
        partial = cute.make_rmem_tensor(tCcC.shape, cutlass.Float32)
        final = cute.make_rmem_tensor(tCcC.shape, cutlass.Float32)
        final.fill(0.0)

        # ---- decode thread mapping: 2 threads per B row, 32 raw B/thread --- #
        sb_off = cute.make_rmem_tensor(2, cutlass.Int32)
        for j in cutlass.range_constexpr(2):
            sb_off[j] = sBop_layout((tidx // 8 + 32 * j, (tidx % 8) * 16, 0))
        stageBop = cute.cosize(sBop_layout) // _STAGES
        w4 = cute.make_rmem_tensor(4, cutlass.Int32)
        w2 = cute.make_tensor(
            cute.recast_ptr(w4.iterator, dtype=cutlass.Int64), cute.make_layout(2))

        # prmt magnitude LUTs: e4m3 encodings of E2M1 magnitudes 0..7
        _L0 = 0x3C383000
        _L1 = 0x4C484440

        # ---- scale staging: threads 0..127 fetch a_sf, 128..255 fetch b_sf - #
        sfk = K >> 7
        sf_is_a = tidx < _BM
        sf_do_b = (tidx >= _BM) & (tidx < _BM + _BN)
        sf_i = tidx
        if not sf_is_a:
            sf_i = tidx - _BM
        if (not sf_is_a) and (not sf_do_b):
            sf_i = 0  # idle scale lanes: harmless duplicate
        sf_r = bm + sf_i
        if sf_r >= end:
            sf_r = end - 1
        sf_c = bn + sf_i
        if sf_c >= N:
            sf_c = N - 1
        if cutlass.const_expr(A_T):
            sfa_base = sf_r                  # + kb * MTOT (coalesced)
            sfa_step = MTOT
        else:
            sfa_base = sf_r * sfk            # + kb (strided gather)
            sfa_step = cutlass.Int32(1)
        sfb_base = bz * sfk * N + sf_c       # + kb * N (coalesced)

        num_k_blocks = cute.size(tCrA, mode=[2])
        num_acc4 = cute.size(tCcC) // 4
        k_tiles = K >> 7

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, _STAGES)
        read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, _STAGES)
        release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, _STAGES)

        # ---- prologue: TMA-prefetch stages 0..STAGES-2, stage the scales --- #
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
                    ssrc = cute.make_tensor((a_sf.iterator + (sfa_base + kb0 * sfa_step)).align(4),
                                            cute.make_layout(1))
                    sdst = cute.make_tensor((sSFA.iterator + (sslot * _BM + sf_i)).align(4),
                                            cute.make_layout(1))
                    cute.copy(atom_g2s4, ssrc, sdst)
                if sf_do_b:
                    ssrc = cute.make_tensor((b_sf.iterator + (sfb_base + kb0 * N)).align(4),
                                            cute.make_layout(1))
                    sdst = cute.make_tensor((sSFB.iterator + (sslot * _BN + sf_i)).align(4),
                                            cute.make_layout(1))
                    cute.copy(atom_g2s4, ssrc, sdst)
            cute.arch.cp_async_commit_group()
            producer_state.advance()

        # ---- decode tile 0 ---- #
        mainloop_pipe.consumer_wait(read_state)
        rb = read_state.index * (_RAW_STAGE_BYTES // 8) + tidx
        for j in cutlass.range_constexpr(2):
            v = sBraw64[rb + 256 * j]
            for h in cutlass.range_constexpr(2):
                # nibbles are already in k-order within an LE word:
                # the masked word IS the prmt selector (bits 2:0 per
                # nibble = magnitude LUT index), sign via 2nd prmt.
                w = ((v >> (32 * h)) & 0xFFFFFFFF).to(cutlass.Int32)
                w16 = w >> 16
                w4[2 * h] = cutlass.Int32(cute.arch.prmt(_L0, _L1, w & 0x7777)) \
                    | cutlass.Int32(cute.arch.prmt(0, 0x80808080, (w & 0x8888) >> 1))
                w4[2 * h + 1] = cutlass.Int32(cute.arch.prmt(_L0, _L1, w16 & 0x7777)) \
                    | cutlass.Int32(cute.arch.prmt(0, 0x80808080, (w16 & 0x8888) >> 1))
            dec_dst = cute.make_tensor(
                (cute.recast_ptr(sBop_lin.iterator, dtype=cutlass.Int64)
                 + ((sb_off[j] + read_state.index * stageBop) >> 3)).align(16),
                cute.make_layout(2))
            cute.copy(atom_128, w2, dec_dst)
        read_state.advance()

        # ---- mainloop ---- #
        for s in cutlass.range(k_tiles, unroll=1):
            # scale(s) landed; decode(s) STS + scale stores must be CTA-visible
            cute.arch.cp_async_wait_group(_STAGES - 2)
            cute.arch.sync_threads()
            cute.arch.fence_proxy("async.shared", space="cta")

            # ---- QGMMA over the 128-K block; first wgmma overwrites ---- #
            buf = release_state.index
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
            cute.nvgpu.warpgroup.fence()
            for kblk in cutlass.range_constexpr(num_k_blocks):
                cute.gemm(
                    tiled_mma,
                    partial,
                    tCrA[(None, None, kblk, buf)],
                    tCrB[(None, None, kblk, buf)],
                    partial,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()

            # ---- TMA-refill stage s+STAGES-1 while the wgmma runs ---- #
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
                    ssrc = cute.make_tensor((a_sf.iterator + (sfa_base + kbn * sfa_step)).align(4),
                                            cute.make_layout(1))
                    sdst = cute.make_tensor((sSFA.iterator + (sslot * _BM + sf_i)).align(4),
                                            cute.make_layout(1))
                    cute.copy(atom_g2s4, ssrc, sdst)
                if sf_do_b:
                    ssrc = cute.make_tensor((b_sf.iterator + (sfb_base + kbn * N)).align(4),
                                            cute.make_layout(1))
                    sdst = cute.make_tensor((sSFB.iterator + (sslot * _BN + sf_i)).align(4),
                                            cute.make_layout(1))
                    cute.copy(atom_g2s4, ssrc, sdst)
                producer_state.advance()
            cute.arch.cp_async_commit_group()

            # ---- decode NEXT tile's raw B while the wgmma runs ---- #
            if read_state.count < k_tiles:
                mainloop_pipe.consumer_wait(read_state)
                rb = read_state.index * (_RAW_STAGE_BYTES // 8) + tidx
                for j in cutlass.range_constexpr(2):
                    v = sBraw64[rb + 256 * j]
                    for h in cutlass.range_constexpr(2):
                        # nibbles are already in k-order within an LE word:
                        # the masked word IS the prmt selector (bits 2:0 per
                        # nibble = magnitude LUT index), sign via 2nd prmt.
                        w = ((v >> (32 * h)) & 0xFFFFFFFF).to(cutlass.Int32)
                        w16 = w >> 16
                        w4[2 * h] = cutlass.Int32(cute.arch.prmt(_L0, _L1, w & 0x7777)) \
                            | cutlass.Int32(cute.arch.prmt(0, 0x80808080, (w & 0x8888) >> 1))
                        w4[2 * h + 1] = cutlass.Int32(cute.arch.prmt(_L0, _L1, w16 & 0x7777)) \
                            | cutlass.Int32(cute.arch.prmt(0, 0x80808080, (w16 & 0x8888) >> 1))
                    dec_dst = cute.make_tensor(
                        (cute.recast_ptr(sBop_lin.iterator, dtype=cutlass.Int64)
                         + ((sb_off[j] + read_state.index * stageBop) >> 3)).align(16),
                        cute.make_layout(2))
                    cute.copy(atom_128, w2, dec_dst)
                read_state.advance()

            cute.nvgpu.warpgroup.wait_group(0)

            # ---- promotion: final += a_sf[row] * b_sf[col] * partial ---- #
            # col pairs (2j, 2j+1) are adjacent in sSFB -> one LDS.64 each
            sa0 = sSFA[buf * _BM + tCcC[0][0]]
            sa1 = sSFA[buf * _BM + tCcC[2][0]]
            sbp = cute.make_rmem_tensor(2, cutlass.Float32)
            for n4 in cutlass.range_constexpr(num_acc4):
                i0 = 4 * n4
                sbs = cute.make_tensor(
                    (sSFB.iterator + (buf * _BN + tCcC[i0][1])).align(8),
                    cute.make_layout(2))
                cute.copy(atom_64, sbs, sbp)
                sb0 = sbp[0]
                sb1 = sbp[1]
                final[i0] = final[i0] + sa0 * sb0 * partial[i0]
                final[i0 + 1] = final[i0 + 1] + sa0 * sb1 * partial[i0 + 1]
                final[i0 + 2] = final[i0 + 2] + sa1 * sb0 * partial[i0 + 2]
                final[i0 + 3] = final[i0 + 3] + sa1 * sb1 * partial[i0 + 3]

            mainloop_pipe.consumer_release(release_state)
            release_state.advance()

        # ---- epilogue: pack adjacent col pairs into 32-bit stores ---- #
        ov = cute.make_rmem_tensor(2, cutlass.BFloat16)
        atom_32 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=32)
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
    a8: cute.Tensor,       # (M, K) fp8
    b8: cute.Tensor,       # (N, K//2, G) fp8-typed packed FP4 (permuted view)
    a_sf: cute.Tensor,
    b_sf: cute.Tensor,
    m_indptr: cute.Tensor,
    out: cute.Tensor,
    N: cutlass.Int32,
    K: cutlass.Int32,
    MTOT: cutlass.Int32,
    stream: cuda.CUstream,
    GRID_M: cutlass.Constexpr,
    GRID_N: cutlass.Constexpr,
    G: cutlass.Constexpr,
    A_T: cutlass.Constexpr,
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
    sBop_layout = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, (_BM, _BN, _BK), cutlass.Float8E4M3FN, _STAGES)
    sBraw_stage = cute.make_layout((_BN, _BKH), stride=(_BKH, 1))

    tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        a8,
        cute.slice_(sA_layout, (None, None, 0)),
        (_BM, _BK),
    )
    tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        b8,
        sBraw_stage,
        (_BN, _BKH),
    )

    _w4a8_tma_kernel(
        tiled_mma,
        tma_atom_a, tma_tensor_a,
        tma_atom_b, tma_tensor_b,
        a_sf, b_sf, m_indptr, out, N, K, MTOT,
        sA_layout, sBop_layout, A_T,
    ).launch(
        grid=[GRID_M, GRID_N, G],
        block=[_NUM_THREADS, 1, 1],
        stream=stream,
    )


_COMPILE_CACHE: dict = {}
_INDPTR_CACHE: dict = {}
_BSF_T_CACHE: dict = {}
_A_T_MIN_M = 4096   # transpose a_sf per call only when M amortizes it


def _bsf_transposed(b_sf: torch.Tensor):
    # Key on data_ptr but validate OBJECT IDENTITY via weakref: a freed
    # tensor's address can be reused by a different same-shape tensor, and
    # returning the stale transpose then silently corrupts the scales.
    key = (b_sf.data_ptr(), tuple(b_sf.shape))
    hit = _BSF_T_CACHE.get(key)
    if hit is not None:
        src_ref, t = hit
        if src_ref() is b_sf:
            return t
    t = b_sf.transpose(1, 2).contiguous()   # (G, SFK, N)
    _BSF_T_CACHE[key] = (weakref.ref(b_sf), t)
    return t


def _launch_grouped(a8, b8_nkg, a_sf, b_sf_t, m_indptr, out, grid_m, a_t):
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
            from_dlpack(a_sf).mark_layout_dynamic(1),
            from_dlpack(b_sf_t).mark_layout_dynamic(2),
            from_dlpack(m_indptr).mark_layout_dynamic(0),
            from_dlpack(out).mark_layout_dynamic(1),
            cutlass.Int32(N), cutlass.Int32(K), cutlass.Int32(M), stream,
        )

    constexpr_args = (grid_m, grid_n, G, a_t)
    key = (M, N, K, G, grid_m, a_t)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_launch_jit, *_dyn_args(), *constexpr_args)
        _COMPILE_CACHE[key] = compiled
    compiled(*_dyn_args())


_DEFAULT_WEIGHTS = None


def _default_weights(device):
    global _DEFAULT_WEIGHTS
    if _DEFAULT_WEIGHTS is None:
        import importlib.util as _ilu
        from pathlib import Path as _Path
        _pp = _Path(__file__).resolve().parent / "problem.py"
        _spec = _ilu.spec_from_file_location("problem", _pp)
        problem = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(problem)
        b_fp4, b_sf, grouped_layout, gran_k = problem.get_init_inputs()
        _DEFAULT_WEIGHTS = (b_fp4.to(device), b_sf.to(device),
                            grouped_layout.to(device), gran_k)
    return _DEFAULT_WEIGHTS


def _group_offsets(grouped_layout: torch.Tensor, G: int):
    key = (grouped_layout.data_ptr(), grouped_layout.numel(), G)
    hit = _INDPTR_CACHE.get(key)
    if hit is not None:
        src_ref, m_indptr, max_m = hit
        if src_ref() is grouped_layout:
            return m_indptr, max_m
    counts = torch.bincount(grouped_layout.to(torch.int64), minlength=G)
    m_indptr = torch.zeros(G + 1, dtype=torch.int32, device=grouped_layout.device)
    m_indptr[1:] = counts.cumsum(0).to(torch.int32)
    max_m = int(counts.max().item())
    _INDPTR_CACHE[key] = (weakref.ref(grouped_layout), m_indptr, max_m)
    return m_indptr, max_m


def kernel_function(
    a_fp8: torch.Tensor,          # (M, K) fp8_e4m3
    a_sf: torch.Tensor,           # (M, K // gran_k) fp32
    b_fp4: torch.Tensor = None,   # (G, N, K // 2) int8 packed FP4-E2M1
    b_sf: torch.Tensor = None,    # (G, N, K // gran_k) fp32
    grouped_layout: torch.Tensor = None,  # (M,) int32, contiguous groups
    gran_k: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Contiguous m-grouped FP8xFP4 GEMM; returns (M, N) bf16."""
    if b_fp4 is None:
        b_fp4, b_sf, grouped_layout, gran_k = _default_weights(a_fp8.device)
    if a_fp8.dtype != torch.float8_e4m3fn:
        a_fp8 = a_fp8.to(torch.float8_e4m3fn)
    if a_sf.dtype != torch.float32:
        a_sf = a_sf.to(torch.float32)
    if b_sf.dtype != torch.float32:
        b_sf = b_sf.to(torch.float32)
    assert out_dtype == torch.bfloat16
    M, K = a_fp8.shape
    G, N, half_k = b_fp4.shape
    assert half_k * 2 == K
    assert K % _BK == 0 and gran_k == _BK, "kernel assumes 128-K scale blocks"
    assert a_sf.shape == (M, K // gran_k)
    assert b_sf.shape == (G, N, K // gran_k)

    out = torch.empty(M, N, device=a_fp8.device, dtype=out_dtype)
    if M == 0 or N == 0:
        return out

    m_indptr, max_m = _group_offsets(grouped_layout, G)
    if max_m == 0:
        return out
    grid_m = (max_m + _BM - 1) // _BM

    # (G, N, K/2) raw packed bytes viewed as (N, K/2, G) for the 3-D TMA
    b8_nkg = b_fp4.contiguous().view(torch.float8_e4m3fn).permute(1, 2, 0)

    b_sf_t = _bsf_transposed(b_sf.contiguous())     # (G, SFK, N), cached
    a_t = M >= _A_T_MIN_M
    a_sf_x = a_sf.t().contiguous() if a_t else a_sf.contiguous()

    _launch_grouped(
        a_fp8.contiguous(), b8_nkg, a_sf_x, b_sf_t,
        m_indptr, out, grid_m, a_t,
    )
    return out
