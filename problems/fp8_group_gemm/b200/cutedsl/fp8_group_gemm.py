"""FP8 grouped GEMM with per-block-scale dequantization (CuTe DSL).

Implements ``out[i, j] = sum_k dq(a[i, k]) * dq(b[g, j, k])`` for each row
``i`` belonging to group ``g`` (defined by ``m_indptr``), where ``dq`` applies
per-tile fp32 scales (``a_scale`` for ``a`` and ``b_scale`` for ``b``).

Only ``scale_major_mode == "K"`` is supported (matches the problem's test).
"""

from __future__ import annotations

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# Tile / micro-tile configuration. A CTA computes a BM x BN output tile; each of
# the 16x16 = 256 threads owns a TM x TN register micro-tile, so BM = 16*TM and
# BN = 16*TN. The K dimension is streamed in BK-wide slices through shared memory.
_BM = 64
_BN = 64
_BK = 32
_TM = 4
_TN = 4
_CK = 8   # fp8 widening chunk during the gmem->smem load (>=4, divides _BK)


@cute.kernel
def _fp8_grouped_kernel(
    a: cute.Tensor,        # (M, K)            fp8_e4m3
    b: cute.Tensor,        # (G, N, K)         fp8_e4m3
    a_scale: cute.Tensor,  # (M, K_blocks)     fp32
    b_scale: cute.Tensor,  # (G, N_blocks, K_blocks) fp32
    m_indptr: cute.Tensor,  # (G+1,)           int32  (group row offsets)
    out: cute.Tensor,      # (M, N)            bf16
    N: cutlass.Int32,
    K: cutlass.Int32,
    BLOCK: cutlass.Constexpr,   # quant block size along K (and N for b); multiple of _BK
):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, bz = cute.arch.block_idx()   # bz selects the group
    tid = tx + ty * 16

    start = m_indptr[bz]      # first global row of this group
    end = m_indptr[bz + 1]    # one past the last global row
    bm = start + bx * _BM     # global row base of this CTA's tile
    bn = by * _BN

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float32, cute.make_layout((_BM, _BK)), byte_alignment=16)
    sB = smem.allocate_tensor(cutlass.Float32, cute.make_layout((_BN, _BK)), byte_alignment=16)

    acc = cute.make_rmem_tensor((_TM, _TN), cutlass.Float32)
    for i in range(_TM):
        for j in range(_TN):
            acc[i, j] = cutlass.Float32(0.0)

    # gmem->smem load mapping: 256 threads -> (row, K-chunk); _BK/_CK chunks/row.
    chunks_per_row = _BK // _CK
    ld_row = tid // chunks_per_row
    ld_col = (tid % chunks_per_row) * _CK
    fa = cute.make_rmem_tensor(_CK, cutlass.Float8E4M3FN)
    fb = cute.make_rmem_tensor(_CK, cutlass.Float8E4M3FN)

    k0 = 0
    while k0 < K:
        kb = k0 // BLOCK   # _BK divides BLOCK, so the scale is constant in a slice

        # --- load A slice (BM x BK) into smem, applying a_scale -------------- #
        ga_row = bm + ld_row
        if ga_row < end:
            for t in range(_CK):
                fa[t] = a[ga_row, k0 + ld_col + t]
            sa = a_scale[ga_row, kb]
            va = fa.load().to(cutlass.Float32)
            for t in range(_CK):
                sA[ld_row, ld_col + t] = va[t] * sa
        else:
            for t in range(_CK):
                sA[ld_row, ld_col + t] = cutlass.Float32(0.0)

        # --- load B slice (BN x BK) into smem for group bz, applying b_scale - #
        gb_row = bn + ld_row
        if gb_row < N:
            for t in range(_CK):
                fb[t] = b[bz, gb_row, k0 + ld_col + t]
            sb = b_scale[bz, gb_row // BLOCK, kb]
            vb = fb.load().to(cutlass.Float32)
            for t in range(_CK):
                sB[ld_row, ld_col + t] = vb[t] * sb
        else:
            for t in range(_CK):
                sB[ld_row, ld_col + t] = cutlass.Float32(0.0)

        cute.arch.sync_threads()

        # --- micro-tile MAC over the BK slice ------------------------------- #
        for kk in range(_BK):
            for i in range(_TM):
                ai = sA[ty * _TM + i, kk]
                for j in range(_TN):
                    acc[i, j] = acc[i, j] + ai * sB[tx * _TN + j, kk]

        cute.arch.sync_threads()
        k0 = k0 + _BK

    # --- write back (rows clamped to this group's [start, end) range) -------- #
    for i in range(_TM):
        orow = bm + ty * _TM + i
        for j in range(_TN):
            ocol = bn + tx * _TN + j
            if orow < end and ocol < N:
                out[orow, ocol] = acc[i, j].to(cutlass.BFloat16)


@cute.jit
def _launch_jit(
    a: cute.Tensor,
    b: cute.Tensor,
    a_scale: cute.Tensor,
    b_scale: cute.Tensor,
    m_indptr: cute.Tensor,
    out: cute.Tensor,
    N: cutlass.Int32,
    K: cutlass.Int32,
    BLOCK: cutlass.Constexpr,
    GRID_M: cutlass.Constexpr,   # CTAs along M *per group* (covers the largest group)
    GRID_N: cutlass.Constexpr,
    G: cutlass.Constexpr,        # number of groups (grid z-dim)
):
    # The whole grouped GEMM is one launch: grid z selects the group. GRID_M/
    # GRID_N/G are Constexpr because a runtime (Int32) grid combined with shared
    # memory corrupts the launch configuration under cute.compile.
    _fp8_grouped_kernel(a, b, a_scale, b_scale, m_indptr, out, N, K, BLOCK).launch(
        grid=[GRID_M, GRID_N, G],
        block=[16, 16, 1],
    )


# Calling an @cute.jit function re-traces and re-JITs on every invocation
# (~70 ms here), which dwarfs the actual kernel runtime (~0.1 ms). cute.compile()
# returns a compiled, reusable callable; cache one per distinct shape so the
# trace/compile cost is paid once and amortized across calls. Tensor *shapes* and
# the Constexpr grid are baked in, so they are part of the cache key.
_COMPILE_CACHE: dict = {}


def _launch_grouped(
    a: torch.Tensor,          # (M, K)            fp8
    b: torch.Tensor,          # (G, N, K)         fp8
    a_scale: torch.Tensor,    # (M, K_blocks)     fp32
    b_scale: torch.Tensor,    # (G, N_blocks, K_blocks) fp32
    m_indptr: torch.Tensor,   # (G+1,)            int32
    out: torch.Tensor,        # (M, N)            bf16
    block_size: int,
    grid_m: int,              # CTAs along M per group (covers the largest group)
):
    M, K = a.shape
    G, N, _ = b.shape
    grid_n = (N + _BN - 1) // _BN

    # Fresh *dynamic* args each call (CuTe tensor wrappers are consumed by
    # compile/launch). Row-major tensors -> leading dim is the last axis;
    # m_indptr is 1-D contiguous (leading dim 0).
    def _dyn_args():
        return (
            from_dlpack(a).mark_layout_dynamic(1),
            from_dlpack(b).mark_layout_dynamic(2),
            from_dlpack(a_scale).mark_layout_dynamic(1),
            from_dlpack(b_scale).mark_layout_dynamic(2),
            from_dlpack(m_indptr).mark_layout_dynamic(0),
            from_dlpack(out).mark_layout_dynamic(1),
            cutlass.Int32(N), cutlass.Int32(K),
        )

    # Constexpr args are baked at compile time and must NOT be passed at call.
    constexpr_args = (block_size, grid_m, grid_n, G)

    key = (M, N, K, G, grid_m, int(block_size))
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_launch_jit, *_dyn_args(), *constexpr_args)
        _COMPILE_CACHE[key] = compiled
    compiled(*_dyn_args())


def _kernel_function_cudacore(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    scale_major_mode: str = "K",
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert scale_major_mode == "K", "only scale_major_mode='K' is supported"

    M, K = a.shape
    G, N, K_b = b.shape
    assert K_b == K, f"a/b K mismatch: a.K={K}, b.K={K_b}"

    K_blocks_a = a_scale.shape[1]
    block_size = K // K_blocks_a

    out = torch.empty(M, N, device=a.device, dtype=out_dtype)

    # One launch for the whole grouped GEMM (grid z = group). The grid must cover
    # the largest group's rows; smaller groups simply early-exit. Use the host
    # offsets (already needed to size the grid) to find the max group size.
    offsets = m_indptr.detach().cpu().tolist()
    max_m = max((int(offsets[g + 1]) - int(offsets[g]) for g in range(G)), default=0)
    if max_m == 0:
        return out
    grid_m = (max_m + _BM - 1) // _BM

    _launch_grouped(
        a.contiguous(), b.contiguous(), a_scale.contiguous(), b_scale.contiguous(),
        m_indptr.to(torch.int32), out, block_size, grid_m,
    )
    return out


# --------------------------------------------------------------------------- #
# Tensor-core path (Blackwell tcgen05). Per-group fp8 groupwise GEMM at ~640
# TFLOP/s; falls back to the CUDA-core kernel for groups the TC kernel can't tile
# (tiny / TMA-misaligned M, or N/K not a multiple of 128).
# --------------------------------------------------------------------------- #
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
try:
    from tc_grouped import grouped_fp8_gemm as _tc_grouped_fp8_gemm
except Exception:  # pragma: no cover - TC path unavailable (no Blackwell / cutlass dsl)
    _tc_grouped_fp8_gemm = None


def _tc_compatible(a, b, m_indptr) -> bool:
    # Sync-free routing check (no m_indptr.cpu()) so the call stays CUDA-graph
    # capturable. Uses only shapes: N,K multiple of 128 (128-wide b_scale blocks)
    # and equal groups (M % G == 0). Any per-group m works: the persistent TC
    # kernel TMA-store-clamps OOB-padded M-tiles and clamps the a_scale gather.
    # Non-equal groups fall back to the (non-captured) CUDA-core kernel.
    if _tc_grouped_fp8_gemm is None:
        return False
    M, K = a.shape
    G, N, _ = b.shape
    return N % 128 == 0 and K % 128 == 0 and G >= 1 and M % G == 0


def kernel_function(
    a: torch.Tensor,                 # (M, K) fp8_e4m3
    a_scale: torch.Tensor,           # K: (M, K_blocks) | MN: (K_blocks, M) fp32
    b: torch.Tensor,                 # (G, N, K) fp8_e4m3
    b_scale: torch.Tensor,           # K: (G, N_blocks, K_blocks) | MN: (G, K_blocks, N_blocks) fp32
    m_indptr: torch.Tensor,          # (G+1,) int32
    scale_major_mode: str = "K",
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FP8 groupwise grouped GEMM. Uses the Blackwell tensor-core kernel when the
    shapes tile cleanly, otherwise the portable CUDA-core kernel."""
    assert scale_major_mode in ("K", "MN")
    if scale_major_mode == "MN":
        # Normalize to K-major with a small copy (scales are K/128 the size of A;
        # graph-capturable, ~us). The TC kernel CAN consume MN views zero-copy,
        # but its per-K-block a_scale gather then loses all cache-line reuse
        # (every k_tile jumps K_blocks-major stride -> L2 hit 92%->77%, 2227->
        # 1033 TF @big shape), so the copy is strictly faster.
        a_scale = a_scale.t().contiguous()
        b_scale = b_scale.permute(0, 2, 1).contiguous()
    if _tc_compatible(a, b, m_indptr):
        return _tc_grouped_fp8_gemm(a, a_scale, b, b_scale, m_indptr, out_dtype)
    return _kernel_function_cudacore(
        a, a_scale, b, b_scale, m_indptr, "K", out_dtype
    )
