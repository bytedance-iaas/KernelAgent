"""SM90 FP8xFP4 contiguous m-grouped GEMM (CuTe DSL).

Implements ``out[i, j] = sum_k dq8(a[i, k]) * dq4(b[g, j, k])`` for each row
``i`` belonging to expert group ``g`` (rows are grouped contiguously, mapping
given by ``grouped_layout``), where

  dq8(a[i, k])    = fp8_e4m3_to_f32(a[i, k]) * a_sf[i, k // gran_k]
  dq4(b[g, j, k]) = e2m1_decode(nibble(b_packed[g, j, k // 2], k % 2))
                    * b_sf[g, j, k // gran_k]

B is packed FP4-E2M1: two codes per int8 byte, low nibble = even k. The
16-entry code -> value table {0, .5, 1, 1.5, 2, 3, 4, 6} (negated for codes
8..15) is materialized once per CTA in shared memory and used as a LUT
during the gmem->smem load, so the MAC loop runs on plain fp32 smem tiles.

Structure follows problems/fp8_group_gemm/sm100/cutedsl (SIMT correctness
kernel: BM x BN CTA tile, 16x16 threads, TM x TN register micro-tiles).
"""

from __future__ import annotations

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# Tile / micro-tile configuration. A CTA computes a BM x BN output tile; each
# of the 16x16 = 256 threads owns a TM x TN register micro-tile, so BM = 16*TM
# and BN = 16*TN. K is streamed in BK-wide slices through shared memory.
_BM = 64
_BN = 64
_BK = 32
_TM = 4
_TN = 4
_CK = 8   # elements per thread per slice-load (>= 4, divides _BK; even)

# FP4 E2M1 decode table: code -> value. bits [S E E M]; E=0 subnormal
# {0, 0.5}, E>=1 -> (1 + 0.5*M) * 2^(E-1). Codes 8..15 are the negated
# magnitudes (code 8 is -0 == 0).
_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


@cute.kernel
def _w4a8_grouped_kernel(
    a: cute.Tensor,        # (M, K)             fp8_e4m3
    b: cute.Tensor,        # (G, N, K // 2)     int8 (packed FP4-E2M1)
    a_sf: cute.Tensor,     # (M, K // gran_k)   fp32
    b_sf: cute.Tensor,     # (G, N, K // gran_k) fp32
    m_indptr: cute.Tensor,  # (G + 1,)          int32 (group row offsets)
    out: cute.Tensor,      # (M, N)             bf16
    N: cutlass.Int32,
    K: cutlass.Int32,
    GRAN_K: cutlass.Constexpr,  # scale K-block size; multiple of _BK
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
    lut = smem.allocate_tensor(cutlass.Float32, cute.make_layout(16), byte_alignment=16)

    # Materialize the FP4-E2M1 decode LUT once per CTA (static unrolled loop;
    # range_constexpr keeps `c` a Python int inside the dynamic `if`).
    if tid == 0:
        for c in cutlass.range_constexpr(16):
            lut[c] = cutlass.Float32(_E2M1_LUT[c])

    acc = cute.make_rmem_tensor((_TM, _TN), cutlass.Float32)
    for i in range(_TM):
        for j in range(_TN):
            acc[i, j] = cutlass.Float32(0.0)

    # gmem->smem load mapping: 256 threads -> (row, K-chunk); _BK/_CK chunks/row.
    chunks_per_row = _BK // _CK
    ld_row = tid // chunks_per_row
    ld_col = (tid % chunks_per_row) * _CK
    fa = cute.make_rmem_tensor(_CK, cutlass.Float8E4M3FN)

    cute.arch.sync_threads()  # LUT visible before the first B load

    k0 = 0
    while k0 < K:
        kb = k0 // GRAN_K   # _BK divides GRAN_K, so the scale is slice-constant

        # --- load A slice (BM x BK) into smem, applying a_sf ----------------- #
        ga_row = bm + ld_row
        if ga_row < end:
            for t in range(_CK):
                fa[t] = a[ga_row, k0 + ld_col + t]
            sa = a_sf[ga_row, kb]
            va = fa.load().to(cutlass.Float32)
            for t in range(_CK):
                sA[ld_row, ld_col + t] = va[t] * sa
        else:
            for t in range(_CK):
                sA[ld_row, ld_col + t] = cutlass.Float32(0.0)

        # --- load B slice (BN x BK): unpack FP4 pairs, LUT-decode, scale ----- #
        gb_row = bn + ld_row
        if gb_row < N:
            sb = b_sf[bz, gb_row, kb]
            kbyte = (k0 + ld_col) // 2
            for t in range(_CK // 2):
                bv = b[bz, gb_row, kbyte + t].to(cutlass.Int32)
                lo = bv & 0xF          # even k (low nibble first)
                hi = (bv >> 4) & 0xF   # odd k
                sB[ld_row, ld_col + 2 * t] = lut[lo] * sb
                sB[ld_row, ld_col + 2 * t + 1] = lut[hi] * sb
        else:
            for t in range(_CK):
                sB[ld_row, ld_col + t] = cutlass.Float32(0.0)

        cute.arch.sync_threads()

        # --- micro-tile MAC over the BK slice -------------------------------- #
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
    a_sf: cute.Tensor,
    b_sf: cute.Tensor,
    m_indptr: cute.Tensor,
    out: cute.Tensor,
    N: cutlass.Int32,
    K: cutlass.Int32,
    GRAN_K: cutlass.Constexpr,
    GRID_M: cutlass.Constexpr,   # CTAs along M *per group* (covers the largest group)
    GRID_N: cutlass.Constexpr,
    G: cutlass.Constexpr,        # number of groups (grid z-dim)
):
    _w4a8_grouped_kernel(a, b, a_sf, b_sf, m_indptr, out, N, K, GRAN_K).launch(
        grid=[GRID_M, GRID_N, G],
        block=[16, 16, 1],
    )


# cute.compile() returns a reusable compiled callable; cache one per distinct
# shape so trace/compile cost (~100 ms) is paid once.
_COMPILE_CACHE: dict = {}


def _launch_grouped(
    a: torch.Tensor,          # (M, K)              fp8_e4m3
    b: torch.Tensor,          # (G, N, K // 2)      int8
    a_sf: torch.Tensor,       # (M, K_blocks)       fp32
    b_sf: torch.Tensor,       # (G, N, K_blocks)    fp32
    m_indptr: torch.Tensor,   # (G + 1,)            int32
    out: torch.Tensor,        # (M, N)              bf16
    gran_k: int,
    grid_m: int,
):
    M, K = a.shape
    G, N, _ = b.shape
    grid_n = (N + _BN - 1) // _BN

    def _dyn_args():
        return (
            from_dlpack(a).mark_layout_dynamic(1),
            from_dlpack(b).mark_layout_dynamic(2),
            from_dlpack(a_sf).mark_layout_dynamic(1),
            from_dlpack(b_sf).mark_layout_dynamic(2),
            from_dlpack(m_indptr).mark_layout_dynamic(0),
            from_dlpack(out).mark_layout_dynamic(1),
            cutlass.Int32(N), cutlass.Int32(K),
        )

    constexpr_args = (gran_k, grid_m, grid_n, G)

    key = (M, N, K, G, grid_m, gran_k)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_launch_jit, *_dyn_args(), *constexpr_args)
        _COMPILE_CACHE[key] = compiled
    compiled(*_dyn_args())


_DEFAULT_WEIGHTS = None


def _default_weights(device):
    # Benchmark-harness path: when called with only (a_fp8, a_sf), pull the
    # weight-side tensors from the problem definition (opt-workdir convention).
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


def kernel_function(
    a_fp8: torch.Tensor,          # (M, K) fp8_e4m3
    a_sf: torch.Tensor,           # (M, K // gran_k) fp32
    b_fp4: torch.Tensor = None,   # (G, N, K // 2) int8 packed FP4-E2M1
    b_sf: torch.Tensor = None,    # (G, N, K // gran_k) fp32
    grouped_layout: torch.Tensor = None,  # (M,) int32, row -> group (contiguous)
    gran_k: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Contiguous m-grouped FP8xFP4 GEMM; returns (M, N) bf16."""
    if b_fp4 is None:
        b_fp4, b_sf, grouped_layout, gran_k = _default_weights(a_fp8.device)
    # Benchmark harnesses may up-cast quantized inputs (fp8 -> bf16); coerce
    # back (exact round-trip for values that were fp8/fp32 to begin with).
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
    assert K % _BK == 0 and gran_k % _BK == 0
    assert a_sf.shape == (M, (K + gran_k - 1) // gran_k)
    assert b_sf.shape == (G, N, (K + gran_k - 1) // gran_k)

    out = torch.empty(M, N, device=a_fp8.device, dtype=out_dtype)
    if M == 0 or N == 0:
        return out

    # grouped_layout holds contiguous group ids; convert to prefix offsets.
    counts = torch.bincount(grouped_layout.to(torch.int64), minlength=G)
    m_indptr = torch.zeros(G + 1, dtype=torch.int32, device=a_fp8.device)
    m_indptr[1:] = counts.cumsum(0).to(torch.int32)

    max_m = int(counts.max().item())
    if max_m == 0:
        return out
    grid_m = (max_m + _BM - 1) // _BM

    _launch_grouped(
        a_fp8.contiguous(), b_fp4.contiguous(),
        a_sf.contiguous(), b_sf.contiguous(),
        m_indptr, out, gran_k, grid_m,
    )
    return out
