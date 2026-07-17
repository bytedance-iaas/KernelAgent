"""Cross-verification unit test for the mxfp8_fp8_group_gemm_contiguous problem.

Ported from reference/cuda/sgl-DeepGEMM/tests/test_sm90_mxfp8_fp8.py (the four
*contiguous* accuracy cases). For each case we drive BOTH:

  * the original CUTLASS/CUDA kernel
    `deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous` (the accuracy contract), and
  * the pure-PyTorch reference `run()` from problem.py (the parsed KernelBench Model),

on identical seeded inputs, and assert they agree within the original test's
tolerance (`calc_diff < 0.03`). We additionally reproduce each original test's
own bf16-matmul reference and check `run()` against it, so the Model is pinned
to the same numerics the DeepGEMM tests use.

The kernel accepts several UE8M0 scale encodings (raw uint8, packed int32,
MN-major packed int32) and gran_k in {32, 128} on A. The problem.py Model models
the canonical raw-uint8, gran_k=32 layout; this test feeds the kernel its native
layout while feeding the Model the equivalent raw-uint8 gran_k=32 exponents
(a gran_k=128 scale is expanded to four identical gran_k=32 bytes), so both sides
represent the same dequantized operands.

Standalone: `python test_mxfp8_fp8_group_gemm_contiguous.py` — runs every case,
prints per-case diffs, prints PASS and exits 0 iff all cases pass. On a non-sm_90
GPU (or no CUDA) every case is SKIPPED with the reason printed.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # problem.py
import problem

DIFF_TOL = 0.03  # the original tests' tolerance: assert calc_diff < 0.03


# ---- import the original kernel (installed deep_gemm; NOT the repo source) --- #
try:
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp8

    _HAVE_KERNEL = hasattr(deep_gemm, "m_grouped_mxfp8_fp8_gemm_nt_contiguous")
    _IMPORT_ERR = None if _HAVE_KERNEL else "deep_gemm lacks m_grouped_mxfp8_fp8_gemm_nt_contiguous"
except Exception as exc:  # pragma: no cover
    _HAVE_KERNEL = False
    _IMPORT_ERR = f"deep_gemm import failed: {exc}"


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is required for SM90 MXFP8FP8 tests"
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        return f"SM90 MXFP8FP8 tests require sm_90, got sm_{major}x"
    if not _HAVE_KERNEL:
        return _IMPORT_ERR
    return None


# ---- helpers ported verbatim from the original test file -------------------- #

def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def _cast_back_from_fp8_1d(x, sf, gran_k):
    group_idx = torch.arange(x.size(-1), device=x.device) // gran_k
    return x.float() * sf[..., group_idx]


def _e8m0_from_fp32_pow2(sf):
    return ((sf.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def _fp32_from_e8m0_u8(sf):
    return torch.bitwise_left_shift(sf.to(torch.int32), 23).contiguous().view(torch.float32)


def _pack_ue8m0_u8_to_i32(sf):
    assert sf.dtype == torch.uint8
    if sf.shape[-1] % 4 != 0:
        padded = torch.zeros((*sf.shape[:-1], ((sf.shape[-1] + 3) // 4) * 4),
                             device=sf.device, dtype=sf.dtype)
        padded[..., : sf.shape[-1]] = sf
        sf = padded
    sf_i32 = sf.contiguous().view(*sf.shape[:-1], sf.shape[-1] // 4, 4).to(torch.int32)
    return (sf_i32[..., 0]
            | torch.bitwise_left_shift(sf_i32[..., 1], 8)
            | torch.bitwise_left_shift(sf_i32[..., 2], 16)
            | torch.bitwise_left_shift(sf_i32[..., 3], 24)).contiguous()


def _pack_ue8m0_u8_to_i32_mn_major(sf):
    packed = _pack_ue8m0_u8_to_i32(sf)
    return packed.transpose(-1, -2).contiguous().transpose(-1, -2)


def _expand_gran(sf_u8, factor):
    """gran_k=128 UE8M0 bytes -> equivalent gran_k=32 bytes (repeat each 4x)."""
    return sf_u8.repeat_interleave(factor, dim=-1).contiguous()


# ---- the four ported contiguous cases --------------------------------------- #
# Each returns (name, model_out, kernel_out, test_ref) sliced to the valid rows.

def case_contiguous_e8m0_scale():
    """Ported from test_m_grouped_mxfp8_fp8_contiguous_e8m0_scale_accuracy."""
    torch.manual_seed(0)
    groups, m_per_group, n, k = 2, 128, 48, 128
    m = groups * m_per_group
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data, a_sf_fp32 = per_token_cast_to_fp8(a_ref, use_ue8m0=True, gran_k=32)
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for g in range(groups):
        b_data[g], b_sf_fp32[g] = per_token_cast_to_fp8(b_ref[g], use_ue8m0=True, gran_k=32)

    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32).repeat_interleave(m_per_group)
    a_sf_u8 = _e8m0_from_fp32_pow2(a_sf_fp32)
    b_sf_u8 = _e8m0_from_fp32_pow2(b_sf_fp32)

    # original bf16-matmul reference
    a_dequant = _cast_back_from_fp8_1d(a_data, a_sf_fp32, gran_k=32)
    test_ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for g in range(groups):
        s, e = g * m_per_group, (g + 1) * m_per_group
        b_dequant = _cast_back_from_fp8_1d(b_data[g], b_sf_fp32[g], gran_k=32)
        test_ref[s:e] = (a_dequant[s:e] @ b_dequant.t()).to(torch.bfloat16)

    # kernel (native raw-u8 gran32 layout)
    d = torch.empty_like(test_ref)
    deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous(
        (a_data, a_sf_u8), (b_data, b_sf_u8), d, grouped_layout)

    # problem.py Model (raw-u8 gran32)
    model_out = problem.run(a_data, a_sf_u8, b_data, b_sf_u8, grouped_layout)
    return "contiguous_e8m0_scale", model_out, d, test_ref


def case_contiguous_packed_int32_scale():
    """Ported from test_m_grouped_mxfp8_fp8_contiguous_packed_int32_scale_accuracy.
    A: gran_k=128 packed int32; B: gran_k=32 packed int32."""
    torch.manual_seed(0)
    groups, m_per_group, n, k = 2, 128, 48, 640
    m = groups * m_per_group
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data, a_sf_fp32 = per_token_cast_to_fp8(a_ref, use_ue8m0=True, gran_k=128)
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf_fp32 = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for g in range(groups):
        b_data[g], b_sf_fp32[g] = per_token_cast_to_fp8(b_ref[g], use_ue8m0=True, gran_k=32)

    a_exp_u8 = _e8m0_from_fp32_pow2(a_sf_fp32)    # (m, k//128)
    b_exp_u8 = _e8m0_from_fp32_pow2(b_sf_fp32)    # (groups, n, k//32)
    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32).repeat_interleave(m_per_group)

    a_dequant = _cast_back_from_fp8_1d(a_data, a_sf_fp32, gran_k=128)
    test_ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for g in range(groups):
        s, e = g * m_per_group, (g + 1) * m_per_group
        b_dequant = _cast_back_from_fp8_1d(b_data[g], b_sf_fp32[g], gran_k=32)
        test_ref[s:e] = (a_dequant[s:e] @ b_dequant.t()).to(torch.bfloat16)

    # kernel: packed int32 scales + recipes
    d = torch.empty_like(test_ref)
    deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous(
        (a_data, _pack_ue8m0_u8_to_i32(a_exp_u8)),
        (b_data, _pack_ue8m0_u8_to_i32(b_exp_u8)),
        d, grouped_layout, recipe_a=(1, 128), recipe_b=(1, 32))

    # Model: equivalent raw-u8 gran32 (expand A's gran128 bytes 4x)
    model_out = problem.run(a_data, _expand_gran(a_exp_u8, 4), b_data, b_exp_u8, grouped_layout)
    return "contiguous_packed_int32_scale", model_out, d, test_ref


def case_contiguous_deepep_normal_scale():
    """Ported from test_m_grouped_mxfp8_fp8_contiguous_deepep_normal_scale_layout_accuracy.
    A: MN-major packed int32 gran_k=128; B: raw uint8 gran_k=32; synthetic near-1.0 scales."""
    torch.manual_seed(0)
    groups, m_per_group, n, k = 3, 128, 80, 640
    m = groups * m_per_group
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)
    a_data, _ = per_token_cast_to_fp8(a_ref, use_ue8m0=True, gran_k=128)
    b_data = torch.empty((groups, n, k), device="cuda", dtype=torch.float8_e4m3fn)
    for g in range(groups):
        b_data[g], _ = per_token_cast_to_fp8(b_ref[g], use_ue8m0=True, gran_k=32)

    a_exp = (126
             + (torch.arange(m, device="cuda", dtype=torch.uint8).view(m, 1) % 2)
             + (torch.arange(k // 128, device="cuda", dtype=torch.uint8).view(1, -1) % 2))
    b_exp = (126
             + (torch.arange(groups, device="cuda", dtype=torch.uint8).view(groups, 1, 1) % 2)
             + (torch.arange(n, device="cuda", dtype=torch.uint8).view(1, n, 1) % 2)
             + (torch.arange(k // 32, device="cuda", dtype=torch.uint8).view(1, 1, -1) % 2))

    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32).repeat_interleave(m_per_group)

    a_dequant = _cast_back_from_fp8_1d(a_data, _fp32_from_e8m0_u8(a_exp), gran_k=128)
    b_scale_fp32 = _fp32_from_e8m0_u8(b_exp)
    test_ref = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for g in range(groups):
        s, e = g * m_per_group, (g + 1) * m_per_group
        b_dequant = _cast_back_from_fp8_1d(b_data[g], b_scale_fp32[g], gran_k=32)
        test_ref[s:e] = (a_dequant[s:e] @ b_dequant.t()).to(torch.bfloat16)

    # kernel: A MN-major packed int32 gran128, B raw uint8 gran32
    d = torch.empty_like(test_ref)
    deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous(
        (a_data, _pack_ue8m0_u8_to_i32_mn_major(a_exp)),
        (b_data, b_exp.contiguous()),
        d, grouped_layout, recipe_a=(1, 128), recipe_b=(1, 32))

    # Model: raw-u8 gran32 (expand A's gran128 bytes 4x); B already gran32 raw
    model_out = problem.run(a_data, _expand_gran(a_exp, 4), b_data, b_exp.contiguous(), grouped_layout)
    return "contiguous_deepep_normal_scale", model_out, d, test_ref


def case_contiguous_dense_linear_raw_u8_scale():
    """Ported from test_m_grouped_mxfp8_fp8_contiguous_dense_linear_raw_u8_scale_accuracy.
    One logical group, raw uint8 gran_k=32 on both, padded M with m_indices=-1."""
    torch.manual_seed(1)
    m, padded_m, n, k = 137, 256, 96, 640
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    a_data, _ = per_token_cast_to_fp8(a_ref, use_ue8m0=True, gran_k=32)
    b_data, _ = per_token_cast_to_fp8(b_ref, use_ue8m0=True, gran_k=32)

    a_exp = (124
             + (torch.arange(m, device="cuda", dtype=torch.uint8).view(m, 1) % 5)
             + (torch.arange(k // 32, device="cuda", dtype=torch.uint8).view(1, -1) % 3))
    b_exp = (124
             + (torch.arange(n, device="cuda", dtype=torch.uint8).view(n, 1) % 5)
             + (torch.arange(k // 32, device="cuda", dtype=torch.uint8).view(1, -1) % 3))

    kernel_a = torch.zeros((padded_m, k), device="cuda", dtype=torch.float8_e4m3fn)
    kernel_a[:m] = a_data
    kernel_a_scale = torch.zeros((padded_m, k // 32), device="cuda", dtype=torch.uint8)
    kernel_a_scale[:m] = a_exp
    kernel_b = b_data.unsqueeze(0).contiguous()
    kernel_b_scale = b_exp.unsqueeze(0).contiguous()
    m_indices = torch.full((padded_m,), -1, device="cuda", dtype=torch.int32)
    m_indices[:m] = 0

    a_dequant = _cast_back_from_fp8_1d(a_data, _fp32_from_e8m0_u8(a_exp), gran_k=32)
    b_dequant = _cast_back_from_fp8_1d(b_data, _fp32_from_e8m0_u8(b_exp), gran_k=32)
    test_ref = (a_dequant @ b_dequant.t()).to(torch.bfloat16)         # (m, n)

    d_padded = torch.empty((padded_m, n), device="cuda", dtype=torch.bfloat16)
    deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous(
        (kernel_a, kernel_a_scale), (kernel_b, kernel_b_scale),
        d_padded, m_indices, recipe_a=(1, 32), recipe_b=(1, 32))

    # Model over the full padded batch; -1 rows stay zero, compare valid rows only
    model_padded = problem.run(kernel_a, kernel_a_scale, kernel_b, kernel_b_scale, m_indices)
    return "contiguous_dense_linear_raw_u8", model_padded[:m], d_padded[:m], test_ref


CASES = [
    case_contiguous_e8m0_scale,
    case_contiguous_packed_int32_scale,
    case_contiguous_deepep_normal_scale,
    case_contiguous_dense_linear_raw_u8_scale,
]


def main() -> int:
    reason = _skip_reason()
    if reason is not None:
        for fn in CASES:
            print(f"SKIP {fn.__name__}: {reason}")
        print("SKIPPED")
        return 0

    all_ok = True
    for fn in CASES:
        name, model_out, kernel_out, test_ref = fn()
        d_mk = calc_diff(model_out, kernel_out)   # Model vs original kernel
        d_mr = calc_diff(model_out, test_ref)     # Model vs bf16-matmul reference
        d_kr = calc_diff(kernel_out, test_ref)    # kernel vs reference (sanity)
        ok = d_mk < DIFF_TOL and d_mr < DIFF_TOL and d_kr < DIFF_TOL
        all_ok &= ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: "
              f"diff(model,kernel)={d_mk:.6f}  diff(model,ref)={d_mr:.6f}  "
              f"diff(kernel,ref)={d_kr:.6f}  (tol {DIFF_TOL})")

    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
