"""Tests for the FP8 grouped-GEMM reference against FlashInfer's kernel.

Reference impl:   fp8_group_gemm (pure-torch dequant + per-group mm)
Kernel under ref: flashinfer.gemm.group_gemm_fp8_nt_groupwise

The FlashInfer kernel is Blackwell-only (sm100+), so the cross-check tests skip
unless a suitable GPU and a working flashinfer install are present. The pure
reference path is exercised on CPU regardless.

Layout contract (matches the flashinfer docstring, default granularity
``(1, 128, 128)`` -> block size 128 along N and K):

    a        : (cum_m, k)              fp8, row-major
    b        : (G, n, k)               fp8, row-major (the _nt_ kernel reads b
                                        as (n, k) and transposes internally)
    a_scale  : (cum_m, k // 128)       fp32, K-major
               (k // 128, cum_m)       fp32, MN-major
    b_scale  : (G, n // 128, k // 128) fp32, K-major
               (G, k // 128, n // 128) fp32, MN-major
    m_indptr : (G + 1,)                int32, each entry a multiple of 4
    out      : (cum_m, n)              bf16 / fp16
"""

import pytest
import torch

from fp8_group_gemm import dequant_fp8, quant_fp8, fp8_group_gemm, Model

BLOCK = 128
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


# --------------------------------------------------------------------------- #
# Quantization helpers (groupwise, matching dequant_fp8's block semantics).
# --------------------------------------------------------------------------- #
def make_indptr(seg_lengths, device):
    """Cumulative int32 indptr; every entry is a multiple of 4 by construction."""
    assert all(s % 4 == 0 for s in seg_lengths), "segment lengths must be x4"
    offsets = torch.tensor(seg_lengths, dtype=torch.int32, device=device)
    return torch.cat([torch.zeros(1, dtype=torch.int32, device=device),
                      torch.cumsum(offsets, dim=0).to(torch.int32)])

def build_problem(seg_lengths, n, k, scale_major_mode, device, seed=0):
    """Build a full fp8 grouped-GEMM problem in the requested scale layout."""
    torch.manual_seed(seed)
    group_size = len(seg_lengths)
    cum_m = sum(seg_lengths)
    tile_size = BLOCK

    a_f = torch.randn(cum_m, k, device=device)
    b_f = torch.randn(group_size, n, k, device=device)

    if scale_major_mode == "K":
        a_scale_shape = (cum_m, k // tile_size)
        b_scale_shape = (group_size, n // tile_size, k // tile_size)
    else:
        a_scale_shape = (k // tile_size, cum_m)
        b_scale_shape = (group_size, k // tile_size, n // tile_size)

    a_q, a_scale = quant_fp8(a_f, a_scale_shape, (1, tile_size), scale_major_mode)
    b_q, b_scale = quant_fp8(b_f, b_scale_shape, (1, tile_size, tile_size), scale_major_mode)

    m_indptr = make_indptr(seg_lengths, device)
    return a_q, b_q, a_scale.float(), b_scale.float(), m_indptr


# --------------------------------------------------------------------------- #
# CPU-only tests for the reference path (no GPU / flashinfer required).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale_major_mode", ["K", "MN"])
def test_reference_writes_into_preallocated_out(scale_major_mode):
    seg_lengths = [128, 256, 128]
    n, k = 256, 256
    a, b, a_s, b_s, indptr = build_problem(seg_lengths, n, k, scale_major_mode, "cpu")
    cum_m = sum(seg_lengths)

    out = torch.empty(cum_m, n, dtype=torch.bfloat16)
    ret = fp8_group_gemm(a, b, a_s, b_s, indptr, scale_major_mode, out=out)

    assert ret is out, "result must be written into the preallocated out tensor"
    assert ret.dtype == torch.bfloat16 and tuple(ret.shape) == (cum_m, n)

    # Independent reference: dequant + plain per-group matmul.
    a_d = dequant_fp8(a, a_s, scale_major_mode)
    b_d = dequant_fp8(b, b_s, scale_major_mode)
    expected = torch.empty(cum_m, n, dtype=torch.float32)
    off = indptr.tolist()
    for gi in range(len(seg_lengths)):
        s, e = off[gi], off[gi + 1]
        expected[s:e] = a_d[s:e] @ b_d[gi].t()
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


def test_reference_handles_empty_group():
    # Middle group is empty; rows must be left to the surrounding groups only.
    seg_lengths = [128, 0, 128]
    n, k = 128, 128
    a, b, a_s, b_s, indptr = build_problem(seg_lengths, n, k, "K", "cpu")
    out = fp8_group_gemm(a, b, a_s, b_s, indptr, "K")
    assert tuple(out.shape) == (sum(seg_lengths), n)
    assert torch.isfinite(out.float()).all()


def test_model_forward_shape_and_dtype():
    seg_lengths = [128, 128]
    n, k = 128, 256
    a, b, a_s, b_s, indptr = build_problem(seg_lengths, n, k, "K", "cpu")
    model = Model(b, b_s, indptr, "K")
    y = model.forward(a, a_s)
    assert tuple(y.shape) == (sum(seg_lengths), n) and y.dtype == torch.bfloat16


# --------------------------------------------------------------------------- #
# Cross-check against the FlashInfer kernel (Blackwell GPU required).
# --------------------------------------------------------------------------- #
flashinfer = pytest.importorskip("flashinfer", reason="flashinfer not installed")


def _is_blackwell():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10


requires_blackwell = pytest.mark.skipif(
    not _is_blackwell(),
    reason="group_gemm_fp8_nt_groupwise requires a Blackwell (sm100+) GPU",
)


def assert_close_bf16(actual, expected, rel=2e-2):
    """Magnitude-independent check: relative L2 error must stay small.

    Both kernel and reference emit bf16, so per-element error is ~0.4% of the
    value. An absolute tolerance does not scale to large k (where outputs reach
    the hundreds), so compare the relative L2 norm instead.
    """
    a, e = actual.float(), expected.float()
    rel_l2 = (a - e).norm() / e.norm().clamp(min=1e-6)
    assert rel_l2 < rel, f"relative L2 error {rel_l2:.4e} exceeds {rel:.1e}"


@requires_blackwell
@pytest.mark.parametrize("scale_major_mode", ["K", "MN"])
@pytest.mark.parametrize("m", [4, 128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("n", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("k", [128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
def test_matches_flashinfer(scale_major_mode, m, n, k, group_size):
    from flashinfer.gemm import group_gemm_fp8_nt_groupwise

    device = "cuda"
    # `m` is the per-group row count; build `group_size` equal segments. Every
    # `m` value is a multiple of 4, so each segment satisfies the kernel's
    # indptr-alignment requirement.
    seg_lengths = [m] * group_size
    a, b, a_s, b_s, indptr = build_problem(
        seg_lengths, n, k, scale_major_mode, device, seed=1234
    )

    out_kernel = group_gemm_fp8_nt_groupwise(
        a, b, a_s, b_s, indptr,
        scale_granularity_mnk=(1, BLOCK, BLOCK),
        scale_major_mode=scale_major_mode,
        out_dtype=torch.bfloat16,
    )

    out_ref = fp8_group_gemm(
        a, b, a_s, b_s, indptr, scale_major_mode, out_dtype=torch.bfloat16
    )

    assert out_kernel.shape == (m * group_size, n) == out_ref.shape
    assert_close_bf16(out_kernel, out_ref)


@requires_blackwell
def test_flashinfer_writes_into_out():
    from flashinfer.gemm import group_gemm_fp8_nt_groupwise

    seg_lengths = [128, 128]
    n, k = 128, 256
    a, b, a_s, b_s, indptr = build_problem(seg_lengths, n, k, "K", "cuda")
    out = torch.empty(sum(seg_lengths), n, dtype=torch.bfloat16, device="cuda")

    ret = group_gemm_fp8_nt_groupwise(
        a, b, a_s, b_s, indptr,
        scale_major_mode="K", out=out, out_dtype=torch.bfloat16,
    )
    assert ret.data_ptr() == out.data_ptr()
    ref = fp8_group_gemm(a, b, a_s, b_s, indptr, "K", out_dtype=torch.bfloat16)
    assert_close_bf16(out, ref)


@requires_blackwell
def test_kernelbench_inputs_match_flashinfer():
    """Drive the module's get_inputs/get_init_inputs sample through both the
    Model wrapper and the FlashInfer kernel and require them to agree."""
    from flashinfer.gemm import group_gemm_fp8_nt_groupwise
    import fp8_group_gemm as mod

    torch.manual_seed(0)
    a_q, a_scale = mod.get_inputs()                                  # forward args
    b, b_sf, m_indptr, scale_major_mode, out_dtype = mod.get_init_inputs()

    # The sample inputs are built on CPU; both paths run on the GPU.
    a_q, a_scale = a_q.cuda(), a_scale.cuda()
    b, b_sf, m_indptr = b.cuda(), b_sf.cuda(), m_indptr.cuda()

    # Reference: Model preallocates out and calls fp8_group_gemm.
    model = mod.Model(b, b_sf, m_indptr, scale_major_mode, out_dtype)
    out_ref = model(a_q, a_scale)

    # FlashInfer kernel fed the identical fp8 tensors / scales / indptr.
    out_kernel = group_gemm_fp8_nt_groupwise(
        a_q, b, a_scale, b_sf, m_indptr,
        scale_granularity_mnk=(1, BLOCK, BLOCK),
        scale_major_mode=scale_major_mode,
        out_dtype=out_dtype,
    )

    assert out_ref.shape == out_kernel.shape == (mod.m * mod.g, mod.n)
    assert out_ref.dtype == out_dtype
    assert_close_bf16(out_kernel, out_ref)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
