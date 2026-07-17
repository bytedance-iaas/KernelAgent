---
name: mxfp8_fp8_group_gemm_contiguous
source: reference/cuda/sgl-DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_mxfp8_fp8_gemm_1d2d.cuh
kernel_symbol: deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous
custom_inputs_entrypoint: get_inputs
---

# mxfp8_fp8_group_gemm_contiguous

SM90 (Hopper) **M-grouped, contiguous** MXFP8-FP8 GEMM, the reference for
DeepGEMM's `sm90_mxfp8_fp8_gemm_1d2d_impl` kernel (`kMasked = false`,
`GemmType::MGroupedContiguous`). Both operands are `float8_e4m3` with
**MXFP8** (micro-scaling) block scales: every contiguous 32-wide chunk of
the K dimension carries its own **UE8M0** (power-of-two, `e8m0`) scale byte.

`A` is a single `[m, k]` activation matrix whose rows are partitioned into
`groups` contiguous blocks; `grouped_layout[i]` names the expert group that
row `i` belongs to. `B` holds one `[n, k]` weight matrix per group. The
kernel computes, for each row `i` in group `g = grouped_layout[i]`, the
`NT` product `dequant(A[i]) @ dequant(B[g]).T`, accumulating in FP32 and
writing `bfloat16`.

Dequantization of an FP8 value is `x.float() * 2^(e8m0_byte - 127)`, where
the UE8M0 byte is the raw biased FP32 exponent of the (power-of-two) scale,
i.e. `scale_fp32 = reinterpret_as_f32(byte << 23)`. The kernel combines the
A and B UE8M0 scales in the integer exponent domain
(`e8m0_mul_to_float`), which is bit-exact with two IEEE FP32 scale
multiplies over the normal exponent range MXFP8 amax-quantization emits, so
the pure-PyTorch reference below (a plain FP32 dequant-then-matmul) is the
kernel's accuracy contract. Because the operands are 8-bit, the tolerance
against a bf16-input matmul is loose (the DeepGEMM tests accept
`calc_diff < 0.03`).

The problem models the canonical scale layout used by the default kernel
call: **raw `uint8` UE8M0 scales with `gran_k = 32` on both A and B**. The
kernel additionally supports packed-int32 and MN-major scale layouts and
`gran_k = 128` on A; those are alternate encodings of the same UE8M0
exponents and are exercised in the ported unit test
(`test_mxfp8_fp8_group_gemm_contiguous.py`).

## Axes

| axis | type | value | description |
|------|------|-------|-------------|
| groups | var | - | Number of expert groups (one B matrix per group) |
| m_per_group | var | - | Rows of A per group (contiguous) |
| n | var | - | Output columns / rows of each B matrix |
| k | var | - | Contraction dimension (multiple of 128) |
| gran_k | const | 32 | MXFP8 scale block size along K (UE8M0 per 32 K) |
| m | expr | groups * m_per_group | Total rows of A |
| k_scale | expr | k // gran_k | Number of UE8M0 scale bytes along K |

## Inputs

| name | shape | dtype | role | description |
|------|-------|-------|------|-------------|
| a_data | [m, k] | float8_e4m3fn | input | Quantized activation A (row-major) |
| a_scale | [m, k_scale] | uint8 | input | UE8M0 (e8m0) block scales for A, one per 32 K |
| b_data | [groups, n, k] | float8_e4m3fn | input | Quantized per-group weights B |
| b_scale | [groups, n, k_scale] | uint8 | input | UE8M0 (e8m0) block scales for B, one per 32 K |
| grouped_layout | [m] | int32 | input | Per-row group id (contiguous m-grouping) |

## Outputs

| name | shape | dtype | description |
|------|-------|-------|-------------|
| d | [m, n] | bfloat16 | Grouped GEMM result, row i = dequant(A[i]) @ dequant(B[group(i)]).T |

## Reference

```python
import torch


# --- MXFP8 / UE8M0 helpers (pure PyTorch, self-contained) ---------------- #

def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round a positive fp32 scale up to the next power of two, returned as
    an fp32 value whose mantissa is zero (a pure UE8M0/e8m0 scale)."""
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def _e8m0_u8_from_fp32(sf: torch.Tensor) -> torch.Tensor:
    """Extract the biased fp32 exponent byte of a power-of-two scale."""
    return ((sf.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def _fp32_from_e8m0_u8(sf_u8: torch.Tensor) -> torch.Tensor:
    """UE8M0 byte -> fp32 scale = 2^(byte - 127) = reinterpret(byte << 23)."""
    return (sf_u8.to(torch.int32) << 23).view(torch.float32)


def _per_token_cast_to_mxfp8(x: torch.Tensor, gran_k: int = 32):
    """Quantize a [rows, k] tensor to float8_e4m3fn with per-(row, 32-K-block)
    UE8M0 scales. Mirrors DeepGEMM's per_token_cast_to_fp8(use_ue8m0=True)."""
    rows, k = x.shape
    assert k % gran_k == 0
    xv = x.float().view(rows, k // gran_k, gran_k)
    amax = xv.abs().amax(dim=2).clamp(1e-4)
    sf = _ceil_to_ue8m0(amax / 448.0)                       # fp32 power-of-two
    x_fp8 = (xv * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(rows, k)
    return x_fp8.contiguous(), _e8m0_u8_from_fp32(sf)


def _dequant(x_fp8: torch.Tensor, sf_u8: torch.Tensor, gran_k: int) -> torch.Tensor:
    """Dequantize FP8 data with UE8M0 block scales to fp32."""
    sf_fp32 = _fp32_from_e8m0_u8(sf_u8)
    group_idx = torch.arange(x_fp8.size(-1), device=x_fp8.device) // gran_k
    return x_fp8.float() * sf_fp32[..., group_idx]


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Build one workload's quantized inputs (deterministic; the harness
    seeds torch before calling this)."""
    groups = int(axes_and_scalars["groups"])
    m_per_group = int(axes_and_scalars["m_per_group"])
    n = int(axes_and_scalars["n"])
    k = int(axes_and_scalars["k"])
    gran_k = 32
    m = groups * m_per_group

    a_ref = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device=device, dtype=torch.bfloat16)

    a_data, a_scale = _per_token_cast_to_mxfp8(a_ref, gran_k)
    b_data = torch.empty((groups, n, k), device=device, dtype=torch.float8_e4m3fn)
    b_scale = torch.empty((groups, n, k // gran_k), device=device, dtype=torch.uint8)
    for g in range(groups):
        b_data[g], b_scale[g] = _per_token_cast_to_mxfp8(b_ref[g], gran_k)

    grouped_layout = torch.arange(
        groups, device=device, dtype=torch.int32
    ).repeat_interleave(m_per_group)

    return {
        "a_data": a_data,
        "a_scale": a_scale,
        "b_data": b_data,
        "b_scale": b_scale,
        "grouped_layout": grouped_layout,
    }


@torch.no_grad()
def run(a_data, a_scale, b_data, b_scale, grouped_layout):
    """SM90 MXFP8-FP8 M-grouped contiguous GEMM (NT), FP32 accumulation.

    Args:
        a_data:         [m, k]           float8_e4m3fn quantized A
        a_scale:        [m, k // 32]     uint8 UE8M0 block scales for A
        b_data:         [groups, n, k]   float8_e4m3fn quantized B
        b_scale:        [groups, n, k//32] uint8 UE8M0 block scales for B
        grouped_layout: [m]              int32 per-row group id (contiguous)
    Returns:
        d: [m, n] bfloat16, row i = dequant(A[i]) @ dequant(B[group(i)]).T
    """
    gran_k = 32
    m, k = a_data.shape
    groups, n, _ = b_data.shape

    a_deq = _dequant(a_data, a_scale, gran_k)                # [m, k] fp32
    out = torch.zeros((m, n), device=a_data.device, dtype=torch.bfloat16)
    gl = grouped_layout.to(torch.long)
    for g in range(groups):
        rows = gl == g
        if not bool(rows.any()):
            continue
        b_deq = _dequant(b_data[g], b_scale[g], gran_k)      # [n, k] fp32
        out[rows] = (a_deq[rows] @ b_deq.t()).to(torch.bfloat16)
    return out
```

## Workloads

```jsonl
{"uuid": "1b3c0a10-0001-4a00-9000-mxfp8ctg0001", "axes": {"groups": 2, "m_per_group": 128, "n": 48, "k": 128}, "inputs": {"a_data": {"type": "custom"}, "a_scale": {"type": "custom"}, "b_data": {"type": "custom"}, "b_scale": {"type": "custom"}, "grouped_layout": {"type": "custom"}}, "tolerance": {"max_atol": 0.5, "max_rtol": 0.05, "required_matched_ratio": 0.97}}
{"uuid": "1b3c0a10-0002-4a00-9000-mxfp8ctg0002", "axes": {"groups": 2, "m_per_group": 128, "n": 128, "k": 512}, "inputs": {"a_data": {"type": "custom"}, "a_scale": {"type": "custom"}, "b_data": {"type": "custom"}, "b_scale": {"type": "custom"}, "grouped_layout": {"type": "custom"}}, "tolerance": {"max_atol": 1.0, "max_rtol": 0.05, "required_matched_ratio": 0.97}}
{"uuid": "1b3c0a10-0003-4a00-9000-mxfp8ctg0003", "axes": {"groups": 4, "m_per_group": 128, "n": 1024, "k": 1024}, "inputs": {"a_data": {"type": "custom"}, "a_scale": {"type": "custom"}, "b_data": {"type": "custom"}, "b_scale": {"type": "custom"}, "grouped_layout": {"type": "custom"}}, "tolerance": {"max_atol": 2.0, "max_rtol": 0.05, "required_matched_ratio": 0.97}}
{"uuid": "1b3c0a10-0004-4a00-9000-mxfp8ctg0004", "axes": {"groups": 4, "m_per_group": 512, "n": 1024, "k": 1024}, "inputs": {"a_data": {"type": "custom"}, "a_scale": {"type": "custom"}, "b_data": {"type": "custom"}, "b_scale": {"type": "custom"}, "grouped_layout": {"type": "custom"}}, "tolerance": {"max_atol": 2.0, "max_rtol": 0.05, "required_matched_ratio": 0.97}}
```
