import torch
import torch.nn as nn
from typing import Tuple

# ---------------------------------------------------------------------------
# Quantization helpers (pure-PyTorch mirrors of deep_gemm/utils/math.py).
# The source kernel consumes pre-quantized operands:
#   A: FP8 E4M3 with per-(token, 128-K-block) fp32 scales
#   B: packed FP4 E2M1 (two codes per int8 byte, low nibble first) with
#      per-(row, gran_k_b-K-block) UE8M0 (power-of-two) fp32 scales;
#      the RS kernel's default "path-B" uses gran_k_b = 32.
# ---------------------------------------------------------------------------


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    # Round scales up to the nearest power of two (UE8M0 exponent-only format).
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def per_token_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool,
                          gran_k: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous()
    return x_fp8, sf


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    # FP4 E2M1 magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                              device=x.device, dtype=ax.dtype)
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def per_token_cast_to_fp4(x: torch.Tensor, use_ue8m0: bool,
                          gran_k: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    assert n % 2 == 0
    padded_n = align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = x_amax / 6.0
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed[:, :n // 2].contiguous(), sf


def dequant_fp8(x: torch.Tensor, sf: torch.Tensor, gran_k: int = 128) -> torch.Tensor:
    # x: (..., K) fp8, sf: (..., K // gran_k) fp32 -> fp32
    group_idx = torch.arange(x.size(-1), device=x.device) // gran_k
    return x.float() * sf[..., group_idx]


def dequant_fp4(packed: torch.Tensor, sf: torch.Tensor, gran_k: int = 128) -> torch.Tensor:
    # packed: (rows, K // 2) int8 (two FP4-E2M1 codes per byte, low nibble is
    # the even element), sf: (rows, K // gran_k) fp32 -> (rows, K) fp32
    rows, half_k = packed.shape
    k = half_k * 2
    unpacked = torch.zeros((rows, k), dtype=torch.int8, device=packed.device)
    unpacked[:, 0::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    fp4_values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                              device=packed.device, dtype=torch.float)
    sign, value_idx = (unpacked & 0x08) != 0, (unpacked & 0x07).to(torch.int)
    value = fp4_values[value_idx]
    dequantized = torch.where(sign & (value_idx != 0), -value, value)
    group_idx = torch.arange(k, device=packed.device) // gran_k
    return dequantized * sf[:, group_idx]


class Model(nn.Module):
    """
    M-grouped masked FP8xFP4 GEMM (reference for DeepGEMM's SM90
    `sm90_m_grouped_fp8_fp4_gemm_masked_1d1d_fused`, launched through the
    1D2D-RS runtime): each expert group g owns a (max_m, K) FP8 activation
    slab of which only the first masked_m[g] rows are valid; those rows are
    multiplied by the expert's packed-FP4 weights after dequantizing both
    operands to fp32, producing bf16 output.

    Note: the CUDA kernel leaves rows >= masked_m[g] undefined; this
    reference zeroes them, so comparisons must be restricted to valid rows.
    """

    def __init__(self, b_fp4: torch.Tensor, b_sf: torch.Tensor,
                 gran_k_a: int = 128, gran_k_b: int = 32):
        """
        Args:
            b_fp4: int8 tensor (num_groups, N, K // 2), packed FP4-E2M1 weights.
            b_sf: fp32 tensor (num_groups, N, K // gran_k_b), power-of-two scales.
            gran_k_a: K-block scale granularity for A (the kernel requires 128).
            gran_k_b: K-block scale granularity for B (32 or 128).
        """
        super(Model, self).__init__()
        self.register_buffer('b_fp4', b_fp4)
        self.register_buffer('b_sf', b_sf)
        self.gran_k_a = gran_k_a
        self.gran_k_b = gran_k_b

    def forward(self, a_fp8: torch.Tensor, a_sf: torch.Tensor,
                masked_m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a_fp8: float8_e4m3fn tensor (num_groups, max_m, K).
            a_sf: fp32 tensor (num_groups, max_m, K // gran_k_a).
            masked_m: int32 tensor (num_groups,), valid row count per group.

        Returns:
            bf16 tensor (num_groups, max_m, N) with
            out[g, :masked_m[g]] = dequant(A[g, :masked_m[g]]) @ dequant(B[g]).T
            and zeros elsewhere.
        """
        num_groups, n, _ = self.b_fp4.shape
        max_m = a_fp8.size(1)
        out = torch.zeros((num_groups, max_m, n), dtype=torch.bfloat16,
                          device=a_fp8.device)
        for group_id in range(num_groups):
            valid_m = int(masked_m[group_id].item())
            if valid_m == 0:
                continue
            a = dequant_fp8(a_fp8[group_id, :valid_m], a_sf[group_id, :valid_m],
                            self.gran_k_a)
            b = dequant_fp4(self.b_fp4[group_id], self.b_sf[group_id], self.gran_k_b)
            out[group_id, :valid_m] = (a @ b.t()).to(torch.bfloat16)
        return out


# Workload shapes: test_sm90_fp8_fp4_masked in
# reference/cuda/sgl-DeepGEMM/tests/test_sm90_fp8_fp4.py (DSV4 MoE decode
# shape; b_gran_k=32 exercises the RS kernel's default k32 "path-B").
num_groups = 8
max_m = 128
m_per_group = 16
N = 4096
K = 7168
gran_k_a = 128
gran_k_b = 32


def get_inputs():
    # The kernel only makes sense on pre-quantized operands, so the inputs are
    # generated by quantizing bf16 activations (matches the source unit test).
    a_ref = torch.randn(num_groups, max_m, K, dtype=torch.bfloat16)
    a_fp8 = torch.empty(num_groups, max_m, K, dtype=torch.float8_e4m3fn)
    a_sf = torch.empty(num_groups, max_m, K // gran_k_a, dtype=torch.float)
    for group_id in range(num_groups):
        a_fp8[group_id], a_sf[group_id] = per_token_cast_to_fp8(
            a_ref[group_id], use_ue8m0=False, gran_k=gran_k_a)
    masked_m = torch.full((num_groups,), m_per_group, dtype=torch.int32)
    return [a_fp8, a_sf, masked_m]


def get_init_inputs():
    b_ref = torch.randn(num_groups, N, K, dtype=torch.bfloat16)
    b_fp4 = torch.empty(num_groups, N, K // 2, dtype=torch.int8)
    b_sf = torch.empty(num_groups, N, K // gran_k_b, dtype=torch.float)
    for group_id in range(num_groups):
        b_fp4[group_id], b_sf[group_id] = per_token_cast_to_fp4(
            b_ref[group_id], use_ue8m0=True, gran_k=gran_k_b)
    return [b_fp4, b_sf, gran_k_a, gran_k_b]
