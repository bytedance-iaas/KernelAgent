import torch
import torch.nn as nn
from typing import Optional

def dequant_fp8(x, x_scale, scale_major_mode):
    ndim = x.ndim
    assert ndim in [2, 3], f"x.ndim must be 2 or 3, but got {ndim}"
    assert ndim == len(x_scale.shape)

    if ndim == 2:
        if scale_major_mode == "K":
            s0, s1 = x_scale.shape
        else:
            s1, s0 = x_scale.shape

        # rearrange(x, "(s0 t0) (s1 t1) -> s0 s1 t0 t1")
        H, W = x.shape
        t0, t1 = H // s0, W // s1
        x = x.to(torch.float32).view(s0, t0, s1, t1).permute(0, 2, 1, 3)  # s0 s1 t0 t1

        # rearrange(x_scale, "s0 s1 -> s0 s1 1 1") or "s0 s1 -> s1 s0 1 1"
        if scale_major_mode == "K":
            scale = x_scale[:, :, None, None]                               # s0 s1 1  1
        else:
            scale = x_scale.T[:, :, None, None]                             # s0 s1 1  1  (s1,s0 -> transposed)

        # rearrange(x * scale, "s0 s1 t0 t1 -> (s0 t0) (s1 t1)")
        out = (
            (x * scale)                                                      # s0 s1 t0 t1
            .permute(0, 2, 1, 3)                                             # s0 t0 s1 t1
            .reshape(s0 * t0, s1 * t1)
        )

    elif ndim == 3:
        if scale_major_mode == "K":
            s0, s1, s2 = x_scale.shape
        else:
            s0, s2, s1 = x_scale.shape

        # rearrange(x, "(s0 t0) (s1 t1) (s2 t2) -> s0 s1 s2 t0 t1 t2")
        D, H, W = x.shape
        t0, t1, t2 = D // s0, H // s1, W // s2
        x = (
            x.to(torch.float32)
            .view(s0, t0, s1, t1, s2, t2)
            .permute(0, 2, 4, 1, 3, 5)                                      # s0 s1 s2 t0 t1 t2
        )

        # rearrange(x_scale, "s0 s1 s2 -> s0 s1 s2 1 1 1") or "s0 s1 s2 -> s0 s2 s1 1 1 1"
        if scale_major_mode == "K":
            scale = x_scale[:, :, :, None, None, None]                      # s0 s1 s2 1  1  1
        else:
            scale = x_scale.permute(0, 2, 1)[:, :, :, None, None, None]    # s0 s1 s2 1  1  1

        # rearrange(x * scale, "s0 s1 s2 t0 t1 t2 -> (s0 t0) (s1 t1) (s2 t2)")
        out = (
            (x * scale)                                                      # s0 s1 s2 t0 t1 t2
            .permute(0, 3, 1, 4, 2, 5)                                      # s0 t0 s1 t1 s2 t2
            .reshape(s0 * t0, s1 * t1, s2 * t2)
        )

    return out


def quant_fp8(x, scale_shape, tile_shape, scale_major_mode):
    """
    Quantizes a 2D or 3D tensor to FP8.

    Args:
        x (torch.Tensor): The 2D or 3D input tensor.
        scale_shape (tuple): The shape of the scale tensor.
        tile_shape (tuple): The shape of the tiles.
        scale_major_mode (str): The tiling order, "K" for row-major like,
                                or another value for column-major like.

    Returns:
        tuple: A tuple containing the quantized FP8 tensor and the
               calculated float32 per-tile scales.
    """
    # 1. Assertions and initial setup
    ndim = x.ndim
    assert ndim in (2, 3), f"x.ndim must be 2 or 3, but got {ndim}"
    assert ndim == len(scale_shape) == len(tile_shape)

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_amax = torch.tensor(fp8_info.max, device=x.device, dtype=torch.float32)

    def to_pow2_scale(abs_max):
        # Round each tile's scale up to the nearest power of two.
        x_scale = abs_max / fp8_amax
        return torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

    # 2. Tiling and scale calculation
    if ndim == 2:
        s0, s1 = scale_shape
        t0, t1 = tile_shape
        if scale_major_mode == "K":
            # x: (s0*t0, s1*t1) -> tiles (s0, s1, t0, t1)
            # rearrange "(s0 t0) (s1 t1) -> s0 s1 t0 t1"
            x_tiled = x.reshape(s0, t0, s1, t1).permute(0, 2, 1, 3)
            abs_max = x_tiled.abs().amax(dim=(-2, -1)).clamp(min=1e-4)
            x_scale = to_pow2_scale(abs_max)  # (s0, s1)

            # repeat "s0 s1 -> (s0 t0) (s1 t1)"  (block-repeat each scale)
            scales_repeated = (
                x_scale.reshape(s0, 1, s1, 1)
                .expand(s0, t0, s1, t1)
                .reshape(s0 * t0, s1 * t1)
            )
        else:
            # x: (s1*t0, s0*t1) -> tiles (s0, s1, t0, t1)
            # rearrange "(s1 t0) (s0 t1) -> s0 s1 t0 t1"
            x_tiled = x.reshape(s1, t0, s0, t1).permute(2, 0, 1, 3)
            abs_max = x_tiled.abs().amax(dim=(-2, -1)).clamp(min=1e-4)
            x_scale = to_pow2_scale(abs_max)  # (s0, s1)

            # transpose to (s1, s0), then repeat "s1 s0 -> (s1 t0) (s0 t1)"
            scales_repeated = (
                x_scale.permute(1, 0)
                .reshape(s1, 1, s0, 1)
                .expand(s1, t0, s0, t1)
                .reshape(s1 * t0, s0 * t1)
            )

    else:  # ndim == 3
        s0, s1, s2 = scale_shape
        t0, t1, t2 = tile_shape
        if scale_major_mode == "K":
            # x: (s0*t0, s1*t1, s2*t2) -> tiles (s0, s1, s2, t0, t1, t2)
            # rearrange "(s0 t0) (s1 t1) (s2 t2) -> s0 s1 s2 t0 t1 t2"
            x_tiled = x.reshape(s0, t0, s1, t1, s2, t2).permute(0, 2, 4, 1, 3, 5)
            abs_max = x_tiled.abs().amax(dim=(-3, -2, -1)).clamp(min=1e-4)
            x_scale = to_pow2_scale(abs_max)  # (s0, s1, s2)

            # repeat "s0 s1 s2 -> (s0 t0) (s1 t1) (s2 t2)"
            scales_repeated = (
                x_scale.reshape(s0, 1, s1, 1, s2, 1)
                .expand(s0, t0, s1, t1, s2, t2)
                .reshape(s0 * t0, s1 * t1, s2 * t2)
            )
        else:
            # x: (s0*t0, s2*t1, s1*t2) -> tiles (s0, s1, s2, t0, t1, t2)
            # rearrange "(s0 t0) (s2 t1) (s1 t2) -> s0 s1 s2 t0 t1 t2"
            x_tiled = x.reshape(s0, t0, s2, t1, s1, t2).permute(0, 4, 2, 1, 3, 5)
            abs_max = x_tiled.abs().amax(dim=(-3, -2, -1)).clamp(min=1e-4)
            x_scale = to_pow2_scale(abs_max)  # (s0, s1, s2)

            # permute to (s0, s2, s1), then
            # repeat "s0 s2 s1 -> (s0 t0) (s2 t1) (s1 t2)"
            scales_repeated = (
                x_scale.permute(0, 2, 1)
                .reshape(s0, 1, s2, 1, s1, 1)
                .expand(s0, t0, s2, t1, s1, t2)
                .reshape(s0 * t0, s2 * t1, s1 * t2)
            )

    # 3. Final quantization
    x_fp32 = x / (scales_repeated + 1e-8)
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)

    return x_fp8, x_scale

def fp8_group_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    scale_major_mode: str,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Performs a grouped GEMM: for each group ``g`` the rows of ``a`` in
    ``[m_indptr[g], m_indptr[g + 1])`` are multiplied by the weight ``b[g]``.

    Args:
        a (torch.Tensor): Quantized fp8 input of shape (M, K).
        b (torch.Tensor): Quantized fp8 weights of shape (G, N, K).
        a_scale (torch.Tensor): Dequantization scales for ``a``.
        b_scale (torch.Tensor): Dequantization scales for ``b``.
        m_indptr (torch.Tensor): Group row offsets of shape (G + 1,); a prefix
            sum so group ``g`` owns rows ``[m_indptr[g], m_indptr[g + 1])``.
        scale_major_mode (str): Scale layout, "K" or "MN".
        out (torch.Tensor, optional): Preallocated output of shape (M, N). The
            result is written here in place. Allocated if not provided.
        out_dtype (torch.dtype, optional): Output dtype. Defaults to ``out``'s
            dtype when ``out`` is given, otherwise bfloat16.

    Returns:
        torch.Tensor: The output tensor of shape (M, N) (the same object as
        ``out`` when it is provided).
    """
    # Dequantize the input tensors to float32 for the matmul.
    a_dequant = dequant_fp8(a, a_scale, scale_major_mode)
    b_dequant = dequant_fp8(b, b_scale, scale_major_mode)

    m, _ = a_dequant.shape
    group_size, n, _ = b_dequant.shape

    if out is None:
        out = torch.empty(m, n, device=a_dequant.device, dtype=out_dtype)

    # Grouped GEMM over variable-sized row segments described by m_indptr.
    offsets = m_indptr.tolist()
    for g in range(group_size):
        start, end = offsets[g], offsets[g + 1]
        if end > start:
            # out[start:end] = a[start:end] @ b[g].T ; cast handled by copy.
            out[start:end] = torch.mm(a_dequant[start:end], b_dequant[g].t()).to(out_dtype)

    return out


class Model(nn.Module):
    def __init__(
        self,
        w: torch.Tensor,
        w_scale: torch.Tensor,
        m_indptr: torch.Tensor,
        scale_major_mode: str,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        super(Model, self).__init__()
        self.w = w
        self.w_scale = w_scale
        self.m_indptr = m_indptr
        self.scale_major_mode = scale_major_mode
        self.out_dtype = out_dtype

    def forward(self, x: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
        a, a_sf = x, x_scale
        b, b_sf = self.w, self.w_scale
        
        out = torch.empty(a.size(0), b.size(1), device=a.device, dtype=self.out_dtype)
        fp8_group_gemm(a, b, a_sf, b_sf, self.m_indptr, self.scale_major_mode, out, self.out_dtype)
        return out


# Test code
m = 128
n = 256
k = 256
g = 2
scale_major_mode="K"
block_size=128
out_dtype=torch.bfloat16

def get_inputs():
    a_scale_shape = (m * g, k // block_size)
    a_f = torch.randn(m * g, k)
    a_q, a_scale = quant_fp8(a_f, a_scale_shape, (1, block_size), scale_major_mode)
    return [a_q, a_scale]

def get_init_inputs():
    b_scale_shape = (g, n // block_size, k // block_size)
    b_f = torch.randn(g, n, k)
    b, b_sf = quant_fp8(b_f, b_scale_shape, (1, block_size, block_size), scale_major_mode)
    m_indptr = torch.arange(0, g + 1, dtype=torch.int32) * m
    return [b, b_sf, m_indptr, scale_major_mode, out_dtype]
