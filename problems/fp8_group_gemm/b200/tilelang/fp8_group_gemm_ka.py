import torch
import tilelang
import tilelang.language as T
from functools import lru_cache
@tilelang.jit
def _build_fp8_grouped_gemm_kernel(
    M: int,
    N: int,
    K: int,
    G: int,
    M_PER_GROUP: int,
    SCALE_BLOCK: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    threads: int,
):
    in_dtype = "float8_e4m3fn"
    scale_dtype = "float32"
    accum_dtype = "float32"
    out_dtype = "bfloat16"
    SCALE_BLOCKS = K // SCALE_BLOCK
    K_TILES_PER_SCALE = SCALE_BLOCK // BLOCK_K
    ROW_TILES_PER_GROUP = M_PER_GROUP // BLOCK_M
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        AScale: T.Tensor((M, SCALE_BLOCKS), scale_dtype),
        B: T.Tensor((G, N, K), in_dtype),
        BScale: T.Tensor((G, 1, SCALE_BLOCKS), scale_dtype),
        MIndptr: T.Tensor((G + 1,), "int32"),
        Out: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), in_dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), in_dtype)
            C_accum = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            C_part = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            group = by // ROW_TILES_PER_GROUP
            row_tile_in_group = by - group * ROW_TILES_PER_GROUP
            row_base = MIndptr[group] + row_tile_in_group * BLOCK_M
            col_base = bx * BLOCK_N
            T.clear(C_accum)
            for sb in T.serial(SCALE_BLOCKS):
                T.clear(C_part)
                for ko in T.serial(K_TILES_PER_SCALE):
                    k_base = sb * SCALE_BLOCK + ko * BLOCK_K
                    T.copy(A[row_base, k_base], A_shared)
                    T.copy(B[group, col_base, k_base], B_shared)
                    T.gemm(A_shared, B_shared, C_part, transpose_B=True)
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    C_accum[i, j] += (
                        C_part[i, j]
                        * AScale[row_base + i, sb]
                        * BScale[group, 0, sb]
                    )
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                Out[row_base + i, col_base + j] = C_accum[i, j].astype(out_dtype)
    return main
@lru_cache(maxsize=1)
def _get_kernel():
    return _build_fp8_grouped_gemm_kernel(
        256,  # M
        128,  # N
        256,  # K
        2,    # G
        128,  # M_PER_GROUP
        128,  # SCALE_BLOCK
        16,   # BLOCK_M
        16,   # BLOCK_N
        64,   # BLOCK_K
        32,   # threads: one warp, matching TileLang's 16x16 GEMM warp partition
    )
def _check_tensor(name, tensor, shape, dtype, device_type):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device.type != device_type:
        raise ValueError(f"{name} must be on {device_type}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
def kernel_function(a, a_scale, b, b_scale, m_indptr, out=None):
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("torch.float8_e4m3fn is not available")
    _check_tensor("a", a, (256, 256), fp8_dtype, "cuda")
    _check_tensor("a_scale", a_scale, (256, 2), torch.float32, "cuda")
    _check_tensor("b", b, (2, 128, 256), fp8_dtype, "cuda")
    _check_tensor("b_scale", b_scale, (2, 1, 2), torch.float32, "cuda")
    _check_tensor("m_indptr", m_indptr, (3,), torch.int32, "cuda")
    if a_scale.device != a.device:
        raise ValueError(f"a_scale must be on the same device as a, got {a_scale.device} vs {a.device}")
    if b.device != a.device:
        raise ValueError(f"b must be on the same device as a, got {b.device} vs {a.device}")
    if b_scale.device != a.device:
        raise ValueError(f"b_scale must be on the same device as a, got {b_scale.device} vs {a.device}")
    if m_indptr.device != a.device:
        raise ValueError(f"m_indptr must be on the same device as a, got {m_indptr.device} vs {a.device}")
    if out is None:
        out = torch.empty((256, 128), device=a.device, dtype=torch.bfloat16)
    else:
        _check_tensor("out", out, (256, 128), torch.bfloat16, "cuda")
        if out.device != a.device:
            raise ValueError(f"out must be on the same device as a, got {out.device} vs {a.device}")
    kernel = _get_kernel()
    kernel(a, a_scale, b, b_scale, m_indptr, out)
    return out
