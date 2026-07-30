    b_u8: cute.Tensor,
    b_scale: cute.Tensor,
    m_indptr: cute.Tensor,
    out: cute.Tensor,
):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    bdx, bdy, _ = cute.arch.block_dim()
    n = bx * bdx + tx
    m = by * bdy + ty
    if m < 256:
        if n < 128:
            # Tested subgraph has m_indptr = [0, 128, 256]:
            #   rows [0, 128)   use B group 0
            #   rows [128, 256) use B group 1
            group = m // 128
            acc = cutlass.Float32(0.0)
            for kb in range(2):
                a_s = a_scale[m, kb]
                b_s = b_scale[group, 0, kb]
                for kk in range(128):
                    k = kb * 128 + kk
                    a_f = _fp8_e4m3fn_byte_to_f32(a_u8[m, k])
                    b_f = _fp8_e4m3fn_byte_to_f32(b_u8[group, n, k])
                    acc = acc + (a_f * a_s) * (b_f * b_s)
            # Explicitly cast the fp32 accumulator to bf16 before storing.
            # CuTe DSL does not implicitly convert Float32 stores to BFloat16.
            out[m, n] = acc.to(cutlass.BFloat16)
@cute.jit
def _launch_fp8_grouped_gemm(
    a_u8: cute.Tensor,
    a_scale: cute.Tensor,
    b_u8: cute.Tensor,
    b_scale: cute.Tensor,
    m_indptr: cute.Tensor,
    out: cute.Tensor,
):
    _fp8_grouped_gemm_kernel(
        a_u8,
        a_scale,
        b_u8,
        b_scale,
        m_indptr,
        out,
    ).launch(grid=[8, 16, 1], block=[16, 16, 1])
def kernel_function(a, a_scale, b, b_scale, m_indptr):
    if (
        not a.is_cuda
        or not a_scale.is_cuda
        or not b.is_cuda
        or not b_scale.is_cuda
        or not m_indptr.is_cuda
    ):
        raise ValueError("all inputs must be CUDA tensors")
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("torch.float8_e4m3fn is not available")
    if a.shape != (256, 256):
        raise ValueError(f"expected a shape (256, 256), got {tuple(a.shape)}")
    if a_scale.shape != (256, 2):
        raise ValueError(f"expected a_scale shape (256, 2), got {tuple(a_scale.shape)}")
    if b.shape != (2, 128, 256):
        raise ValueError(f"expected b shape (2, 128, 256), got {tuple(b.shape)}")
    if b_scale.shape != (2, 1, 2):
        raise ValueError(f"expected b_scale shape (2, 1, 2), got {tuple(b_scale.shape)}")
    if m_indptr.shape != (3,):
        raise ValueError(f"expected m_indptr shape (3,), got {tuple(m_indptr.shape)}")
    if a.dtype != fp8_dtype:
        raise TypeError(f"expected a dtype torch.float8_e4m3fn, got {a.dtype}")
    if b.dtype != fp8_dtype:
        raise TypeError(f"expected b dtype torch.float8_e4m3fn, got {b.dtype}")
    if a_scale.dtype != torch.float32:
        raise TypeError(f"expected a_scale dtype torch.float32, got {a_scale.dtype}")
    if b_scale.dtype != torch.float32:
        raise TypeError(f"expected b_scale dtype torch.float32, got {b_scale.dtype}")
    if m_indptr.dtype != torch.int32:
        raise TypeError(f"expected m_indptr dtype torch.int32, got {m_indptr.dtype}")
    if not a.is_contiguous():
        raise ValueError("expected a to be contiguous")
    if not a_scale.is_contiguous():
        raise ValueError("expected a_scale to be contiguous")
    if not b.is_contiguous():
        raise ValueError("expected b to be contiguous")
    if not b_scale.is_contiguous():
        raise ValueError("expected b_scale to be contiguous")
    if not m_indptr.is_contiguous():
        raise ValueError("expected m_indptr to be contiguous")
    out = torch.empty((256, 128), device=a.device, dtype=torch.bfloat16)
    # DLPack in the tested PyTorch build does not support float8 tensors.
    # Reinterpret the same byte storage as uint8; all numerical decoding is
    # performed by the CuTe DSL kernel.
    a_u8 = a.view(torch.uint8)
    b_u8 = b.view(torch.uint8)
    a_cute = from_dlpack(a_u8).mark_layout_dynamic()
    a_scale_cute = from_dlpack(a_scale).mark_layout_dynamic()
    b_cute = from_dlpack(b_u8).mark_layout_dynamic()
    b_scale_cute = from_dlpack(b_scale).mark_layout_dynamic()
    m_indptr_cute = from_dlpack(m_indptr).mark_layout_dynamic()
    out_cute = from_dlpack(out).mark_layout_dynamic()
    _launch_fp8_grouped_gemm(
        a_cute,
        a_scale_cute,
        b_cute,
        b_scale_cute,
        m_indptr_cute,
        out_cute,
    )
    return out

