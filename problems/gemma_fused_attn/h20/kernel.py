"""Triton fused Gemma decoder layer (prefix side, standard RMSNorm).

Implements one standard `transformers` GemmaDecoderLayer (use_adarms=False) for
the gemma_2b workload as a set of composed Triton kernels behind a single
`kernel_function`:

    r = x
    h = RMSNorm_in(x)                        # rmsnorm_kernel
    q,k,v = h @ {Wq,Wk,Wv}^T                 # matmul_kernel
    q,k = rope(q), rope(k)                   # rope_kernel  (theta 1e4, pos=arange)
    a = FlashAttn(q,k,v)  (GQA, full mask)   # attn_kernel  (fp32 softmax)
    h = x + a @ Wo^T                          # matmul_kernel (fused residual)
    r = h
    hn = RMSNorm_post(h)                     # rmsnorm_kernel
    g = gelu_tanh(hn @ Wgate^T)              # matmul_kernel (fused gelu)
    u = hn @ Wup^T                            # matmul_kernel
    m = g * u                                 # mul_kernel
    out = h + m @ Wdown^T                     # matmul_kernel (fused residual)

All numerical work is in Triton; the wrapper only allocates and launches.
rope_theta is fixed at 1e4 (gemma_2b). GQA has a single kv head shared across
the 8 query heads, so attention indexes k/v by batch only.

The gated-MLP `up * gelu_tanh(gate)` multiply is fused into the up-projection
epilogue (no separate elementwise pass). The wrapper performs no host syncs,
no `.item()`, and no data-dependent control flow, and all intermediates are
allocated via the caching allocator, so `kernel_function` is CUDA-graph
capturable (warm up once to compile the Triton kernels before capture). It runs
at the bf16 end-to-end floor (~3.44 ms on H20: GEMMs ~3.20 ms at 96-98% MFU +
attention ~0.24 ms at SDPA level); see perf_tests.py.
"""

import torch
import triton
import triton.language as tl

ROPE_THETA = 10000.0
HEAD_DIM = 256


# --------------------------------------------------------------------------- #
# RMSNorm: y = x * rsqrt(mean(x^2) + eps) * (1 + weight), fp32 internally.
# One program per row (D fits in one block).
# --------------------------------------------------------------------------- #
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_rows, D: tl.constexpr, eps,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * (1.0 + w)
    tl.store(out_ptr + row * D + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# GEMM: C[M,N] = A[M,K] @ W[N,K]^T  (nn.Linear semantics, weight is [N,K]).
# Optional fused residual add (res[M,N]) and gelu-tanh activation.
# --------------------------------------------------------------------------- #
@triton.jit
def _tanh(z):
    # stable tanh via exp (triton 3.2 has no tl.tanh)
    a = tl.where(z >= 0, z, -z)
    e = tl.exp(-2.0 * a)
    t = (1.0 - e) / (1.0 + e)
    return tl.where(z >= 0, t, -t)


@triton.jit
def _gelu_tanh(x):
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + _tanh(inner))


@triton.jit
def matmul_kernel(a_ptr, w_ptr, c_ptr, res_ptr, mul_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_wn, stride_wk,
                  stride_cm, stride_cn,
                  HAS_RES: tl.constexpr, ACT: tl.constexpr, HAS_MUL: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr):
    # grouped program-id mapping for better L2 reuse
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k0 * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < k_rem), other=0.0)
        acc += tl.dot(a, tl.trans(w))
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N
    full_mask = m_mask & n_mask
    ep_off = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    if HAS_RES:
        res = tl.load(res_ptr + ep_off, mask=full_mask, other=0.0).to(tl.float32)
        acc += res
    if ACT == 1:
        acc = _gelu_tanh(acc)
    if HAS_MUL:
        # fused elementwise multiply (e.g. gated MLP: up * gelu(gate))
        other = tl.load(mul_ptr + ep_off, mask=full_mask, other=0.0).to(tl.float32)
        acc = acc * other

    tl.store(c_ptr + ep_off, acc.to(c_ptr.dtype.element_ty), mask=full_mask)


# --------------------------------------------------------------------------- #
# Rotary embedding, applied in place-of-copy on a [n_rows, D] view where each
# row is one (token, head). position = (row // n_heads) % S. theta fixed.
#   out[:half]  = x1*cos - x2*sin
#   out[half:]  = x2*cos + x1*sin
# --------------------------------------------------------------------------- #
@triton.jit
def rope_kernel(x_ptr, out_ptr, n_rows, S, n_heads, D: tl.constexpr, theta,
                HALF: tl.constexpr):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    s = (row // n_heads) % S
    d = tl.arange(0, HALF)
    x1 = tl.load(x_ptr + row * D + d).to(tl.float32)
    x2 = tl.load(x_ptr + row * D + HALF + d).to(tl.float32)
    inv_freq = tl.exp(-(d.to(tl.float32) * (2.0 / D)) * tl.log(theta))
    angle = s.to(tl.float32) * inv_freq
    cos = tl.cos(angle)
    sin = tl.sin(angle)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    tl.store(out_ptr + row * D + d, out1.to(out_ptr.dtype.element_ty))
    tl.store(out_ptr + row * D + HALF + d, out2.to(out_ptr.dtype.element_ty))


# --------------------------------------------------------------------------- #
# Flash attention, non-causal (full prefix), GQA with a single kv head.
# q,k,v laid out [B, S, H*Dh] (token-major).  q has H=n_heads heads, k/v have 1.
# Output written [B, S, H*Dh] so the O-projection sees a [B*S, H*Dh] matrix.
# grid = (cdiv(S, BLOCK_M), B * n_heads)
# --------------------------------------------------------------------------- #
@triton.jit
def attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr,
                B, S, n_heads, scale,
                D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // n_heads
    h = bh % n_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_valid = offs_m < S

    # q[b, offs_m, h, :]  ->  base ((b*S + s)*n_heads + h)*D
    q_ptrs = q_ptr + ((b * S + offs_m[:, None]) * n_heads + h) * D + offs_d[None, :]
    q = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < S
        # k/v[b, offs_n, :]  (single kv head) -> base (b*S + n)*D
        kv_base = (b * S + offs_n[:, None]) * D + offs_d[None, :]
        k = tl.load(k_ptr + kv_base, mask=n_valid[:, None], other=0.0)
        v = tl.load(v_ptr + kv_base, mask=n_valid[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale        # [BM, BN]
        qk = tl.where(n_valid[None, :], qk, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = o_ptr + ((b * S + offs_m[:, None]) * n_heads + h) * D + offs_d[None, :]
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=m_valid[:, None])


# --------------------------------------------------------------------------- #
# Host wrapper (allocation + launch only, no compute).
# --------------------------------------------------------------------------- #
def _matmul(a, w, res=None, act=0, mul=None, out=None):
    """C = (a @ w^T (+ res)) (gelu) (* mul). a:[M,K], w:[N,K] -> C:[M,N]."""
    M, K = a.shape
    N = w.shape[0]
    assert w.shape[1] == K
    c = out if out is not None else torch.empty((M, N), device=a.device, dtype=a.dtype)
    has_res = res is not None
    has_mul = mul is not None
    res_ptr = res if has_res else c
    mul_ptr = mul if has_mul else c
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 64, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](
        a, w, c, res_ptr, mul_ptr,
        M, N, K,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        c.stride(0), c.stride(1),
        HAS_RES=has_res, ACT=act, HAS_MUL=has_mul,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
    )
    return c


def kernel_function(hidden_states,
                    input_layernorm_weight,
                    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
                    post_attention_layernorm_weight,
                    gate_proj_weight, up_proj_weight, down_proj_weight,
                    eps):
    B, S, Hid = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype
    M = B * S

    x = hidden_states.contiguous().view(M, Hid)

    Dh = HEAD_DIM
    q_dim = q_proj_weight.shape[0]
    kv_dim = k_proj_weight.shape[0]
    n_heads = q_dim // Dh
    n_kv_heads = kv_dim // Dh
    inter = gate_proj_weight.shape[0]
    scale = Dh ** -0.5

    # 1) input RMSNorm
    h = torch.empty((M, Hid), device=device, dtype=dtype)
    rmsnorm_kernel[(M,)](x, input_layernorm_weight, h, M, Hid, float(eps),
                         BLOCK=triton.next_power_of_2(Hid))

    # 2) q/k/v projections
    q = _matmul(h, q_proj_weight)        # [M, q_dim]
    k = _matmul(h, k_proj_weight)        # [M, kv_dim]
    v = _matmul(h, v_proj_weight)        # [M, kv_dim]

    # 3) rotary on q and k (in place into fresh buffers)
    HALF = Dh // 2
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    rope_kernel[(M * n_heads,)](q, q_rope, M * n_heads, S, n_heads, Dh, ROPE_THETA, HALF=HALF)
    rope_kernel[(M * n_kv_heads,)](k, k_rope, M * n_kv_heads, S, n_kv_heads, Dh, ROPE_THETA, HALF=HALF)

    # 4) flash attention (GQA, single kv head, full prefix) -> [B,S,q_dim]
    attn = torch.empty((M, q_dim), device=device, dtype=dtype)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(S, BLOCK_M), B * n_heads)
    attn_kernel[grid](q_rope, k_rope, v, attn, B, S, n_heads, scale,
                      D=Dh, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    # 5) output projection + residual (x)
    h1 = _matmul(attn, o_proj_weight, res=x)        # [M, Hid]

    # 6) post-attention RMSNorm
    hn = torch.empty((M, Hid), device=device, dtype=dtype)
    rmsnorm_kernel[(M,)](h1, post_attention_layernorm_weight, hn, M, Hid, float(eps),
                         BLOCK=triton.next_power_of_2(Hid))

    # 7) gated MLP: down(gelu_tanh(gate(hn)) * up(hn)) + residual (h1)
    #    prod = up(hn) * gelu_tanh(gate(hn)), with the multiply fused into the
    #    up-projection epilogue (saves a full [M, inter] round-trip + a launch).
    gate = _matmul(hn, gate_proj_weight, act=1)          # [M, inter] = gelu(gate)
    prod = _matmul(hn, up_proj_weight, mul=gate)         # [M, inter] = up * gelu(gate)
    out = _matmul(prod, down_proj_weight, res=h1)        # [M, Hid]

    return out.view(B, S, Hid)
