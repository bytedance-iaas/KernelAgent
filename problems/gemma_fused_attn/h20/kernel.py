"""Triton fused Gemma decoder layer (Pi0.5) — prefix (standard RMSNorm) AND
suffix / action-expert (adaRMS) variants, with an arbitrary additive attention
mask and explicit RoPE positions.

One standard `transformers` GemmaDecoderLayer, fused into a small set of Triton
kernels behind `kernel_function`:

    r = x
    h, g1 = Norm_in(x, cond)                  # rmsnorm_kernel  (adaRMS if cond)
    q,k,v = h @ {Wq,Wk,Wv}^T                   # matmul_kernel
    q,k = rope(q,k, position_ids)              # rope_kernel   (theta 1e4)
    a = FlashAttn(q,k,v, mask)                 # attn_kernel   (fp32 softmax, +bias)
    h = r + (a @ Wo^T) * g1                     # matmul_kernel (gated residual)
    r = h
    h, g2 = Norm_post(h, cond)                 # rmsnorm_kernel
    g = gelu_tanh(h @ Wgate^T)                  # matmul_kernel (fused gelu)
    m = (h @ Wup^T) * g                          # matmul_kernel (fused mul)
    out = r + (m @ Wdown^T) * g2                # matmul_kernel (gated residual)

Two operating modes, selected purely by whether `adarms_cond` is given:

* **prefix** (`adarms_cond=None`): standard RMSNorm `normed*(1+weight)`, plain
  residual add. This is the gemma_2b VLM prefix layer.
* **suffix / action-expert** (`adarms_cond` is a `[B, cond_dim]` tensor):
  adaptive RMSNorm — a `Linear(cond_dim, 3*hidden)` per norm produces
  `(scale, shift, gate)`; the norm becomes `normed*(1+scale)+shift` (no
  `weight`) and the residual is gated `r + y*gate`. This is the gemma_300m
  action-expert layer the PPO actor's denoise recompute runs.

The attention core always honours the real additive mask (`[B,1,Sq,Sk]`, 0 where
attended / large-negative where masked), added to the fp32 scores before the
softmax — exactly like the eager path — so block-diagonal / prefix-LM masks are
correct, not just the all-zero full-prefix case. RoPE uses the supplied
`position_ids` (`cumsum(pad)-1`, offset for the suffix); it falls back to
`arange(S)` only when positions are omitted.

The wrapper allocates and launches only (no host syncs, `.item()`, or
data-dependent control flow) so `kernel_function` is CUDA-graph capturable; the
adaRMS `cond->3*hidden` projections are tiny `torch` linears (a few
microseconds, also capturable). See perf_tests.py for the roofline; on the
prefix workload the kernel runs at the bf16 end-to-end floor (~3.44 ms on H20).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

ROPE_THETA = 10000.0
HEAD_DIM = 256


# --------------------------------------------------------------------------- #
# RMSNorm: y = normed * (1 + weight)                         [standard]
#      or  y = normed * (1 + scale[b]) + shift[b]            [adaRMS, HAS_MOD]
# where normed = x * rsqrt(mean(x^2) + eps), fp32 internally.
# scale/shift are per-batch [B, D]; row r maps to batch r // S.
# One program per row (D fits in one block).
# --------------------------------------------------------------------------- #
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, scale_ptr, shift_ptr, out_ptr,
                   n_rows, S, D: tl.constexpr, eps,
                   HAS_MOD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = x * rstd
    if HAS_MOD:
        b = row // S
        scale = tl.load(scale_ptr + b * D + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + b * D + cols, mask=mask, other=0.0).to(tl.float32)
        y = normed * (1.0 + scale) + shift
    else:
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = normed * (1.0 + w)
    tl.store(out_ptr + row * D + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# GEMM: C[M,N] = A[M,K] @ W[N,K]^T  (nn.Linear semantics, weight is [N,K]).
# Fused epilogues (applied in this order):
#   ACT==1     : gelu-tanh activation
#   HAS_MUL    : elementwise multiply by mul[M,N]        (gated MLP: up*gelu(gate))
#   HAS_GATE   : multiply by gate[b, :] broadcast over seq (adaRMS residual gate)
#   HAS_RES    : add residual res[M,N]                    (== r + gate*y when both)
# gate is per-batch [B, N]; row r maps to batch r // S.
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
def matmul_kernel(a_ptr, w_ptr, c_ptr, res_ptr, mul_ptr, gate_ptr,
                  M, N, K, S,
                  stride_am, stride_ak,
                  stride_wn, stride_wk,
                  stride_cm, stride_cn,
                  HAS_RES: tl.constexpr, ACT: tl.constexpr, HAS_MUL: tl.constexpr,
                  HAS_GATE: tl.constexpr,
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

    if ACT == 1:
        acc = _gelu_tanh(acc)
    if HAS_MUL:
        other = tl.load(mul_ptr + ep_off, mask=full_mask, other=0.0).to(tl.float32)
        acc = acc * other
    if HAS_GATE:
        # gate[b, n] broadcast over the sequence; b = row // S
        b_idx = offs_m // S
        g = tl.load(gate_ptr + b_idx[:, None] * N + offs_n[None, :],
                    mask=full_mask, other=0.0).to(tl.float32)
        acc = acc * g
    if HAS_RES:
        res = tl.load(res_ptr + ep_off, mask=full_mask, other=0.0).to(tl.float32)
        acc += res

    tl.store(c_ptr + ep_off, acc.to(c_ptr.dtype.element_ty), mask=full_mask)


# --------------------------------------------------------------------------- #
# Rotary embedding on a [n_rows, D] view where each row is one (token, head).
# position = position_ids[row // n_heads] if HAS_POS else (row // n_heads) % S.
#   out[:half]  = x1*cos - x2*sin
#   out[half:]  = x2*cos + x1*sin
# --------------------------------------------------------------------------- #
@triton.jit
def rope_kernel(x_ptr, out_ptr, pos_ptr, n_rows, S, n_heads, D: tl.constexpr, theta,
                HAS_POS: tl.constexpr, HALF: tl.constexpr):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    tok = row // n_heads
    if HAS_POS:
        s = tl.load(pos_ptr + tok).to(tl.float32)
    else:
        s = (tok % S).to(tl.float32)
    d = tl.arange(0, HALF)
    x1 = tl.load(x_ptr + row * D + d).to(tl.float32)
    x2 = tl.load(x_ptr + row * D + HALF + d).to(tl.float32)
    inv_freq = tl.exp(-(d.to(tl.float32) * (2.0 / D)) * tl.log(theta))
    angle = s * inv_freq
    cos = tl.cos(angle)
    sin = tl.sin(angle)
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    tl.store(out_ptr + row * D + d, out1.to(out_ptr.dtype.element_ty))
    tl.store(out_ptr + row * D + HALF + d, out2.to(out_ptr.dtype.element_ty))


# --------------------------------------------------------------------------- #
# Flash attention, GQA with a single kv head, arbitrary additive mask.
# q laid out [B, S, H*Dh]; k/v [B, S, Dh] (single kv head, indexed by batch).
# mask is [B, 1, S, S] additive bias (0 attend / -inf masked), added to fp32
# scores before the softmax.  Output written [B, S, H*Dh].
# grid = (cdiv(S, BLOCK_M), B * n_heads)
# --------------------------------------------------------------------------- #
@triton.jit
def attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, mask_ptr,
                B, S, n_heads, scale,
                HAS_MASK: tl.constexpr,
                D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // n_heads
    h = bh % n_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_valid = offs_m < S

    q_ptrs = q_ptr + ((b * S + offs_m[:, None]) * n_heads + h) * D + offs_d[None, :]
    q = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    mask_row = b * S * S + offs_m[:, None] * S  # base for [BLOCK_M, :] mask rows

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < S
        kv_base = (b * S + offs_n[:, None]) * D + offs_d[None, :]
        k = tl.load(k_ptr + kv_base, mask=n_valid[:, None], other=0.0)
        v = tl.load(v_ptr + kv_base, mask=n_valid[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale        # [BM, BN]
        if HAS_MASK:
            bias = tl.load(mask_ptr + mask_row + offs_n[None, :],
                           mask=m_valid[:, None] & n_valid[None, :], other=0.0).to(tl.float32)
            qk = qk + bias
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
def _matmul(a, w, res=None, act=0, mul=None, gate=None, S=1, out=None):
    """C = ((a @ w^T) (gelu) (* mul) (* gate[b]) (+ res)). a:[M,K], w:[N,K]."""
    M, K = a.shape
    N = w.shape[0]
    assert w.shape[1] == K
    c = out if out is not None else torch.empty((M, N), device=a.device, dtype=a.dtype)
    has_res = res is not None
    has_mul = mul is not None
    has_gate = gate is not None
    res_ptr = res if has_res else c
    mul_ptr = mul if has_mul else c
    gate_ptr = gate if has_gate else c
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 64, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](
        a, w, c, res_ptr, mul_ptr, gate_ptr,
        M, N, K, S,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        c.stride(0), c.stride(1),
        HAS_RES=has_res, ACT=act, HAS_MUL=has_mul, HAS_GATE=has_gate,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
    )
    return c


def _rmsnorm(x, weight, eps, scale=None, shift=None, S=1):
    """RMSNorm (standard weight, or adaRMS scale/shift). x:[M,D] -> [M,D]."""
    M, D = x.shape
    out = torch.empty_like(x)
    has_mod = scale is not None
    w_ptr = weight if weight is not None else x
    scale_ptr = scale if has_mod else x
    shift_ptr = shift if has_mod else x
    rmsnorm_kernel[(M,)](x, w_ptr, scale_ptr, shift_ptr, out, M, S, D, float(eps),
                         HAS_MOD=has_mod, BLOCK=triton.next_power_of_2(D))
    return out


def kernel_function(hidden_states,
                    input_layernorm_weight,
                    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
                    post_attention_layernorm_weight,
                    gate_proj_weight, up_proj_weight, down_proj_weight,
                    eps,
                    attention_mask=None,
                    position_ids=None,
                    adarms_cond=None,
                    input_dense=None,
                    post_dense=None):
    """Fused Gemma decoder layer forward.

    Extra (optional, backward-compatible) inputs beyond the prefix-layer set:
      attention_mask : [B, 1, S, S] additive bias (0 attend / -inf masked).
                       None => full non-causal prefix (all-zero mask).
      position_ids   : [B, S] int RoPE positions.  None => arange(S).
      adarms_cond    : [B, cond_dim] time-embedding condition.  When given, the
                       layer runs the adaRMS variant; input_layernorm_weight /
                       post_attention_layernorm_weight are ignored and the two
                       (weight, bias) dense projections must be supplied:
      input_dense    : (Linear(cond_dim, 3*H).weight, .bias) for Norm_in.
      post_dense     : (Linear(cond_dim, 3*H).weight, .bias) for Norm_post.
    """
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
    scale = Dh ** -0.5

    adarms = adarms_cond is not None
    if adarms:
        # tiny cond -> (scale, shift, gate) projections (torch; ~microseconds,
        # cuda-graph capturable). Each is [B, H]; broadcast over the sequence.
        cond = adarms_cond
        mod_in = F.linear(cond, input_dense[0], input_dense[1])      # [B, 3H]
        scale_in, shift_in, gate_in = (t.contiguous() for t in mod_in.chunk(3, dim=-1))
        mod_po = F.linear(cond, post_dense[0], post_dense[1])
        scale_po, shift_po, gate_po = (t.contiguous() for t in mod_po.chunk(3, dim=-1))
    else:
        scale_in = shift_in = gate_in = None
        scale_po = shift_po = gate_po = None

    # 1) input RMSNorm  (adaRMS if cond)
    h = _rmsnorm(x, input_layernorm_weight, eps, scale=scale_in, shift=shift_in, S=S)

    # 2) q/k/v projections
    q = _matmul(h, q_proj_weight)        # [M, q_dim]
    k = _matmul(h, k_proj_weight)        # [M, kv_dim]
    v = _matmul(h, v_proj_weight)        # [M, kv_dim]

    # 3) rotary on q and k (positions from position_ids, else arange)
    HALF = Dh // 2
    has_pos = position_ids is not None
    pos_flat = position_ids.contiguous().view(M).to(torch.int32) if has_pos else q  # placeholder ptr
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    rope_kernel[(M * n_heads,)](q, q_rope, pos_flat, M * n_heads, S, n_heads, Dh, ROPE_THETA,
                                HAS_POS=has_pos, HALF=HALF)
    rope_kernel[(M * n_kv_heads,)](k, k_rope, pos_flat, M * n_kv_heads, S, n_kv_heads, Dh, ROPE_THETA,
                                   HAS_POS=has_pos, HALF=HALF)

    # 4) flash attention (GQA, single kv head, arbitrary additive mask)
    attn = torch.empty((M, q_dim), device=device, dtype=dtype)
    has_mask = attention_mask is not None
    mask_ptr = attention_mask.contiguous().view(B, S, S) if has_mask else attn  # placeholder ptr
    # 64x64 fits H20 shared memory at the edge; the fp32 mask-bias tile needs
    # the extra room, so drop BLOCK_N to 32 when a mask is present.
    BLOCK_M, BLOCK_N = (64, 32) if has_mask else (64, 64)
    grid = (triton.cdiv(S, BLOCK_M), B * n_heads)
    attn_kernel[grid](q_rope, k_rope, v, attn, mask_ptr, B, S, n_heads, scale,
                      HAS_MASK=has_mask, D=Dh, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    # 5) output projection + (gated) residual (x)
    h1 = _matmul(attn, o_proj_weight, res=x, gate=gate_in, S=S)      # [M, Hid]

    # 6) post-attention RMSNorm  (adaRMS if cond)
    hn = _rmsnorm(h1, post_attention_layernorm_weight, eps, scale=scale_po, shift=shift_po, S=S)

    # 7) gated MLP: down(gelu_tanh(gate(hn)) * up(hn)) + (gated) residual (h1)
    gate = _matmul(hn, gate_proj_weight, act=1)                     # gelu(gate)
    prod = _matmul(hn, up_proj_weight, mul=gate)                    # up * gelu(gate)
    out = _matmul(prod, down_proj_weight, res=h1, gate=gate_po, S=S)

    return out.view(B, S, Hid)
