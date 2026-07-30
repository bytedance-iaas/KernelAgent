"""Cross-expert / joint flash attention for the Pi0.5 actor path (fwd + bwd).

openpi's `PaliGemmaWithExpertModel.forward` runs the actor's joint/suffix forward
by hand (`compute_layer_complete` in gemma_pytorch.py): it norms + projects q/k/v
per expert, **concatenates q/k/v across both experts**, applies RoPE, then runs
ONE joint attention over the block-diagonal mask via
`modeling_gemma.eager_attention_forward`, and splits the result back per expert.
The whole-layer fused kernel in `kernel.py` cannot express this (its attention
owns its own within-stream K/V); this module supplies the missing primitive.

`fused_attention_forward` is a drop-in replacement for `eager_attention_forward`
— same signature, same math (fp32 softmax, additive mask, GQA repeat-kv), and it
handles the rectangular case `Sq != Sk` needed by:
  * the joint forward   (q,k,v all length prefix+suffix), and
  * the suffix recompute with a prefix KV cache (Sq = suffix, Sk = prefix+suffix).

Both the **forward and the backward are fused Triton flash kernels**: the
backward recomputes the softmax `P` on the fly from (Q, K, L=logsumexp), never
materialising the `[Sq,Sk]` attention matrix or its gradient (O(S) memory, not
O(S^2)). It produces dQ, dK, dV directly — the FlashAttention-2 backward — with
GQA handled by summing per-query-head K/V grads back onto the shared kv head.
The surrounding projection-GEMM gradients (q/k/v/o/gate/up/down) are plain
transposed matmuls that cuBLAS already runs at roofline, so they stay in torch.

Layouts follow the model: q is [B, Hq, Sq, D], k/v are [B, Hkv, Sk, D], the
additive mask is [B, 1, Sq, Sk], and the output is [B, Sq, Hq, D]. RoPE is
applied by the caller before this function, exactly like the eager path.
"""

import torch
import triton
import triton.language as tl


# =========================================================================== #
# Forward: o = softmax(q·kᵀ*scale + mask)·v, fp32 softmax; also emit L=logsumexp.
# q:[B,Hq,Sq,D], k/v:[B,Hkv,Sk,D], mask:[B,1,Sq,Sk] -> o:[B,Sq,Hq,D], L:[B,Hq,Sq]
# =========================================================================== #
@triton.jit
def _fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, mask_ptr,
                B, Hq, Hkv, Sq, Sk, scale, groups,
                HAS_MASK: tl.constexpr,
                D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // Hq
    h = bh % Hq
    kv = h // groups

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_valid = offs_m < Sq

    q_ptrs = q_ptr + ((b * Hq + h) * Sq + offs_m[:, None]) * D + offs_d[None, :]
    q = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    mask_row = b * Sq * Sk + offs_m[:, None] * Sk

    for start_n in range(0, Sk, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < Sk
        kv_base = ((b * Hkv + kv) * Sk + offs_n[:, None]) * D + offs_d[None, :]
        k = tl.load(k_ptr + kv_base, mask=n_valid[:, None], other=0.0)
        v = tl.load(v_ptr + kv_base, mask=n_valid[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
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

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    o_ptrs = o_ptr + ((b * Sq + offs_m[:, None]) * Hq + h) * D + offs_d[None, :]
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=m_valid[:, None])
    # L = m + log(l): the log-sum-exp, so P = exp(scale*q·kᵀ + mask - L) in bwd.
    L = m_i + tl.log(l_safe)
    tl.store(l_ptr + (b * Hq + h) * Sq + offs_m, L, mask=m_valid)


# =========================================================================== #
# Backward dQ: parallel over query blocks. Recompute P from (q,k,L); accumulate
#   dQ = scale * (P ∘ (dO·Vᵀ - delta)) · K.
# delta[b,h,qi] = sum_d dO*O (precomputed).  q:[B,Hq,Sq,D] do,o:[B,Sq,Hq,D]
# =========================================================================== #
@triton.jit
def _bwd_dq_kernel(q_ptr, k_ptr, v_ptr, do_ptr, l_ptr, delta_ptr, dq_ptr, mask_ptr,
                   B, Hq, Hkv, Sq, Sk, scale, groups,
                   HAS_MASK: tl.constexpr,
                   D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // Hq
    h = bh % Hq
    kv = h // groups

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_valid = offs_m < Sq

    q = tl.load(q_ptr + ((b * Hq + h) * Sq + offs_m[:, None]) * D + offs_d[None, :],
                mask=m_valid[:, None], other=0.0)
    do = tl.load(do_ptr + ((b * Sq + offs_m[:, None]) * Hq + h) * D + offs_d[None, :],
                 mask=m_valid[:, None], other=0.0)
    L = tl.load(l_ptr + (b * Hq + h) * Sq + offs_m, mask=m_valid, other=0.0)
    delta = tl.load(delta_ptr + (b * Hq + h) * Sq + offs_m, mask=m_valid, other=0.0)

    dq = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    mask_row = b * Sq * Sk + offs_m[:, None] * Sk

    for start_n in range(0, Sk, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < Sk
        kv_base = ((b * Hkv + kv) * Sk + offs_n[:, None]) * D + offs_d[None, :]
        k = tl.load(k_ptr + kv_base, mask=n_valid[:, None], other=0.0)
        v = tl.load(v_ptr + kv_base, mask=n_valid[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        if HAS_MASK:
            bias = tl.load(mask_ptr + mask_row + offs_n[None, :],
                           mask=m_valid[:, None] & n_valid[None, :], other=0.0).to(tl.float32)
            qk = qk + bias
        qk = tl.where(n_valid[None, :], qk, -float("inf"))
        p = tl.exp(qk - L[:, None])                              # [BM,BN]
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)             # [BM,BN]
        ds = p * (dp - delta[:, None])                          # softmax jacobian
        dq += tl.dot(ds.to(k.dtype), k)                         # [BM,D]

    dq = dq * scale
    tl.store(dq_ptr + ((b * Hq + h) * Sq + offs_m[:, None]) * D + offs_d[None, :],
             dq.to(dq_ptr.dtype.element_ty), mask=m_valid[:, None])


# =========================================================================== #
# Backward dK,dV: parallel over key blocks; per query head (summed to kv head in
#   the wrapper). dV = Pᵀ·dO ; dK = scale * (P ∘ (dO·Vᵀ - delta))ᵀ · Q.
# outputs dk_h,dv_h:[B,Hq,Sk,D]
# =========================================================================== #
@triton.jit
def _bwd_dkdv_kernel(q_ptr, k_ptr, v_ptr, do_ptr, l_ptr, delta_ptr,
                     dk_ptr, dv_ptr, mask_ptr,
                     B, Hq, Hkv, Sq, Sk, scale, groups,
                     HAS_MASK: tl.constexpr,
                     D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // Hq
    h = bh % Hq
    kv = h // groups

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    n_valid = offs_n < Sk

    kv_base = ((b * Hkv + kv) * Sk + offs_n[:, None]) * D + offs_d[None, :]
    k = tl.load(k_ptr + kv_base, mask=n_valid[:, None], other=0.0)
    v = tl.load(v_ptr + kv_base, mask=n_valid[:, None], other=0.0)

    dk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    for start_m in range(0, Sq, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_valid = offs_m < Sq
        q = tl.load(q_ptr + ((b * Hq + h) * Sq + offs_m[:, None]) * D + offs_d[None, :],
                    mask=m_valid[:, None], other=0.0)
        do = tl.load(do_ptr + ((b * Sq + offs_m[:, None]) * Hq + h) * D + offs_d[None, :],
                     mask=m_valid[:, None], other=0.0)
        L = tl.load(l_ptr + (b * Hq + h) * Sq + offs_m, mask=m_valid, other=0.0)
        delta = tl.load(delta_ptr + (b * Hq + h) * Sq + offs_m, mask=m_valid, other=0.0)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale          # [BM,BN]
        if HAS_MASK:
            bias = tl.load(mask_ptr + b * Sq * Sk + offs_m[:, None] * Sk + offs_n[None, :],
                           mask=m_valid[:, None] & n_valid[None, :], other=0.0).to(tl.float32)
            qk = qk + bias
        qk = tl.where(m_valid[:, None] & n_valid[None, :], qk, -float("inf"))
        p = tl.exp(qk - L[:, None])                                 # [BM,BN]
        dv += tl.dot(tl.trans(p).to(do.dtype), do)                  # [BN,D]
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)                 # [BM,BN]
        ds = p * (dp - delta[:, None])                              # [BM,BN]
        dk += tl.dot(tl.trans(ds).to(q.dtype), q)                   # [BN,D]

    dk = dk * scale
    out_base = ((b * Hq + h) * Sk + offs_n[:, None]) * D + offs_d[None, :]
    tl.store(dk_ptr + out_base, dk.to(dk_ptr.dtype.element_ty), mask=n_valid[:, None])
    tl.store(dv_ptr + out_base, dv.to(dv_ptr.dtype.element_ty), mask=n_valid[:, None])


# --------------------------------------------------------------------------- #
# Host wrappers
# --------------------------------------------------------------------------- #
def _flash_forward(q, k, v, attention_mask, scale):
    """q:[B,Hq,Sq,D], k/v:[B,Hkv,Sk,D] -> (o:[B,Sq,Hq,D], L:[B,Hq,Sq])."""
    B, Hq, Sq, D = q.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    groups = Hq // Hkv
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty((B, Sq, Hq, D), device=q.device, dtype=q.dtype)
    L = torch.empty((B, Hq, Sq), device=q.device, dtype=torch.float32)
    has_mask = attention_mask is not None
    mask_ptr = attention_mask.contiguous().view(B, Sq, Sk) if has_mask else out
    BLOCK_M, BLOCK_N = (64, 32) if has_mask else (64, 64)
    grid = (triton.cdiv(Sq, BLOCK_M), B * Hq)
    _fwd_kernel[grid](q, k, v, out, L, mask_ptr, B, Hq, Hkv, Sq, Sk, scale, groups,
                      HAS_MASK=has_mask, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out, L


def _flash_backward(q, k, v, o, L, grad_out, attention_mask, scale):
    """Fused flash backward -> (dq:[B,Hq,Sq,D], dk,dv:[B,Hkv,Sk,D])."""
    B, Hq, Sq, D = q.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    groups = Hq // Hkv
    do = grad_out.contiguous()
    # delta[b,h,qi] = sum_d dO*O  (O(S) elementwise; cheap, kept in torch)
    delta = (do.float() * o.float()).sum(-1).permute(0, 2, 1).contiguous()   # [B,Hq,Sq]
    has_mask = attention_mask is not None
    mask_ptr = attention_mask.contiguous().view(B, Sq, Sk) if has_mask else do
    # backward holds more live fp32 tiles than forward; with head_dim 256 the
    # 32x32 tiles are what fit H20 shared memory (wider tiles spill registers
    # and regress). This trades ~30% wall-time for ~2.7x less peak memory vs
    # the eager backward (no [Sq,Sk] matrix materialised).
    BM, BN = 32, 32
    dq = torch.empty((B, Hq, Sq, D), device=q.device, dtype=q.dtype)
    _bwd_dq_kernel[(triton.cdiv(Sq, BM), B * Hq)](
        q, k, v, do, L, delta, dq, mask_ptr, B, Hq, Hkv, Sq, Sk, scale, groups,
        HAS_MASK=has_mask, D=D, BLOCK_M=BM, BLOCK_N=BN, num_warps=4, num_stages=2)

    dk_h = torch.empty((B, Hq, Sk, D), device=q.device, dtype=q.dtype)
    dv_h = torch.empty((B, Hq, Sk, D), device=q.device, dtype=q.dtype)
    _bwd_dkdv_kernel[(triton.cdiv(Sk, BN), B * Hq)](
        q, k, v, do, L, delta, dk_h, dv_h, mask_ptr, B, Hq, Hkv, Sq, Sk, scale, groups,
        HAS_MASK=has_mask, D=D, BLOCK_M=BM, BLOCK_N=BN, num_warps=4, num_stages=2)

    # GQA: sum per-query-head grads back onto the shared kv head.
    dk = dk_h.view(B, Hkv, groups, Sk, D).sum(2)
    dv = dv_h.view(B, Hkv, groups, Sk, D).sum(2)
    return dq, dk, dv


def _torch_attention(q, k, v, attention_mask, scale):
    """Differentiable reference == modeling_gemma.eager_attention_forward math."""
    Hq, Hkv = q.shape[1], k.shape[1]
    groups = Hq // Hkv
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    s = torch.matmul(q, k.transpose(2, 3)) * scale
    if attention_mask is not None:
        s = s + attention_mask[:, :, :, : k.shape[-2]]
    p = torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    o = torch.matmul(p, v)
    return o.transpose(1, 2).contiguous()


class _AttnFn(torch.autograd.Function):
    """Fully-fused Triton flash attention (forward + backward).

    The backward recomputes P on the fly (O(S) memory), so peak memory is ~2.7x
    (S=1024) to ~10x (S=4096) below the eager backward, which materialises the
    [Sq,Sk] matrix and its gradient. Wall-time is faster than eager for short S
    (<=~512) and ~25% slower for long S at head_dim 256 (the 32x32 tiles forced
    by shared memory underutilise the tensor cores vs cuBLAS). Net: use it when
    attention memory is the constraint (long context / large batch that OOMs
    under eager); the compute-optimal version needs D-split tiling / autotuning.
    """

    @staticmethod
    def forward(ctx, q, k, v, attention_mask, scale):
        o, L = _flash_forward(q, k, v, attention_mask, scale)
        ctx.save_for_backward(q, k, v, o, L)
        ctx.attention_mask = attention_mask
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, o, L = ctx.saved_tensors
        dq, dk, dv = _flash_backward(q, k, v, o, L, grad_out, ctx.attention_mask, ctx.scale)
        return dq, dk, dv, None, None


def fused_attention(q, k, v, attention_mask=None, scale=None):
    """Autograd-capable cross-expert/joint attention (fused fwd + bwd).

    q:[B,Hq,Sq,D], k/v:[B,Hkv,Sk,D], mask:[B,1,Sq,Sk] additive -> [B,Sq,Hq,D].
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _AttnFn.apply(q, k, v, attention_mask, scale)


def fused_attention_forward(module, query, key, value, attention_mask,
                            scaling, dropout=0.0, **kwargs):
    """Drop-in for `transformers ... eager_attention_forward`.

    Swap into openpi `compute_layer_complete` (the joint forward) and the
    GQA attention of the suffix KV-cache path. Returns (attn_output, None) with
    attn_output = [B, Sq, Hq, D] (== eager's transpose(1,2).contiguous()).
    """
    out = fused_attention(query, key, value, attention_mask, scaling)
    return out, None


def enable_fused_attention():
    """Monkeypatch the Triton attention into openpi's Gemma stack.

    Both openpi's joint forward (`compute_layer_complete` ->
    `modeling_gemma.eager_attention_forward(...)`) and the eager attention path
    of `GemmaAttention.forward` (`attention_interface = eager_attention_forward`)
    resolve `eager_attention_forward` from the module namespace at call time, so
    rebinding the module attribute swaps BOTH the joint and suffix-KV-cache
    paths in one shot. Returns the original callable for `disable`/restore.

        from attention import enable_fused_attention, disable_fused_attention
        orig = enable_fused_attention()
        ... run actor get_log_prob_value / denoise ...
        disable_fused_attention(orig)   # optional
    """
    from transformers.models.gemma import modeling_gemma
    orig = modeling_gemma.eager_attention_forward
    modeling_gemma.eager_attention_forward = fused_attention_forward
    return orig


def disable_fused_attention(orig):
    """Restore the stock eager attention captured by `enable_fused_attention`."""
    from transformers.models.gemma import modeling_gemma
    modeling_gemma.eager_attention_forward = orig
