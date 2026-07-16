"""Fast training step (forward + backward) for the fused Gemma prefix layer.

The whole-layer Triton kernel is forward-only; the default autograd wrapper
(ref_tests._FusedFn) makes `.backward()` correct by *recomputing the entire
forward in PyTorch* and autograd-ing through it. Measured, that recompute costs
a full eager fwd+bwd (~12.9 ms) on top of the fast Triton forward (~3.5 ms) ->
16.4 ms, a 0.79x regression vs eager.

This module removes the redundant forward recompute. `prefix_train_forward`
runs the fast path (Triton GEMMs for the 7 projections) but *saves the
activations*; `prefix_train_backward` then computes gradients directly:

  * the 7 projection gradients are plain transposed GEMMs (dX = dY·W,
    dW = dYᵀ·X) — cuBLAS runs these at roofline, so they stay in torch;
  * the cheap nonlinear pieces (RMSNorm, gated-MLP, softmax attention) are
    differentiated by a *local* autograd recompute of just that op (each is
    O(S) or a single S² matmul, the same work eager does), never re-running the
    expensive projection GEMMs;
  * RoPE's gradient is the transpose rotation (dx = dO·cos − rotate_half(dO)·sin),
    done in closed form.

So a training step is `triton_fwd + (grad-only backward)` instead of
`triton_fwd + (eager_fwd + eager_bwd)`. Backward is GEMM-FLOP-bound and
cuBLAS-optimal, so the ceiling is modest (~1.05-1.15x vs eager) — but it turns
the regression into a win. Prefix path only (standard RMSNorm, full attention);
the adaRMS / masked paths keep the exact recompute backward in ref_tests.
"""

import os
import sys

import torch
import torch.nn.functional as F
import triton

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(_HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import triton.language as tl

from kernel import (_matmul, _rmsnorm, rope_kernel, attn_kernel, _tanh,  # noqa: E402
                    _gelu_tanh, HEAD_DIM, ROPE_THETA)
from problem import _rms_norm, _rotate_half                 # noqa: E402

# FlashAttention (FA2 Hopper kernels) — its D=256 backward beats a Triton flash
# backward (which spills registers at head_dim 256) and torch (~0.70 vs 0.99 ms).
from flash_attn.flash_attn_interface import (  # noqa: E402
    _flash_attn_forward as _fa_fwd, _flash_attn_backward as _fa_bwd)


# --------------------------------------------------------------------------- #
# Fused MLP down-projection backward: dprod = dOut @ Wd, then in the epilogue
# produce dgl = gelu_tanh'(gl) * (dprod * u)  and  du = dprod * g directly, so
# the [M,I] dprod is never materialised and the gelu-grad / gated-mul passes are
# folded into the GEMM. (Forward: out = prod @ Wdᵀ, prod = gelu(gl)*u.)
# A = dOut[M,K=H], W = Wdᵀ[N=I,K=H] (pass wd.t()); outputs dgl,du [M,I].
# --------------------------------------------------------------------------- #
@triton.jit
def _down_bwd_kernel(a_ptr, w_ptr, gl_ptr, u_ptr, g_ptr, dgl_ptr, du_ptr,
                     M, N, K,
                     stride_am, stride_ak, stride_wn, stride_wk,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
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

    full = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    off = offs_m[:, None] * N + offs_n[None, :]
    gl = tl.load(gl_ptr + off, mask=full, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + off, mask=full, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + off, mask=full, other=0.0).to(tl.float32)
    # gelu-tanh'(gl)
    inner = 0.7978845608028654 * (gl + 0.044715 * gl * gl * gl)
    t = _tanh(inner)
    dgelu = 0.5 * (1.0 + t) + 0.5 * gl * (1.0 - t * t) * \
        0.7978845608028654 * (1.0 + 3.0 * 0.044715 * gl * gl)
    dgl = dgelu * (acc * u)
    du = acc * g
    tl.store(dgl_ptr + off, dgl.to(dgl_ptr.dtype.element_ty), mask=full)
    tl.store(du_ptr + off, du.to(du_ptr.dtype.element_ty), mask=full)


# --------------------------------------------------------------------------- #
# Forward gated-MLP fusion: two-output GEMM epilogues so the gelu / gated-mul
# passes are folded into the projection GEMMs (and gl/u are still available for
# backward). C = A @ W^T (nn.Linear weight W=[N,K]).
#   gate: gl = hn@Wg^T, g = gelu_tanh(gl)         -> outputs (gl, g)
#   up:   u  = hn@Wu^T, prod = u * g              -> outputs (u, prod)
# --------------------------------------------------------------------------- #
@triton.jit
def _twoout_mm_kernel(a_ptr, w_ptr, other_ptr, o1_ptr, o2_ptr, M, N, K,
                      stride_am, stride_ak, stride_wn, stride_wk,
                      MODE: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
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
    full = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    off = offs_m[:, None] * N + offs_n[None, :]
    if MODE == 0:   # gate: o1=gl (linear), o2=g=gelu(gl)
        o2 = _gelu_tanh(acc)
    else:           # up: o1=u (linear), o2=prod=u*g
        gg = tl.load(other_ptr + off, mask=full, other=0.0).to(tl.float32)
        o2 = acc * gg
    tl.store(o1_ptr + off, acc.to(o1_ptr.dtype.element_ty), mask=full)
    tl.store(o2_ptr + off, o2.to(o2_ptr.dtype.element_ty), mask=full)


def _twoout_mm(a, w, mode, other=None):
    M, K = a.shape
    N = w.shape[0]
    o1 = torch.empty((M, N), device=a.device, dtype=a.dtype)
    o2 = torch.empty((M, N), device=a.device, dtype=a.dtype)
    other_ptr = other if other is not None else o1
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 64, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _twoout_mm_kernel[grid](a, w, other_ptr, o1, o2, M, N, K,
                            a.stride(0), a.stride(1), w.stride(0), w.stride(1),
                            MODE=mode, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                            BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)
    return o1, o2


def _down_bwd_fused(dout, wd, gl, u, g):
    """dgl = gelu'(gl)*(dOut@Wd * u), du = (dOut@Wd) * g.  dout:[M,H], wd:[H,I]."""
    M, K = dout.shape
    N = wd.shape[0] if wd.shape[1] == K else wd.shape[1]   # I
    I = gl.shape[1]
    wt = wd.t()                                            # [I, H] view
    dgl = torch.empty((M, I), device=dout.device, dtype=dout.dtype)
    du = torch.empty((M, I), device=dout.device, dtype=dout.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 64, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(I, BLOCK_N),)
    _down_bwd_kernel[grid](
        dout, wt, gl, u, g, dgl, du, M, I, K,
        dout.stride(0), dout.stride(1), wt.stride(0), wt.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M)
    return dgl, du


@triton.jit
def _rmsnorm_bwd_kernel(x_ptr, dy_ptr, w_ptr, dx_ptr, dw_ptr, n_rows, D, eps,
                        BLOCK: tl.constexpr):
    """One program per row: dx in one fused fp32 pass; dw via atomic add."""
    row = tl.program_id(0)
    if row >= n_rows:
        return
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = x * rstd
    gnorm = dy * (1.0 + w)
    mgxn = tl.sum(gnorm * xn) / D
    dx = rstd * (gnorm - xn * mgxn)
    tl.store(dx_ptr + row * D + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)
    tl.atomic_add(dw_ptr + cols, dy * xn, mask=mask)


def _rmsnorm_bwd(dy, x_in, weight, eps):
    """Fused Triton Gemma RMSNorm backward. y = (x*rstd)*(1+weight).

    ~5x faster than the closed-form torch version (single fp32 pass in
    registers; dw accumulated with a per-column atomic add). x_in,dy:[M,D].
    """
    M, D = x_in.shape
    dx = torch.empty_like(dy)
    dw = torch.zeros(D, device=dy.device, dtype=torch.float32)
    _rmsnorm_bwd_kernel[(M,)](x_in, dy, weight, dx, dw, M, D, float(eps),
                              BLOCK=triton.next_power_of_2(D))
    return dx, dw.to(weight.dtype)


def _rope_tables(S, D, device, dtype):
    half = D // 2
    inv = torch.exp(-(torch.arange(0, half, device=device).float() * (2.0 / D)) *
                    torch.log(torch.tensor(ROPE_THETA)))
    pos = torch.arange(S, device=device).float()
    ang = torch.outer(pos, inv)                     # [S, half]
    emb = torch.cat((ang, ang), dim=-1)             # [S, D]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotary_pos(position_ids, D, dtype):
    """cos/sin for explicit positions (padded prefix uses cumsum(pad)-1). [B,S,D]."""
    half = D // 2
    dev = position_ids.device
    inv = torch.exp(-(torch.arange(0, half, device=dev).float() * (2.0 / D)) *
                    torch.log(torch.tensor(ROPE_THETA)))
    ang = position_ids.float()[:, :, None] * inv[None, None, :]   # [B,S,half]
    emb = torch.cat((ang, ang), dim=-1)                          # [B,S,D]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def prefix_train_forward(x, w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd, eps, meta,
                         attention_mask=None, position_ids=None, use_cache=False):
    """Fast forward that also returns the activations needed for backward.

    attention_mask: optional additive [B,1,S,S] bias (openpi's general mask,
    modeling_gemma.eager_attention_forward L244-246). When given, attention runs
    the Triton masked kernel (fwd) + a torch-manual masked backward — FlashAttn
    can't take an arbitrary additive mask, and varlen (padding-only) is slower
    once you count pack/unpack. When None, the fast FlashAttention path is used.
    position_ids: optional [B,S] RoPE positions (padded prefix = cumsum(pad)-1);
    defaults to arange.
    use_cache: also surface this layer's rope'd K and pre-attention V as
    `ctx["kv_cache"] = (k, v)`, each [B, n_kv, S, hd] — the HF `Cache.update`
    layout — so a prefix-cache build can collect per-layer K/V. The tensors are
    the ones attention already consumed, so this only costs the transpose/copy
    into HF layout (n_kv is 1 here, so a few hundred KB).
    """
    n_heads, n_kv, hd = meta
    B, S, H = x.shape
    M = B * S
    groups = n_heads // n_kv
    scale = hd ** -0.5
    xr = x.reshape(M, H)

    h = _rmsnorm(xr, w_ln, eps)                      # triton [M,H]
    q = _matmul(h, wq)
    k = _matmul(h, wk)
    v = _matmul(h, wv)

    # RoPE (Triton), honouring explicit positions if given.
    half = hd // 2
    has_pos = position_ids is not None
    pos_flat = position_ids.reshape(M).to(torch.int32) if has_pos else q
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    rope_kernel[(M * n_heads,)](q, q_rope, pos_flat, M * n_heads, S, n_heads, hd, ROPE_THETA,
                                HAS_POS=has_pos, HALF=half)
    rope_kernel[(M * n_kv,)](k, k_rope, pos_flat, M * n_kv, S, n_kv, hd, ROPE_THETA,
                             HAS_POS=has_pos, HALF=half)
    if has_pos:
        cos, sin = _rotary_pos(position_ids, hd, x.dtype)       # [B,S,hd]
    else:
        cos, sin = _rope_tables(S, hd, x.device, x.dtype)       # [S,hd]

    masked = attention_mask is not None
    if masked:
        # FlashAttn can't take an arbitrary additive mask; use a materialised
        # torch attention (cuBLAS-batched, fp32 softmax) and SAVE p so the
        # backward reuses it (no s/softmax recompute). [B,Hq,S,hd] layout.
        qr = q_rope.view(B, S, n_heads, hd).transpose(1, 2)   # [B,Hq,S,hd]
        kr = k_rope.view(B, S, n_kv, hd).transpose(1, 2)
        vv = v.view(B, S, n_kv, hd).transpose(1, 2)
        kx = kr.repeat_interleave(groups, 1) if groups > 1 else kr
        vx = vv.repeat_interleave(groups, 1) if groups > 1 else vv
        s = torch.matmul(qr, kx.transpose(-1, -2)).float() * scale + attention_mask.float()
        p = torch.softmax(s, dim=-1)                          # fp32 [B,Hq,S,S]
        ao = torch.matmul(p.to(x.dtype), vx).transpose(1, 2).reshape(M, n_heads * hd)
        out_fa = lse = rng = None
    else:
        qr = q_rope.view(B, S, n_heads, hd)         # [B,S,Hq,hd] (FA layout)
        kr = k_rope.view(B, S, n_kv, hd)
        vv = v.view(B, S, n_kv, hd)
        out_fa, lse, _, rng = _fa_fwd(
            qr, kr, vv, 0.0, scale, causal=False, window_size_left=-1,
            window_size_right=-1, softcap=0.0, alibi_slopes=None, return_softmax=False)
        ao = out_fa.reshape(M, n_heads * hd)
        p = None

    # HF past_key_values wants [B, n_kv, S, hd]: the masked path already holds
    # kr/vv that way, the FlashAttn path keeps them [B, S, n_kv, hd]. Export the
    # *rope'd* k (kr) and the pre-GQA-expansion v (vv) — not kx/vx, whose heads
    # are repeat_interleave'd up to n_heads and which a cache must not store.
    if use_cache:
        kc, vc = (kr, vv) if masked else (kr.transpose(1, 2), vv.transpose(1, 2))
        kv_cache = (kc.contiguous(), vc.contiguous())
    else:
        kv_cache = None

    o = _matmul(ao, wo)                              # triton [M,H]
    h1 = xr + o
    hn = _rmsnorm(h1, w_pln, eps)                    # triton [M,H]
    gl, g = _twoout_mm(hn, wg, mode=0)               # gl, g=gelu(gl)  (fused)
    u, prod = _twoout_mm(hn, wu, mode=1, other=g)    # u, prod=u*g     (fused)
    out = _matmul(prod, wd, res=h1)                  # triton [M,H]  (fused residual)

    ctx = dict(x=x, h=h, ao=ao, h1=h1, hn=hn, gl=gl, g=g, u=u, prod=prod,
               cos=cos, sin=sin, has_pos=has_pos, masked=masked, mask=attention_mask,
               qr=qr, kr=kr, vv=vv, p=p, out_fa=out_fa, lse=lse, rng=rng,
               kv_cache=kv_cache,
               w_ln=w_ln, wq=wq, wk=wk, wv=wv, wo=wo, w_pln=w_pln,
               wg=wg, wu=wu, wd=wd, eps=eps, meta=meta, shape=(B, S, H))
    return out.view(B, S, H), ctx


def prefix_train_backward(ctx, dout, dk_cache=None, dv_cache=None):
    """Grad-only backward. dk_cache/dv_cache are the incoming grads of the
    `use_cache` K/V outputs ([B, n_kv, S, hd]); they fold into this layer's own
    dk/dv, so a suffix that attends to the cached prefix K/V still trains the
    prefix correctly. Both None (the plain training step) is the fast path.
    """
    n_heads, n_kv, hd = ctx["meta"]
    B, S, H = ctx["shape"]
    M = B * S
    groups = n_heads // n_kv
    scale = hd ** -0.5
    dtype = dout.dtype
    (x, h, ao, h1, hn, gl, g, u, prod, cos, sin) = (
        ctx["x"], ctx["h"], ctx["ao"], ctx["h1"], ctx["hn"], ctx["gl"], ctx["g"],
        ctx["u"], ctx["prod"], ctx["cos"], ctx["sin"])
    masked, has_pos = ctx["masked"], ctx["has_pos"]
    wq, wk, wv, wo = ctx["wq"], ctx["wk"], ctx["wv"], ctx["wo"]
    wg, wu, wd, w_ln, w_pln, eps = ctx["wg"], ctx["wu"], ctx["wd"], ctx["w_ln"], ctx["w_pln"], ctx["eps"]

    doutr = dout.reshape(M, H)

    # ---- down proj + gated-MLP epilogue (fused GEMM: dgl, du direct) ----
    # dprod = dOut@Wd never materialised; gelu-grad + gated-mul folded in.
    dgl, du = _down_bwd_fused(doutr, wd, gl, u, g)
    dWd = doutr.t() @ prod                    # [H,I]  (cuBLAS)
    dh1 = doutr.clone()                       # residual2 into h1
    dWg = dgl.t() @ hn
    dWu = du.t() @ hn
    dhn = dgl @ wg + du @ wu                   # [M,H]

    # ---- post-attention RMSNorm (closed form) ----
    dh1n, dw_pln = _rmsnorm_bwd(dhn, h1, w_pln, eps)
    dh1 = dh1 + dh1n

    # ---- o proj + residual1 ----
    dao = dh1 @ wo                            # [M, q_dim]
    dWo = dh1.t() @ ao
    dx = dh1.clone()                          # residual1 into x

    if masked:
        # ---- masked attention backward (torch, reuses saved p) ----
        # p was materialised in the forward, so the backward applies the softmax
        # jacobian directly (no s/softmax recompute). [B,Hq,S,hd] layout.
        qr, kr, vh, p = ctx["qr"], ctx["kr"], ctx["vv"], ctx["p"]
        kx = kr.repeat_interleave(groups, 1) if groups > 1 else kr
        vx = vh.repeat_interleave(groups, 1) if groups > 1 else vh
        dao_r = dao.view(B, S, n_heads, hd).transpose(1, 2)     # [B,Hq,S,hd]
        dvx = torch.matmul(p.to(dtype).transpose(-1, -2), dao_r)
        dp = torch.matmul(dao_r, vx.transpose(-1, -2)).float()
        ds = (p * (dp - (dp * p).sum(-1, keepdim=True))).to(dtype)
        dqr = torch.matmul(ds, kx) * scale                     # [B,Hq,S,hd]
        dkx = torch.matmul(ds.transpose(-1, -2), qr) * scale
        if groups > 1:
            dkr = dkx.view(B, n_kv, groups, S, hd).sum(2)
            dvh = dvx.view(B, n_kv, groups, S, hd).sum(2)
        else:
            dkr, dvh = dkx, dvx
        # kr/vv were also exported as the K/V cache; fold their incoming grads in
        # (already [B,n_kv,S,hd] on this path).
        if dk_cache is not None:
            dkr = dkr + dk_cache
        if dv_cache is not None:
            dvh = dvh + dv_cache
        # RoPE backward, [B,H,S,hd] layout
        cse = cos.view(B, 1, S, hd) if has_pos else cos.view(1, 1, S, hd)
        sne = sin.view(B, 1, S, hd) if has_pos else sin.view(1, 1, S, hd)
        dq = (dqr * cse - _rotate_half(dqr) * sne).transpose(1, 2).reshape(M, n_heads * hd)
        dk = (dkr * cse - _rotate_half(dkr) * sne).transpose(1, 2).reshape(M, n_kv * hd)
        dv = dvh.transpose(1, 2).reshape(M, n_kv * hd)
    else:
        # ---- FlashAttention backward ([B,S,H,hd] layout, GQA-summed) ----
        qr, kr, vv = ctx["qr"], ctx["kr"], ctx["vv"]
        out_fa, lse, rng = ctx["out_fa"], ctx["lse"], ctx["rng"]
        dao_bshd = dao.reshape(B, S, n_heads, hd)
        dqr = torch.empty_like(qr)
        dkr = torch.empty_like(kr)
        dvv = torch.empty_like(vv)
        _fa_bwd(dao_bshd, qr, kr, vv, out_fa, lse, dqr, dkr, dvv,
                0.0, scale, False, -1, -1, 0.0, None, False, rng_state=rng)
        # Cache grads arrive [B,n_kv,S,hd]; this path works in [B,S,n_kv,hd].
        if dk_cache is not None:
            dkr = dkr + dk_cache.transpose(1, 2)
        if dv_cache is not None:
            dvv = dvv + dv_cache.transpose(1, 2)
        # RoPE backward, [B,S,H,hd] layout
        cse = cos.view(B, S, 1, hd) if has_pos else cos.view(1, S, 1, hd)
        sne = sin.view(B, S, 1, hd) if has_pos else sin.view(1, S, 1, hd)
        dq = (dqr * cse - _rotate_half(dqr) * sne).reshape(M, n_heads * hd)
        dk = (dkr * cse - _rotate_half(dkr) * sne).reshape(M, n_kv * hd)
        dv = dvv.reshape(M, n_kv * hd)

    # ---- q/k/v proj ----
    dWq = dq.t() @ h
    dWk = dk.t() @ h
    dWv = dv.t() @ h
    dh = dq @ wq + dk @ wk + dv @ wv           # [M,H]

    # ---- input RMSNorm (closed form) ----
    dxn, dw_ln = _rmsnorm_bwd(dh, x.reshape(M, H), w_ln, eps)
    dx = dx + dxn

    return (dx.view(B, S, H), dw_ln, dWq, dWk, dWv, dWo, dw_pln, dWg, dWu, dWd)


class PrefixTrainFn(torch.autograd.Function):
    """Fast fwd + grad-only backward for the standard (prefix) Gemma layer.

    With `use_cache=True` the call returns `(hidden_states, k, v)` instead of a
    bare `hidden_states`; k/v are this layer's rope'd K and its V, each
    [B, n_kv, S, hd] — the layout `Cache.update(k, v, layer_idx)` expects — so a
    prefix-cache build can collect per-layer K/V and a suffix can attend to them
    without recomputing the prefix. k/v are differentiable: their grads fold back
    into this layer's dk/dv (see prefix_train_backward). Default `use_cache=False`
    keeps the single-output signature the training step already uses.
    """

    @staticmethod
    def forward(ctx, x, w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd, eps, meta,
                attention_mask=None, position_ids=None, use_cache=False):
        out, saved = prefix_train_forward(x, w_ln, wq, wk, wv, wo, w_pln,
                                          wg, wu, wd, eps, meta,
                                          attention_mask=attention_mask,
                                          position_ids=position_ids,
                                          use_cache=use_cache)
        ctx.saved = saved
        if use_cache:
            k, v = saved["kv_cache"]
            return out, k, v
        return out

    @staticmethod
    def backward(ctx, grad_out, grad_k=None, grad_v=None):
        g = prefix_train_backward(ctx.saved, grad_out.contiguous(),
                                  dk_cache=grad_k, dv_cache=grad_v)
        # grads for (x, w_ln, wq..wd, eps, meta, attention_mask, position_ids,
        #            use_cache)
        return (*g, None, None, None, None, None)
