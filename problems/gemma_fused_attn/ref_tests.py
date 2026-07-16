# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness harness for a fused Gemma decoder layer (Pi0.5).

Validates a KernelAgent-written fused replacement of the standard
`GemmaDecoderLayer` used by openpi against the vendored transformers reference,
for BOTH sides of the Pi0.5 model:

  * prefix (PaliGemma VLM, gemma_2b): standard RMSNorm, use_adarms=False.
  * suffix (action expert, gemma_300m): adaptive RMSNorm (use_adarms=True) with
    a time-embedding `adarms_cond`, gated residuals — the layer the PPO actor's
    denoise recompute runs.

Both are checked under a real additive attention mask (not just the all-zero
full-prefix case) and with explicit RoPE `position_ids`, comparing:
  * forward output   (max/mean abs diff)      -- the fast Triton kernel's value
  * input gradient   (dL/dx, via backward)    -- via the autograd.Function
  * every parameter gradient

The fused module is a normal `nn.Module` whose forward is autograd-capable
(`torch.autograd.Function`): forward runs the fast fused Triton kernel; backward
recomputes gradients through the equivalent (validated) PyTorch math over the
SAME mask / positions / cond, so `loss.backward()` works (e.g. under FSDP) and
the grad checks pass. bf16 fused kernels reorder fp ops, so bit-exactness is not
expected (~2e-2 relative for bf16).

Run (needs openpi's patched transformers, >=4.53 with the adarms API):
  /opt/venv/openpi/bin/python problems/gemma_fused_attn/ref_tests.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto import CONFIG_MAPPING

# ---- dims from openpi/models/gemma.py get_config(...) ----
DIMS = {
    # prefix VLM
    "gemma_2b": dict(hidden=2048, mlp=16384, heads=8, kv=1, head_dim=256, depth=18),
    # suffix action expert
    "gemma_300m": dict(hidden=1024, mlp=4096, heads=8, kv=1, head_dim=256, depth=18),
}
EPS = 1e-6
ROPE_THETA = 10000.0
SEQ = 968       # prefix tokens (256*3 imgs + 200 lang); reused for the tests
BATCH = 2
MASK_NEG = -2.3819763e38  # openpi _prepare_attention_masks_4d masked value


# ---------------------------------------------------------------------------
# Reference (vendored transformers GemmaDecoderLayer)
# ---------------------------------------------------------------------------
def build_reference_layer(device, dtype, variant="gemma_2b", use_adarms=False):
    """One standard GemmaDecoderLayer for `variant`, adarms optional.

    use_adarms=False -> prefix (standard RMSNorm).
    use_adarms=True  -> suffix / action-expert (adaptive RMSNorm, cond_dim=width).
    """
    from transformers.models.gemma import modeling_gemma

    d = DIMS[variant]
    cfg = CONFIG_MAPPING["gemma"](
        head_dim=d["head_dim"],
        hidden_size=d["hidden"],
        intermediate_size=d["mlp"],
        num_attention_heads=d["heads"],
        num_hidden_layers=d["depth"],
        num_key_value_heads=d["kv"],
        vocab_size=257152,
        hidden_activation="gelu_pytorch_tanh",
    )
    cfg.use_adarms = use_adarms
    cfg.adarms_cond_dim = d["hidden"] if use_adarms else None
    cfg._attn_implementation = "eager"
    layer = modeling_gemma.GemmaDecoderLayer(cfg, layer_idx=0).to(device).to(dtype)
    # randomize weights (init is near-zero / adarms dense is zero-inited); we
    # only test numerical equivalence, so make the modulation non-trivial.
    for p in layer.parameters():
        torch.nn.init.normal_(p, std=0.02)
    layer.eval()
    return layer, cfg


def _rotary(position_ids, head_dim, dtype):
    """cos/sin matching GemmaRotaryEmbedding (default rope), [B,S,head_dim]."""
    half = head_dim // 2
    dev = position_ids.device
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, half, device=dev).float() * 2.0 / head_dim))
    freqs = position_ids.float()[:, :, None] * inv[None, None, :]   # [B,S,half]
    emb = torch.cat((freqs, freqs), dim=-1)                         # [B,S,head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def make_inputs(device, dtype, variant="gemma_2b", mask_kind="zero",
                use_adarms=False, seq=SEQ, batch=BATCH, pos_offset=0):
    """Build (x, position_ids, position_embeddings, attn_mask_4d, adarms_cond).

    mask_kind: 'zero'  -> all-zero full-prefix additive mask.
               'block' -> prefix-LM / block-diagonal mask built the openpi way
                          (make_att_2d_masks -> _prepare_attention_masks_4d).
    Returns a 5-tuple; adarms_cond is None unless use_adarms.
    """
    d = DIMS[variant]
    H, head_dim = d["hidden"], d["head_dim"]
    x = torch.randn(batch, seq, H, device=device, dtype=dtype)

    # position_ids = cumsum(pad)-1 (+ prefix offset for the suffix); pad = all-1.
    position_ids = pos_offset + torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
    pos_emb = _rotary(position_ids, head_dim, dtype)

    if mask_kind == "zero":
        attn = torch.zeros(batch, 1, seq, seq, device=device, dtype=dtype)
    elif mask_kind == "block":
        # prefix-LM: first `P` tokens attend bidirectionally, the rest causal.
        # mask_ar = 0 for prefix, 1 afterwards (openpi convention).
        P = seq // 2
        att_masks = torch.zeros(batch, seq, dtype=torch.long, device=device)
        att_masks[:, P:] = 1
        pad_masks = torch.ones(batch, seq, dtype=torch.bool, device=device)
        cumsum = torch.cumsum(att_masks, dim=1)
        att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
        att_2d = att_2d & pad_2d                       # [B,S,S] bool
        attn = torch.where(att_2d[:, None, :, :], 0.0, MASK_NEG).to(dtype)
    else:
        raise ValueError(mask_kind)

    cond = None
    if use_adarms:
        cond = torch.randn(batch, d["hidden"], device=device, dtype=dtype)
    return x, position_ids, pos_emb, attn, cond


def ref_forward(layer, x, position_ids, pos_emb, attn, adarms_cond=None):
    out = layer(
        x,
        attention_mask=attn,
        position_ids=position_ids,
        position_embeddings=pos_emb,
        adarms_cond=adarms_cond,
    )
    return out[0]


# ---------------------------------------------------------------------------
# Differentiable pure-torch equivalent (backward recompute AND a reference that
# runs without the transformers layer). Mirrors GemmaDecoderLayer.forward for
# both the standard and adaRMS paths, honouring mask + position_ids.
# ---------------------------------------------------------------------------
def _torch_norm(x, weight, dense_w, dense_b, cond, eps):
    dtype = x.dtype
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normed = x * torch.rsqrt(var + eps)                       # fp32
    if cond is None:
        return (normed * (1.0 + weight.float())).to(dtype), None
    mod = F.linear(cond, dense_w, dense_b).unsqueeze(1)       # [B,1,3H]
    scale, shift, gate = mod.chunk(3, dim=-1)
    normed = normed * (1.0 + scale.float()) + shift.float()
    return normed.to(dtype), gate.to(dtype)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _gated_residual(r, y, gate):
    return r + y if gate is None else r + y * gate


def _torch_layer(x, w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd,
                 di_w, di_b, dp_w, dp_b, cond, mask, position_ids, eps, meta):
    """Same math as GemmaDecoderLayer.forward (both variants), differentiable."""
    n_heads, n_kv, head_dim = meta
    B, S, _ = x.shape
    dt = x.dtype
    groups = n_heads // n_kv
    if position_ids is None:                     # kernel defaults RoPE to arange
        position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)

    r = x
    h, gate1 = _torch_norm(x, w_ln, di_w, di_b, cond, eps)
    q = F.linear(h, wq).view(B, S, n_heads, head_dim).transpose(1, 2)
    k = F.linear(h, wk).view(B, S, n_kv, head_dim).transpose(1, 2)
    v = F.linear(h, wv).view(B, S, n_kv, head_dim).transpose(1, 2)

    cos, sin = _rotary(position_ids, head_dim, dt)             # [B,S,head_dim]
    cos, sin = cos[:, None], sin[:, None]                      # -> [B,1,S,head_dim]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)

    attn = torch.matmul(q, k.transpose(2, 3)) * (head_dim ** -0.5)
    if mask is not None:
        attn = attn + mask                                    # additive bias
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(dt)
    o = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, -1)
    o = F.linear(o, wo)
    h = _gated_residual(r, o, gate1)

    r = h
    h, gate2 = _torch_norm(h, w_pln, dp_w, dp_b, cond, eps)
    m = F.linear(F.gelu(F.linear(h, wg), approximate="tanh") * F.linear(h, wu), wd)
    return _gated_residual(r, m, gate2)


# ---------------------------------------------------------------------------
# Fused module (Triton forward + autograd backward recompute)
# ---------------------------------------------------------------------------
def _load_kernel():
    _here = os.path.dirname(os.path.abspath(__file__))
    for _d in (_here, os.path.join(_here, "h20")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
    from kernel import kernel_function
    return kernel_function


class _FusedFn(torch.autograd.Function):
    """Forward = fast fused Triton kernel; backward depends on the path.

    Prefix training path (no adaRMS cond, no attention mask, default arange
    positions): the fast grad-only backward in h20/layer_train.py — a Triton
    forward that saves activations + direct gradient computation (no forward
    recompute), which turns the whole-layer training step from ~0.79x into a
    slight win vs eager. Every other path (adaRMS / arbitrary mask / explicit
    positions) uses the exact PyTorch recompute backward.

    `use_cache=True` returns `(out, k, v)` instead of a bare `out`, surfacing
    this layer's rope'd K / pre-GQA-expansion V as [B, n_kv, S, hd] so a
    prefix-cache build can collect them. Prefix path only: the cache is built by
    the gemma_2b VLM prefix, and the adaRMS action expert consumes it rather than
    producing it — kernel_function never surfaces its K/V, hence the explicit
    error instead of a silently empty cache.
    """

    @staticmethod
    def forward(ctx, x, w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd,
                di_w, di_b, dp_w, dp_b, cond, mask, position_ids, eps, meta,
                use_cache=False):
        # Fast path for the standard (non-adaRMS) layer — masked or not. The
        # masked prefix uses the Triton masked-attn fwd + torch-manual masked
        # bwd (FlashAttn can't take an arbitrary additive mask); the unmasked
        # prefix uses the FlashAttention fast path.
        ctx.fast = (cond is None and di_w is None)
        if ctx.fast:
            _load_kernel()  # ensures h20/ is on sys.path for layer_train
            from layer_train import prefix_train_forward
            out, saved = prefix_train_forward(x, w_ln, wq, wk, wv, wo, w_pln,
                                              wg, wu, wd, eps, meta,
                                              attention_mask=mask,
                                              position_ids=position_ids,
                                              use_cache=use_cache)
            ctx.train_ctx = saved
            if use_cache:
                k, v = saved["kv_cache"]
                return out, k, v
            return out

        if use_cache:
            raise NotImplementedError(
                "use_cache is only supported on the prefix (non-adaRMS) path; "
                "the adaRMS action-expert layer consumes a prefix cache rather "
                "than producing one.")

        args = [x, w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd, di_w, di_b, dp_w, dp_b, cond]
        ctx.none_mask = [a is None for a in args]
        ctx.save_for_backward(*[a for a in args if a is not None])
        ctx.mask = mask
        ctx.position_ids = position_ids
        ctx.eps = eps
        ctx.meta = meta
        kernel_function = _load_kernel()
        return kernel_function(
            x, w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd, eps,
            attention_mask=mask, position_ids=position_ids, adarms_cond=cond,
            input_dense=(di_w, di_b) if di_w is not None else None,
            post_dense=(dp_w, dp_b) if dp_w is not None else None,
        )

    @staticmethod
    def backward(ctx, grad_out, grad_k=None, grad_v=None):
        if ctx.fast:
            from layer_train import prefix_train_backward
            # grad_k/grad_v are the incoming grads of the use_cache K/V outputs
            # (None unless use_cache); they fold into this layer's own dk/dv, so
            # a suffix attending the cached prefix still trains the prefix.
            g = prefix_train_backward(ctx.train_ctx, grad_out.contiguous(),
                                      dk_cache=grad_k, dv_cache=grad_v)
            # g = (dx, dw_ln, dWq..dWd); pad di_w/di_b/dp_w/dp_b/cond/mask/pos/
            # eps/meta/use_cache
            dx, dw_ln, dWq, dWk, dWv, dWo, dw_pln, dWg, dWu, dWd = g
            return (dx, dw_ln, dWq, dWk, dWv, dWo, dw_pln, dWg, dWu, dWd,
                    None, None, None, None, None, None, None, None, None, None)

        saved = list(ctx.saved_tensors)
        args = [None if is_none else saved.pop(0) for is_none in ctx.none_mask]
        diff = [
            a.detach().requires_grad_(True) if (isinstance(a, torch.Tensor) and a.is_floating_point()) else a
            for a in args
        ]
        with torch.enable_grad():
            y = _torch_layer(*diff, ctx.mask, ctx.position_ids, ctx.eps, ctx.meta)
        needs = [d for d in diff if isinstance(d, torch.Tensor) and d.requires_grad]
        grads = torch.autograd.grad(y, needs, grad_out, allow_unused=True)
        gi = iter(grads)
        out = [next(gi) if (isinstance(d, torch.Tensor) and d.requires_grad) else None for d in diff]
        # mask, position_ids, eps, meta, use_cache
        return (*out, None, None, None, None, None)


def build_fused_layer(ref, cfg, device, dtype, layer_idx=0):
    """Fused Gemma decoder layer backed by the Triton kernel in h20/kernel.py.

    Detects the adaRMS variant from `cfg` and mirrors GemmaDecoderLayer's
    parameter names (self_attn.q_proj.weight, mlp.gate_proj.weight,
    input_layernorm.{weight | dense.weight/dense.bias}, ...) so the per-parameter
    grad comparison in main() engages for every parameter.

    `layer_idx` is the slot this layer occupies in `language_model.layers`; it is
    the index the K/V are written to when a `past_key_value` Cache is passed to
    forward().
    """
    _load_kernel()  # ensure importable / compiled
    H = cfg.hidden_size
    I = cfg.intermediate_size
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    q_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    adarms = bool(getattr(cfg, "use_adarms", False))
    cond_dim = getattr(cfg, "adarms_cond_dim", None)
    meta = (n_heads, n_kv, head_dim)

    class _Norm(nn.Module):
        def __init__(self):
            super().__init__()
            if adarms:
                self.dense = nn.Linear(cond_dim, H * 3, bias=True)
            else:
                self.weight = nn.Parameter(torch.zeros(H))

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(H, q_dim, bias=False)
            self.k_proj = nn.Linear(H, kv_dim, bias=False)
            self.v_proj = nn.Linear(H, kv_dim, bias=False)
            self.o_proj = nn.Linear(q_dim, H, bias=False)

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(H, I, bias=False)
            self.up_proj = nn.Linear(H, I, bias=False)
            self.down_proj = nn.Linear(I, H, bias=False)

    class FusedGemmaLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = _Norm()
            self.post_attention_layernorm = _Norm()
            self.self_attn = _Attn()
            self.mlp = _MLP()
            self.eps = float(cfg.rms_norm_eps)
            self.layer_idx = layer_idx

        def forward(self, x, position_ids=None, position_embeddings=None,
                    attention_mask=None, adarms_cond=None,
                    past_key_value=None, use_cache=False, cache_kwargs=None,
                    **kwargs):
            """Returns `hidden_states`, except when `use_cache=True` is given
            *without* a `past_key_value`, which returns `(hidden_states, k, v)`.

            Passing a `past_key_value` Cache follows the HF convention: this
            layer's K/V are written into it at `self.layer_idx` in place and a
            bare `hidden_states` comes back — which is what lets a prefix-cache
            build collect per-layer K/V by calling the stack and then reading the
            Cache. Without either argument the signature is unchanged.
            """
            a = self.self_attn
            m = self.mlp
            if adarms:
                w_ln = w_pln = None
                di_w, di_b = self.input_layernorm.dense.weight, self.input_layernorm.dense.bias
                dp_w, dp_b = self.post_attention_layernorm.dense.weight, self.post_attention_layernorm.dense.bias
            else:
                w_ln = self.input_layernorm.weight
                w_pln = self.post_attention_layernorm.weight
                di_w = di_b = dp_w = dp_b = None
            want_kv = use_cache or past_key_value is not None
            res = _FusedFn.apply(
                x, w_ln, a.q_proj.weight, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight,
                w_pln, m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight,
                di_w, di_b, dp_w, dp_b, adarms_cond, attention_mask, position_ids,
                self.eps, meta, want_kv,
            )
            if not want_kv:
                return res
            out, k, v = res
            if past_key_value is None:
                return out, k, v
            past_key_value.update(k, v, self.layer_idx, cache_kwargs)
            return out

    fused = FusedGemmaLayer().to(device).to(dtype)
    with torch.no_grad():
        if adarms:
            fused.input_layernorm.dense.weight.copy_(ref.input_layernorm.dense.weight)
            fused.input_layernorm.dense.bias.copy_(ref.input_layernorm.dense.bias)
            fused.post_attention_layernorm.dense.weight.copy_(ref.post_attention_layernorm.dense.weight)
            fused.post_attention_layernorm.dense.bias.copy_(ref.post_attention_layernorm.dense.bias)
        else:
            fused.input_layernorm.weight.copy_(ref.input_layernorm.weight)
            fused.post_attention_layernorm.weight.copy_(ref.post_attention_layernorm.weight)
        fused.self_attn.q_proj.weight.copy_(ref.self_attn.q_proj.weight)
        fused.self_attn.k_proj.weight.copy_(ref.self_attn.k_proj.weight)
        fused.self_attn.v_proj.weight.copy_(ref.self_attn.v_proj.weight)
        fused.self_attn.o_proj.weight.copy_(ref.self_attn.o_proj.weight)
        fused.mlp.gate_proj.weight.copy_(ref.mlp.gate_proj.weight)
        fused.mlp.up_proj.weight.copy_(ref.mlp.up_proj.weight)
        fused.mlp.down_proj.weight.copy_(ref.mlp.down_proj.weight)
    return fused


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------
def report(name, a, b, tol=2e-2):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    rel = d.max().item() / (b.abs().max().item() + 1e-9)
    ok = rel < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:26s} maxd={d.max():.3e} "
          f"meand={d.mean():.3e} rel={rel:.2e}")
    return ok


def run_scenario(name, device, dtype, variant, use_adarms, mask_kind, pos_offset=0):
    print(f"\n==== scenario: {name} ====")
    print(f"     variant={variant} adarms={use_adarms} mask={mask_kind} pos_offset={pos_offset}")
    seq = SEQ if variant == "gemma_2b" else 256  # suffix is short
    ref, cfg = build_reference_layer(device, dtype, variant=variant, use_adarms=use_adarms)
    x, position_ids, pos_emb, attn, cond = make_inputs(
        device, dtype, variant=variant, mask_kind=mask_kind, use_adarms=use_adarms,
        seq=seq, pos_offset=pos_offset)

    # reference fwd + bwd
    xr = x.clone().requires_grad_(True)
    yr = ref_forward(ref, xr, position_ids, pos_emb, attn, adarms_cond=cond)
    yr.float().sum().backward()

    fused = build_fused_layer(ref, cfg, device, dtype)
    xf = x.clone().requires_grad_(True)
    yf = fused(xf, position_ids=position_ids, position_embeddings=pos_emb,
               attention_mask=attn, adarms_cond=cond)
    yf.float().sum().backward()

    ok = report("forward", yf, yr)
    ok &= report("grad_input", xf.grad, xr.grad)
    ref_named = dict(ref.named_parameters())
    for n, p in fused.named_parameters():
        if n in ref_named and ref_named[n].grad is not None and p.grad is not None:
            ok &= report(f"grad:{n}", p.grad, ref_named[n].grad)
    print(f"     -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)
    results = {}
    # Gap-by-gap blocking coverage:
    results["prefix / zero mask"] = run_scenario(
        "prefix, zero mask", device, dtype, "gemma_2b", False, "zero")
    results["prefix / block mask"] = run_scenario(   # Gap 2
        "prefix, block-diagonal mask", device, dtype, "gemma_2b", False, "block")
    results["suffix / adaRMS + block mask"] = run_scenario(  # Gaps 3 + 2, offset pos
        "suffix adaRMS, block mask", device, dtype, "gemma_300m", True, "block",
        pos_offset=SEQ)

    print("\n==== SUMMARY ====")
    all_ok = True
    for k, v in results.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        all_ok &= v
    print(f"\n==== {'ALL PASS' if all_ok else 'FAILURES PRESENT'} ====")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
