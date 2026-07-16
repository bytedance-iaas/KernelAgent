"""Correctness gate for the `use_cache=True` K/V export (layer_train.py).

openpi's prefix path is cache-based: `_build_prefix_cache` runs the prefix VLM
with use_cache=True and collects per-layer K/V; `get_suffix_out` feeds that
past_key_values back in so the suffix action tokens attend the cached prefix
without recomputing it. `PrefixTrainFn` used to return only hidden_states, so
the cache stayed empty ("Cache only has 0 layers, attempted to access layer with
index 0").

This checks the surfaced K/V on every path (masked / unmasked x arange / explicit
positions):

  * layout is HF's [B, n_kv, S, hd] — the rope'd K and the *pre-GQA-expansion* V;
  * values match the PyTorch reference (ref_tests._torch_layer's k/v);
  * hidden_states is bit-identical to the use_cache=False call;
  * a real transformers DynamicCache populates (the reported failure);
  * grads flowing back through the exported K/V are correct — an actor training
    the prefix through a cache-consuming suffix depends on this;
  * ref_tests.build_fused_layer's module honours past_key_value/use_cache;
  * PRODUCER -> CONSUMER: a cache built here, fed to attention.py's
    fused_attention (the Sq!=Sk primitive openpi monkeypatches over
    eager_attention_forward), reproduces a full-sequence no-cache forward and its
    grads. This composition is what caught the non-contiguous-q backward bug in
    attention.py (_AttnFn saved uncontiguous q/k/v that the bwd kernels then
    indexed as if contiguous).

Exit 0 on PASS, 1 on FAIL.
"""

import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.dirname(HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from problem import Model, get_init_inputs, get_inputs, eps as EPS  # noqa: E402
from problem import _rms_norm, _rotate_half                         # noqa: E402
import ref_tests as _rt                                             # noqa: E402
from layer_train import PrefixTrainFn                               # noqa: E402

TOL = 2e-2


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()


def _ref_qkv(x, w_ln, wq, wk, wv, eps, meta, position_ids):
    """Reference rope'd Q/K and V — ref_tests._torch_layer L200-208, pre
    repeat_interleave. q:[B,n_heads,S,hd], k/v:[B,n_kv,S,hd] (a cache stores the
    n_kv heads, not the GQA-expanded n_heads)."""
    n_heads, n_kv, hd = meta
    B, S, _ = x.shape
    if position_ids is None:
        position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S)
    h = _rms_norm(x, w_ln, eps)
    k = F.linear(h, wk).view(B, S, n_kv, hd).transpose(1, 2)
    v = F.linear(h, wv).view(B, S, n_kv, hd).transpose(1, 2)
    cos, sin = _rt._rotary(position_ids, hd, x.dtype)
    cos, sin = cos[:, None], sin[:, None]
    k = k * cos + _rotate_half(k) * sin
    q = None
    if wq is not None:
        q = F.linear(h, wq).view(B, S, n_heads, hd).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
    return q, k, v


def _ref_kv(x, w_ln, wk, wv, eps, meta, position_ids):
    """Reference rope'd K and V, [B, n_kv, S, hd]."""
    _, k, v = _ref_qkv(x, w_ln, None, wk, wv, eps, meta, position_ids)
    return k, v


def main():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available\nPASS")
        return 0
    torch.manual_seed(0)
    dev, dt = torch.device("cuda"), torch.bfloat16
    model = Model(*get_init_inputs()).to(dev).to(dt).eval()
    xin = get_inputs()[0].to(dev).to(dt)
    W = [model.input_layernorm_weight.data, model.q_proj_weight.data, model.k_proj_weight.data,
         model.v_proj_weight.data, model.o_proj_weight.data, model.post_attention_layernorm_weight.data,
         model.gate_proj_weight.data, model.up_proj_weight.data, model.down_proj_weight.data]
    eps = float(EPS)
    ii = get_init_inputs()
    meta = (ii[2], ii[3], ii[4])
    n_heads, n_kv, hd = meta
    B, S, _ = xin.shape
    ok = True

    # additive prefix-LM style mask + padded positions, to cover the masked path
    mask = torch.zeros(B, 1, S, S, device=dev, dtype=dt)
    mask[:, :, :, S // 2:] = torch.finfo(dt).min
    pos = torch.arange(S, device=dev).unsqueeze(0).expand(B, S).contiguous()

    print("==== use_cache=True K/V export ====")
    for tag, m, p in [("unmasked/arange", None, None), ("unmasked/pos", None, pos),
                      ("masked/arange", mask, None), ("masked/pos", mask, pos)]:
        out_nc = PrefixTrainFn.apply(xin, *W, eps, meta, m, p)
        out, k, v = PrefixTrainFn.apply(xin, *W, eps, meta, m, p, True)

        shape_ok = tuple(k.shape) == (B, n_kv, S, hd) == tuple(v.shape)
        same_out = torch.equal(out, out_nc)
        kr, vr = _ref_kv(xin, W[0], W[2], W[3], eps, meta, p)
        rk, rv = _rel(k, kr), _rel(v, vr)
        good = shape_ok and same_out and rk < TOL and rv < TOL
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {tag:<16} k{tuple(k.shape)} "
              f"layout={'ok' if shape_ok else 'BAD'} out_identical={same_out} "
              f"k_rel={rk:.2e} v_rel={rv:.2e}")

    # ---- the reported failure: does a real HF Cache actually populate? ----
    try:
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        _, k, v = PrefixTrainFn.apply(xin, *W, eps, meta, None, None, True)
        cache.update(k, v, 0)
        ck, cv = cache[0]
        good = len(cache) == 1 and torch.equal(ck, k) and torch.equal(cv, v)
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {'DynamicCache':<16} layers={len(cache)} "
              f"(was 0 -> 'attempted to access layer with index 0')")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] DynamicCache raised: {type(e).__name__}: {e}")

    # ---- grads through the exported K/V vs the torch reference ----
    # loss touches out AND k AND v, so dk_cache/dv_cache are both non-zero.
    print("==== grads through exported K/V ====")
    gk = torch.randn(B, n_kv, S, hd, device=dev, dtype=dt)
    gv = torch.randn(B, n_kv, S, hd, device=dev, dtype=dt)
    names = ["grad_input", "input_layernorm.weight", "q_proj", "k_proj", "v_proj",
             "o_proj", "post_attention_layernorm.weight", "gate_proj", "up_proj", "down_proj"]

    for tag, m, p in [("unmasked", None, None), ("masked", mask, pos)]:
        xr = xin.clone().requires_grad_(True)
        Wr = [w.detach().clone().requires_grad_(True) for w in W]
        yr = _rt._torch_layer(xr, *Wr, None, None, None, None, None, m, p, eps, meta)
        krr, vrr = _ref_kv(xr, Wr[0], Wr[2], Wr[3], eps, meta, p)
        (yr.float().sum() + (krr * gk).float().sum() + (vrr * gv).float().sum()).backward()

        xf = xin.clone().requires_grad_(True)
        Wf = [w.detach().clone().requires_grad_(True) for w in W]
        yf, kf, vf = PrefixTrainFn.apply(xf, *Wf, eps, meta, m, p, True)
        (yf.float().sum() + (kf * gk).float().sum() + (vf * gv).float().sum()).backward()

        for n, a, b in zip(names, [xf.grad] + [w.grad for w in Wf],
                           [xr.grad] + [w.grad for w in Wr]):
            r = _rel(a, b)
            good = r < TOL
            ok &= good
            print(f"  [{'PASS' if good else 'FAIL'}] {tag}/{n:<32} rel={r:.2e}")

    ok &= _module_checks()
    ok &= _producer_consumer_checks()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def _producer_consumer_checks():
    """End-to-end: a cache PRODUCED by layer_train feeds the CONSUMER in
    attention.py (fused_attention, the Sq!=Sk primitive openpi monkeypatches over
    eager_attention_forward).

    The invariant is openpi's reason for caching at all: running the suffix
    against a cached prefix must equal one full-sequence forward with no cache.
    Nothing else covers this — test_kv_cache checks the producer against a torch
    reference, test_attention checks the consumer against its own reference, and
    the concat that joins them lives in openpi. So the two halves agree only by
    inspection until something runs them together.

    Catches: cache layout ([B,n_kv,S,hd] vs the consumer's [B,Hkv,Sk,D]), RoPE
    position offset on the suffix, prefix/suffix concat order, GQA head count,
    and the rectangular [B,1,Sq,Sk] mask slice.
    """
    from attention import fused_attention, _torch_attention
    dev, dt = torch.device("cuda"), torch.bfloat16
    ok = True
    print("==== producer -> consumer (cache == no-cache full forward) ====")
    torch.manual_seed(0)

    ii = get_init_inputs()
    H, meta = ii[0], (ii[2], ii[3], ii[4])
    n_heads, n_kv, hd = meta
    eps = float(EPS)
    scale = hd ** -0.5
    B, P, S = 2, 512, 256          # prefix len, suffix len
    total = P + S

    model = Model(*ii).to(dev).to(dt).eval()
    W = [model.input_layernorm_weight.data, model.q_proj_weight.data, model.k_proj_weight.data,
         model.v_proj_weight.data, model.o_proj_weight.data, model.post_attention_layernorm_weight.data,
         model.gate_proj_weight.data, model.up_proj_weight.data, model.down_proj_weight.data]
    x = torch.randn(B, total, H, device=dev, dtype=dt)
    pos = torch.arange(total, device=dev).unsqueeze(0).expand(B, total).contiguous()

    # openpi prefix-LM mask: prefix attends prefix bidirectionally, suffix attends
    # the whole prefix + causally among itself. Prefix K/V therefore never depend
    # on the suffix, which is what makes caching them valid.
    att = torch.zeros(B, total, dtype=torch.long, device=dev)
    att[:, P:] = 1
    cs = torch.cumsum(att, dim=1)
    att2d = cs[:, None, :] <= cs[:, :, None]
    mask_full = torch.where(att2d[:, None, :, :], 0.0, _rt.MASK_NEG).to(dt)

    # ---- path A: one full-sequence forward, no cache; keep the suffix rows ----
    xa = x.clone().requires_grad_(True)
    Wa = [w.detach().clone().requires_grad_(True) for w in W]
    qf, kf, vf = _ref_qkv(xa, Wa[0], Wa[1], Wa[2], Wa[3], eps, meta, pos)
    out_a = _torch_attention(qf, kf, vf, mask_full, scale)[:, P:]   # [B,S,Hq,hd]

    # ---- path B: producer builds the prefix cache, consumer attends it ----
    xb = x.clone().requires_grad_(True)
    Wb = [w.detach().clone().requires_grad_(True) for w in W]
    _, kc, vc = PrefixTrainFn.apply(
        xb[:, :P].contiguous(), *Wb, eps, meta,
        mask_full[:, :, :P, :P].contiguous(), pos[:, :P].contiguous(), True)
    # suffix q/k/v carry absolute positions P..total-1 (the RoPE offset)
    qs, ks, vs = _ref_qkv(xb[:, P:].contiguous(), Wb[0], Wb[1], Wb[2], Wb[3],
                          eps, meta, pos[:, P:])
    k_cat = torch.cat([kc, ks], dim=2)                 # [B,n_kv,total,hd]
    v_cat = torch.cat([vc, vs], dim=2)
    out_b = fused_attention(qs, k_cat, v_cat, mask_full[:, :, P:, :].contiguous(), scale)

    # the cache must be exactly the prefix slice a full forward would compute
    rk, rv = _rel(kc, kf[:, :, :P]), _rel(vc, vf[:, :, :P])
    good = rk < TOL and rv < TOL
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {'cache == full K/V[:P]':<26} k_rel={rk:.2e} v_rel={rv:.2e}")

    ro = _rel(out_b, out_a)
    good = tuple(out_b.shape) == tuple(out_a.shape) and ro < TOL
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {'suffix out: cached==full':<26} rel={ro:.2e} "
          f"shape={tuple(out_b.shape)}")

    # ---- grads must flow back through the cache into the prefix weights ----
    # Path B reaches the prefix ONLY via kc/vc (PrefixTrainFn's hidden_states is
    # unused here, so its grad_out is zero) — so this is the real test of the
    # dk_cache/dv_cache fold, with grads the composition produced rather than
    # synthetic ones.
    gout = torch.randn_like(out_a, dtype=torch.float32)
    (out_a.float() * gout).sum().backward()
    (out_b.float() * gout).sum().backward()

    # only the attention-path params: path A's reference has no o_proj/MLP, and
    # path B's are zero (grad_out=0), so comparing them would be vacuous.
    for name, ia in [("grad_input", None), ("input_layernorm.weight", 0),
                     ("q_proj", 1), ("k_proj", 2), ("v_proj", 3)]:
        a = xb.grad if ia is None else Wb[ia].grad
        b = xa.grad if ia is None else Wa[ia].grad
        r = _rel(a, b)
        good = r < TOL
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<26} rel={r:.2e}")
    return ok


def _module_checks():
    """The module-level API openpi calls: ref_tests.build_fused_layer's
    FusedGemmaLayer(..., past_key_value=cache, use_cache=True) writing per-layer
    K/V into a shared Cache — i.e. _build_prefix_cache's actual shape."""
    from transformers.cache_utils import DynamicCache
    dev, dt = torch.device("cuda"), torch.bfloat16
    ok = True
    print("==== FusedGemmaLayer cache plumbing ====")
    ref, cfg = _rt.build_reference_layer(dev, dt, variant="gemma_2b", use_adarms=False)
    x, position_ids, pos_emb, attn, _ = _rt.make_inputs(
        dev, dt, variant="gemma_2b", mask_kind="zero", use_adarms=False)

    # unchanged signature: no cache args -> bare tensor
    layer0 = _rt.build_fused_layer(ref, cfg, dev, dt, layer_idx=0)
    y_plain = layer0(x, position_ids=position_ids, position_embeddings=pos_emb,
                     attention_mask=attn)
    good = isinstance(y_plain, torch.Tensor)
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {'no cache args':<22} returns bare tensor")

    # a 2-layer stack sharing one Cache == what _build_prefix_cache does
    layer1 = _rt.build_fused_layer(ref, cfg, dev, dt, layer_idx=1)
    cache = DynamicCache()
    h = x
    with torch.no_grad():
        for lyr in (layer0, layer1):
            h = lyr(h, position_ids=position_ids, position_embeddings=pos_emb,
                    attention_mask=attn, past_key_value=cache, use_cache=True)
    good = isinstance(h, torch.Tensor) and len(cache) == 2
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {'2-layer stack':<22} cache layers={len(cache)} "
          f"(expect 2), returns bare tensor")

    # layer 0's cached K/V must be the reference K/V for layer 0's input
    kr, vr = _ref_kv(x, ref.input_layernorm.weight, ref.self_attn.k_proj.weight,
                     ref.self_attn.v_proj.weight, float(cfg.rms_norm_eps),
                     (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim),
                     position_ids)
    ck, cv = cache[0]
    rk, rv = _rel(ck, kr), _rel(cv, vr)
    good = rk < TOL and rv < TOL
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {'cache[0] vs reference':<22} k_rel={rk:.2e} v_rel={rv:.2e}")

    # adaRMS + use_cache must fail loudly, not hand back an empty cache.
    # Driven through _FusedFn directly: build_reference_layer(use_adarms=True) is
    # broken independently of this change (GemmaRMSNorm has no .dense), so going
    # via build_fused_layer would test that bug instead of this guard.
    H = cfg.hidden_size
    cond = torch.randn(x.shape[0], H, device=dev, dtype=dt)
    dense_w = torch.randn(3 * H, H, device=dev, dtype=dt)
    dense_b = torch.randn(3 * H, device=dev, dtype=dt)
    a = layer0.self_attn
    m = layer0.mlp
    try:
        _rt._FusedFn.apply(
            x, None, a.q_proj.weight, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight,
            None, m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight,
            dense_w, dense_b, dense_w, dense_b, cond, attn, position_ids,
            float(cfg.rms_norm_eps),
            (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim), True)
        good, detail = False, "no error raised"
    except NotImplementedError as e:  # noqa: BLE001
        good, detail = True, type(e).__name__
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {'adaRMS + use_cache':<22} {detail}")
    return ok


if __name__ == "__main__":
    sys.exit(main())
