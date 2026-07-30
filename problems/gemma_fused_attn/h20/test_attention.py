"""Parity gate for the cross-expert / joint flash attention (attention.py).

Checks `fused_attention` (fwd + bwd) against:
  * the pure-torch reference `_torch_attention` (always; portable), and
  * the real `transformers ... eager_attention_forward` (gold; if importable).

Covers the shapes the actor path needs:
  * square self-attention  Sq == Sk  (joint forward), block-diagonal mask
  * rectangular             Sq  < Sk  (suffix recompute vs prefix KV cache)
  * GQA groups (Hq > Hkv) and the multi-kv-head general case

Exit 0 on PASS, 1 on FAIL.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from attention import fused_attention, _torch_attention  # noqa: E402

MASK_NEG = -2.3819763e38
TOL = 2e-2

try:
    from transformers.models.gemma import modeling_gemma
    _HAVE_EAGER = hasattr(modeling_gemma, "eager_attention_forward")
except Exception:
    _HAVE_EAGER = False


class _Mod:
    """Minimal stand-in for GemmaAttention (what eager_attention_forward reads)."""
    def __init__(self, groups):
        self.num_key_value_groups = groups
        self.training = False
        self.attention_dropout = 0.0


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()


def _square_mask(B, S, dev, dt):
    att = torch.zeros(B, S, dtype=torch.long, device=dev)
    att[:, S // 2:] = 1
    c = torch.cumsum(att, dim=1)
    a2 = c[:, None, :] <= c[:, :, None]
    return torch.where(a2[:, None, :, :], 0.0, MASK_NEG).to(dt)


def _cross_mask(B, Sq, Sk, dev, dt):
    """Suffix (len Sq) attends all prefix (len Sk-Sq) + causal among suffix."""
    P = Sk - Sq
    m = torch.zeros(B, 1, Sq, Sk, device=dev, dtype=dt)
    causal = torch.triu(torch.ones(Sq, Sq, device=dev), diagonal=1).bool()
    m[:, 0, :, P:] = torch.where(causal, torch.tensor(MASK_NEG, device=dev, dtype=dt),
                                 torch.zeros((), device=dev, dtype=dt))
    return m


def _run(name, B, Hq, Hkv, Sq, Sk, D, mask, dev, dt):
    groups = Hq // Hkv
    scale = D ** -0.5
    g = lambda *s: torch.randn(*s, device=dev, dtype=dt)
    q, k, v = g(B, Hq, Sq, D), g(B, Hkv, Sk, D), g(B, Hkv, Sk, D)

    # forward: fast kernel vs torch reference
    ok = True
    of = fused_attention(q, k, v, mask, scale)
    ot = _torch_attention(q, k, v, mask, scale)
    ok &= _rel(of, ot) < TOL and torch.isfinite(of).all().item()
    r_ref = _rel(of, ot)

    # forward + grads vs the REAL eager path (gold)
    r_eag = gq = gk = gv = float("nan")
    if _HAVE_EAGER:
        qe = q.clone().requires_grad_(True)
        ke = k.clone().requires_grad_(True)
        ve = v.clone().requires_grad_(True)
        oe, _ = modeling_gemma.eager_attention_forward(_Mod(groups), qe, ke, ve, mask, scale)
        oe.float().sum().backward()

        qf = q.clone().requires_grad_(True)
        kf = k.clone().requires_grad_(True)
        vf = v.clone().requires_grad_(True)
        of2 = fused_attention(qf, kf, vf, mask, scale)
        of2.float().sum().backward()

        r_eag = _rel(of2, oe)
        gq, gk, gv = _rel(qf.grad, qe.grad), _rel(kf.grad, ke.grad), _rel(vf.grad, ve.grad)
        ok &= r_eag < TOL and max(gq, gk, gv) < TOL

    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name:30s} fwd(torch)={r_ref:.2e} "
          f"fwd(eager)={r_eag:.2e} grad(q,k,v)=({gq:.1e},{gk:.1e},{gv:.1e})")
    return ok


def main():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available\nPASS")
        return 0
    torch.manual_seed(0)
    dev, dt = torch.device("cuda"), torch.bfloat16
    ok = True
    print("==== cross-expert / joint attention parity ====")
    print(f"     eager gold check: {'on' if _HAVE_EAGER else 'OFF (transformers gemma unavailable)'}")
    B, D = 2, 256
    # joint forward: square self-attention, GQA 8q/1kv, block mask
    ok &= _run("square Sq==Sk, GQA 8/1", B, 8, 1, 192, 192, D,
               _square_mask(B, 192, dev, dt), dev, dt)
    # suffix recompute vs prefix KV cache: rectangular
    ok &= _run("rect Sq<Sk (kv-cache), 8/1", B, 8, 1, 32, 224, D,
               _cross_mask(B, 32, 224, dev, dt), dev, dt)
    # multi-kv-head general case
    ok &= _run("square, GQA 8/2", B, 8, 2, 128, 128, D,
               _square_mask(B, 128, dev, dt), dev, dt)
    # no-mask sanity
    ok &= _run("square, no mask, 8/1", B, 8, 1, 128, 128, D, None, dev, dt)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
