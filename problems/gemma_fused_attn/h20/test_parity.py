"""Portable fwd+bwd parity gate for the fused Gemma layer kernel.

Runs WITHOUT openpi's patched transformers (uses the pure-torch `_torch_layer`
in ref_tests as the reference), so it works in any env with torch+triton. It is
a fast smoke gate; ref_tests.py is the gold check against the real
transformers GemmaDecoderLayer (fwd + bwd + every param grad).

Covers the three delivered capabilities on small dims:
  * prefix (standard RMSNorm), zero mask
  * prefix, block-diagonal additive mask
  * suffix (adaRMS + gated residual), block mask, offset RoPE positions

For each: forward parity (fast Triton kernel vs torch math) and that the
autograd-wrapped module produces grads matching autograd through the same math.

Exit 0 on PASS, 1 on FAIL.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.dirname(HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from kernel import kernel_function, HEAD_DIM          # noqa: E402
from ref_tests import _torch_layer, MASK_NEG, _FusedFn  # noqa: E402

TOL = 2e-2


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()


def _block_mask(B, S, dev, dt):
    """prefix-LM: first S//2 tokens bidirectional, rest causal (openpi style)."""
    att = torch.zeros(B, S, dtype=torch.long, device=dev)
    att[:, S // 2:] = 1
    c = torch.cumsum(att, dim=1)
    a2 = c[:, None, :] <= c[:, :, None]
    return torch.where(a2[:, None, :, :], 0.0, MASK_NEG).to(dt)


def _run(name, H, I, heads, kv, S, B, adarms, mask_kind, pos_offset, dev, dt):
    hd = HEAD_DIM
    q_dim, kv_dim = heads * hd, kv * hd
    g = lambda *s: torch.randn(*s, device=dev, dtype=dt) * 0.02
    x = g(B, S, H)
    wq, wk, wv, wo = g(q_dim, H), g(kv_dim, H), g(kv_dim, H), g(H, q_dim)
    wg, wu, wd = g(I, H), g(I, H), g(H, I)
    if adarms:
        w_ln = w_pln = None
        di_w, di_b, dp_w, dp_b = g(3 * H, H), g(3 * H), g(3 * H, H), g(3 * H)
        cond = g(B, H)
        input_dense, post_dense = (di_w, di_b), (dp_w, dp_b)
    else:
        w_ln, w_pln = g(H), g(H)
        di_w = di_b = dp_w = dp_b = cond = None
        input_dense = post_dense = None
    meta = (heads, kv, hd)

    pos = (pos_offset + torch.arange(S, device=dev)).unsqueeze(0).expand(B, -1).contiguous()
    mask = _block_mask(B, S, dev, dt) if mask_kind == "block" else \
        torch.zeros(B, 1, S, S, device=dev, dtype=dt)
    mask_torch = None if mask_kind == "zero" else mask
    eps = 1e-6

    # forward: fast kernel vs torch math
    yk = kernel_function(x, w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd, eps,
                         attention_mask=mask, position_ids=pos, adarms_cond=cond,
                         input_dense=input_dense, post_dense=post_dense)
    yt = _torch_layer(x, w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd,
                      di_w, di_b, dp_w, dp_b, cond, mask_torch, pos, eps, meta)
    fwd = _rel(yk, yt)

    # backward: autograd-wrapped module vs autograd through the same math
    xk = x.clone().requires_grad_(True)
    yf = _FusedFn.apply(xk, w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd,
                        di_w, di_b, dp_w, dp_b, cond, mask, pos, eps, meta)
    yf.float().sum().backward()
    xt = x.clone().requires_grad_(True)
    yt2 = _torch_layer(xt, w_ln, wq, wk, wv, wo, w_pln, wg, wu, wd,
                       di_w, di_b, dp_w, dp_b, cond, mask_torch, pos, eps, meta)
    yt2.float().sum().backward()
    gi = _rel(xk.grad, xt.grad)

    ok = fwd < TOL and gi < TOL and torch.isfinite(yf).all().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:26s} fwd_rel={fwd:.2e} grad_input_rel={gi:.2e}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available\nPASS")
        return 0
    torch.manual_seed(0)
    dev, dt = torch.device("cuda"), torch.bfloat16
    ok = True
    print("==== fused gemma layer parity (portable, pure-torch reference) ====")
    ok &= _run("prefix / zero mask", 128, 256, 2, 1, 40, 2, False, "zero", 0, dev, dt)
    ok &= _run("prefix / block mask", 128, 256, 2, 1, 40, 2, False, "block", 0, dev, dt)
    ok &= _run("suffix / adaRMS+block", 96, 192, 2, 1, 48, 2, True, "block", 968, dev, dt)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
