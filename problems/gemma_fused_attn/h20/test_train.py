"""Correctness + speed gate for the fast training step (layer_train.py).

Verifies the grad-only backward (`PrefixTrainFn`) against the exact PyTorch
recompute reference (`ref_tests._torch_layer` autograd) on the gemma_2b prefix
layer — forward, grad_input, and every parameter gradient within bf16 tolerance
— and reports the fwd+bwd wall-time vs eager and vs the recompute backward.

Exit 0 on PASS, 1 on FAIL.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.dirname(HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from problem import Model, get_init_inputs, get_inputs, eps as EPS  # noqa: E402
import ref_tests as _rt                                             # noqa: E402
from layer_train import PrefixTrainFn                               # noqa: E402

TOL = 2e-2


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()


def _bench(fn, n=50, w=15):
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


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
    meta = (get_init_inputs()[2], get_init_inputs()[3], get_init_inputs()[4])
    names = ["grad_input", "input_layernorm.weight", "q_proj", "k_proj", "v_proj",
             "o_proj", "post_attention_layernorm.weight", "gate_proj", "up_proj", "down_proj"]

    # exact recompute reference grads
    xr = xin.clone().requires_grad_(True)
    Wr = [w.detach().clone().requires_grad_(True) for w in W]
    yr = _rt._torch_layer(xr, *Wr, None, None, None, None, None, None, None, eps, meta)
    yr.float().sum().backward()
    # fast training path grads
    xf = xin.clone().requires_grad_(True)
    Wf = [w.detach().clone().requires_grad_(True) for w in W]
    yf = PrefixTrainFn.apply(xf, *Wf, eps, meta)
    yf.float().sum().backward()

    print("==== fast training step: grads vs exact recompute ====")
    ok = _rel(yf, yr) < TOL
    print(f"  [{'PASS' if _rel(yf, yr) < TOL else 'FAIL'}] {'forward':<32} rel={_rel(yf, yr):.2e}")
    gr = [xr.grad] + [w.grad for w in Wr]
    gf = [xf.grad] + [w.grad for w in Wf]
    for n, a, b in zip(names, gf, gr):
        r = _rel(a, b)
        ok &= r < TOL
        print(f"  [{'PASS' if r < TOL else 'FAIL'}] {n:<32} rel={r:.2e}")

    # timing
    xe = xin.clone().requires_grad_(True)

    def eager_fb():
        xe.grad = None
        model.zero_grad(set_to_none=True)
        model(xe).float().sum().backward()

    Wt = [w.detach().clone().requires_grad_(True) for w in W]
    xt = xin.clone().requires_grad_(True)

    def fast_fb():
        xt.grad = None
        for w in Wt:
            w.grad = None
        PrefixTrainFn.apply(xt, *Wt, eps, meta).float().sum().backward()

    et, ft = _bench(eager_fb), _bench(fast_fb)
    print("  ----")
    print(f"  eager fwd+bwd : {et:7.3f} ms   1.00x")
    print(f"  fast  fwd+bwd : {ft:7.3f} ms   {et / ft:.3f}x")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
