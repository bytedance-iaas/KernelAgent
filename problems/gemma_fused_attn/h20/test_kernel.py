"""Correctness test for the Triton fused Gemma decoder layer kernel.

Compares `kernel_function` against the validated pure-PyTorch reference
(`problem.Model`) on the gemma_2b workload, in bf16 on GPU. Weights are copied
out of the Model and passed positionally (matching problem.md's run() order).

Tolerance is the reference harness's bf16 regime ("same math, different
reduction order"): global relative error < 2e-2 and >= 99% of elements within
atol=2e-2 + rtol=2e-2 * |ref|. Exits 0 on PASS.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# The runner may execute a copy from a temp subdir; walk up to locate the
# directories that hold kernel.py (h20/) and problem.py (the problem dir).
_d = HERE
for _ in range(6):
    if os.path.exists(os.path.join(_d, "kernel.py")):
        sys.path.insert(0, _d)
    if os.path.exists(os.path.join(_d, "problem.py")):
        sys.path.insert(0, _d)
    _d = os.path.dirname(_d)

from kernel import kernel_function  # noqa: E402
from problem import Model, get_init_inputs, get_inputs, eps as EPS  # noqa: E402


def main():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        print("PASS")
        return 0

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    model = Model(*get_init_inputs()).to(device).to(dtype).eval()
    x = get_inputs()[0].to(device).to(dtype)

    with torch.no_grad():
        ref = model(x)

        out = kernel_function(
            x,
            model.input_layernorm_weight.data,
            model.q_proj_weight.data,
            model.k_proj_weight.data,
            model.v_proj_weight.data,
            model.o_proj_weight.data,
            model.post_attention_layernorm_weight.data,
            model.gate_proj_weight.data,
            model.up_proj_weight.data,
            model.down_proj_weight.data,
            float(EPS),
        )

    if out.shape != ref.shape:
        print(f"[FAIL] shape mismatch: kernel {tuple(out.shape)} vs ref {tuple(ref.shape)}")
        return 1
    if out.device != x.device:
        print(f"[FAIL] device mismatch: {out.device} vs {x.device}")
        return 1

    o32, r32 = out.float(), ref.float()
    if not torch.isfinite(o32).all():
        print("[FAIL] kernel output has nan/inf")
        return 1

    d = (o32 - r32).abs()
    rel = d.max().item() / (r32.abs().max().item() + 1e-9)
    bound = 2e-2 + 2e-2 * r32.abs()
    matched = 1.0 - (d > bound).float().mean().item()

    print(f"max_abs_err={d.max().item():.3e}  mean_abs_err={d.mean().item():.3e}  "
          f"rel={rel:.3e}  matched={matched:.4f}")

    ok = (rel < 2e-2) and (matched >= 0.99)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
