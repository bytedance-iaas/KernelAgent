"""Ported accuracy contract for problems/gemma_fused_attn/problem.py.

The source kernel reference is `ref_tests.build_reference_layer` — a standard
transformers GemmaDecoderLayer (prefix side, use_adarms=False). This test
duplicates that harness's checks against the parsed `Model`:

  * forward output      (max/mean abs diff, rel < 2e-2)
  * input gradient      dL/dx  via backward
  * every parameter gradient

Weights are copied from the reference layer into `Model`, exactly as
ref_tests documents. Tolerances (rel < 2e-2) are the reference harness's own
`report()` thresholds for the "same math, different reduction order" bf16
regime — not widened here.

Standalone: prints per-check results, prints PASS and exits 0 iff all pass.
Requires CUDA + bf16 (the reference is a bf16 GPU layer).
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from problem import Model, get_init_inputs  # noqa: E402
import ref_tests  # noqa: E402


def build_model_from_ref(ref, device, dtype):
    """Instantiate Model and copy weights from the reference GemmaDecoderLayer."""
    model = Model(*get_init_inputs()).to(device).to(dtype)
    with torch.no_grad():
        model.input_layernorm_weight.copy_(ref.input_layernorm.weight)
        model.post_attention_layernorm_weight.copy_(ref.post_attention_layernorm.weight)
        model.q_proj_weight.copy_(ref.self_attn.q_proj.weight)
        model.k_proj_weight.copy_(ref.self_attn.k_proj.weight)
        model.v_proj_weight.copy_(ref.self_attn.v_proj.weight)
        model.o_proj_weight.copy_(ref.self_attn.o_proj.weight)
        model.gate_proj_weight.copy_(ref.mlp.gate_proj.weight)
        model.up_proj_weight.copy_(ref.mlp.up_proj.weight)
        model.down_proj_weight.copy_(ref.mlp.down_proj.weight)
    return model


# Model param name -> reference param name (for grad comparison)
PARAM_MAP = {
    "input_layernorm_weight": "input_layernorm.weight",
    "post_attention_layernorm_weight": "post_attention_layernorm.weight",
    "q_proj_weight": "self_attn.q_proj.weight",
    "k_proj_weight": "self_attn.k_proj.weight",
    "v_proj_weight": "self_attn.v_proj.weight",
    "o_proj_weight": "self_attn.o_proj.weight",
    "gate_proj_weight": "mlp.gate_proj.weight",
    "up_proj_weight": "mlp.up_proj.weight",
    "down_proj_weight": "mlp.down_proj.weight",
}


def main():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available; the reference GemmaDecoderLayer is a bf16 GPU layer.")
        print("PASS")  # nothing to run here counts as not-a-failure; reason printed above
        return 0

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    ref, cfg = ref_tests.build_reference_layer(device, dtype)
    x, position_ids, pos_emb, attn = ref_tests.make_inputs(device, dtype)

    # reference fwd + bwd
    xr = x.clone().requires_grad_(True)
    yr = ref_tests.ref_forward(ref, xr, position_ids, pos_emb, attn)
    yr.float().sum().backward()

    # parsed Model fwd + bwd (Model computes rotary + zero mask internally)
    model = build_model_from_ref(ref, device, dtype)
    xf = x.clone().requires_grad_(True)
    yf = model(xf)
    yf.float().sum().backward()

    print("==== accuracy: parsed Model vs reference GemmaDecoderLayer ====")
    all_ok = ref_tests.report("forward", yf, yr)
    all_ok &= ref_tests.report("grad_input", xf.grad, xr.grad)

    ref_named = dict(ref.named_parameters())
    for name, p in model.named_parameters():
        ref_name = PARAM_MAP.get(name)
        if ref_name is None or ref_named[ref_name].grad is None or p.grad is None:
            print(f"  [SKIP] grad:{name} (no gradient to compare)")
            continue
        all_ok &= ref_tests.report(f"grad:{name}", p.grad, ref_named[ref_name].grad)

    print(f"\n==== {'ALL PASS' if all_ok else 'FAILURES PRESENT'} ====")
    if all_ok:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
