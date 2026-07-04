#!/usr/bin/env python3
"""
Validate a KernelBench-format problem file.

Checks the format contract and executes one forward pass:
    - defines Model(nn.Module), get_inputs(), get_init_inputs()
    - Model(*get_init_inputs()) constructs
    - Model.forward(*get_inputs()) runs and returns tensor(s)
    - forward uses no non-PyTorch kernel DSLs (triton/cutlass/tilelang imports)

Requires torch at runtime; no other dependencies.

Usage:
    python validate_problem.py --problem 2_Standard_matrix_multiplication_.py \
        [--device auto|cpu|cuda] [--seed 0]

Output (JSON to stdout; exit 0 iff valid):
    {"valid": true, "checks": {...}, "init_inputs": [...reprs...],
     "input_specs": [{shape, dtype}...], "output_specs": [...],
     "param_count": N, "device": "cpu"}
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

FORBIDDEN_IMPORTS = ("triton", "tilelang", "cutlass", "cute")


def _spec(t) -> dict:
    import torch

    if isinstance(t, torch.Tensor):
        return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
    return {"type": type(t).__name__, "repr": repr(t)[:80]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate a KernelBench problem file")
    p.add_argument("--problem", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    result: dict = {"valid": False, "checks": {}}
    checks = result["checks"]

    try:
        import torch

        problem_path = Path(args.problem).resolve()
        source = problem_path.read_text(encoding="utf-8")

        # Static check: the reference must be pure PyTorch
        bad = [
            name
            for name in FORBIDDEN_IMPORTS
            if f"import {name}" in source or f"from {name}" in source
        ]
        checks["pure_pytorch"] = not bad
        if bad:
            result["error"] = f"forbidden kernel-DSL imports in problem file: {bad}"
            print(json.dumps(result, indent=2))
            return 1

        torch.manual_seed(args.seed)
        spec = importlib.util.spec_from_file_location("problem", str(problem_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        for attr in ("Model", "get_inputs", "get_init_inputs"):
            checks[f"has_{attr}"] = hasattr(mod, attr)
        if not all(checks[f"has_{a}"] for a in ("Model", "get_inputs", "get_init_inputs")):
            result["error"] = "missing required symbols"
            print(json.dumps(result, indent=2))
            return 1

        checks["model_is_nn_module"] = issubclass(mod.Model, torch.nn.Module)

        init_inputs = mod.get_init_inputs()
        inputs = mod.get_inputs()
        checks["get_init_inputs_is_list"] = isinstance(init_inputs, (list, tuple))
        checks["get_inputs_is_list"] = isinstance(inputs, (list, tuple))

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = mod.Model(*init_inputs).to(device)
        model.eval()
        dev_inputs = [
            x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs
        ]
        with torch.no_grad():
            out = model(*dev_inputs)
        checks["forward_runs"] = True

        outputs = out if isinstance(out, (list, tuple)) else [out]
        checks["returns_tensor"] = all(
            isinstance(o, torch.Tensor) for o in outputs
        )

        result.update(
            {
                "valid": all(v for v in checks.values()),
                "init_inputs": [_spec(x) for x in init_inputs],
                "input_specs": [_spec(x) for x in inputs],
                "output_specs": [_spec(o) for o in outputs],
                "param_count": sum(p.numel() for p in model.parameters()),
                "device": device,
            }
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-2000:]

    print(json.dumps(result, indent=2))
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    sys.exit(main())
