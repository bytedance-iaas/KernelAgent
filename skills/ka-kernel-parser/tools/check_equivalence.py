#!/usr/bin/env python3
"""
Numerically compare a KernelBench problem Model against the source kernel it
was parsed from.

Runs Model(*get_init_inputs()).forward(*get_inputs()) and the source kernel
entry point on the SAME inputs (same seed), and reports allclose + max
abs/rel error. Requires torch; the source kernel's DSL (triton/...) must be
importable and, for GPU DSLs, a CUDA device present.

Usage:
    python check_equivalence.py --problem problem.py --kernel kernel.py \
        [--entry kernel_function] [--device auto] [--dtype auto] \
        [--rtol R --atol A] [--with-model-params]

Entry contract: --entry names a callable in the kernel file taking the same
positional tensors as Model.forward (add --with-model-params to append the
Model's weight/bias tensors in module order, for kernels that fold module
parameters into their arguments). For source kernels that don't fit this
shape (e.g. a class, or a C++/CUTLASS kernel without Python bindings), write
a small ad-hoc harness instead of using this tool.

Output (JSON; exit 0 iff equivalent):
    {"equivalent": true, "max_abs_err": ..., "max_rel_err": ...,
     "rtol": ..., "atol": ..., "dtype": ..., "device": ...}
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

# Default tolerances by working dtype
TOLERANCES = {
    "float32": (1e-4, 1e-4),
    "bfloat16": (2e-2, 2e-2),
    "float16": (1e-2, 1e-2),
}


def _import(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, str(Path(path).resolve()))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Problem-vs-kernel equivalence check")
    p.add_argument("--problem", required=True)
    p.add_argument("--kernel", required=True)
    p.add_argument("--entry", default="kernel_function")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--dtype", default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
        help="auto: infer from dtype names in the kernel source, else float32",
    )
    p.add_argument("--rtol", type=float, default=None)
    p.add_argument("--atol", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--with-model-params", action="store_true",
        help="Append the Model's weight/bias tensors to the kernel arguments",
    )
    args = p.parse_args(argv)

    result: dict = {"equivalent": False}
    try:
        import torch

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        kernel_source = Path(args.kernel).read_text(encoding="utf-8")
        if args.dtype != "auto":
            dtype = getattr(torch, args.dtype)
        else:
            dtype = torch.float32
            for name in ("bfloat16", "float16"):
                if name in kernel_source:
                    dtype = getattr(torch, name)
                    break

        rtol, atol = TOLERANCES[str(dtype).replace("torch.", "")]
        if args.rtol is not None:
            rtol = args.rtol
        if args.atol is not None:
            atol = args.atol

        torch.manual_seed(args.seed)
        problem = _import("problem", args.problem)
        kernel_mod = _import("kernel", args.kernel)
        entry = getattr(kernel_mod, args.entry)

        inputs = problem.get_inputs()
        model = problem.Model(*problem.get_init_inputs()).to(device)
        if dtype in (torch.bfloat16, torch.float16):
            model = model.to(dtype)
        model.eval()

        dev_inputs = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x = x.to(device)
                if x.is_floating_point():
                    x = x.to(dtype)
            dev_inputs.append(x)

        kernel_args = list(dev_inputs)
        if args.with_model_params:
            seen: set[int] = set()
            for module in model.modules():
                for attr in ("weight", "bias"):
                    t = getattr(module, attr, None)
                    if isinstance(t, torch.Tensor) and id(t) not in seen:
                        seen.add(id(t))
                        kernel_args.append(t.detach())

        with torch.no_grad():
            ref = model(*dev_inputs)
            got = entry(*kernel_args)

        refs = ref if isinstance(ref, (list, tuple)) else [ref]
        gots = got if isinstance(got, (list, tuple)) else [got]
        if len(refs) != len(gots):
            raise ValueError(f"output arity mismatch: model {len(refs)} vs kernel {len(gots)}")

        max_abs = 0.0
        max_rel = 0.0
        equivalent = True
        for r, g in zip(refs, gots):
            r32, g32 = r.float(), g.float()
            if r32.shape != g32.shape:
                raise ValueError(f"shape mismatch: model {list(r32.shape)} vs kernel {list(g32.shape)}")
            abs_err = (r32 - g32).abs()
            max_abs = max(max_abs, abs_err.max().item())
            denom = r32.abs().clamp_min(1e-6)
            max_rel = max(max_rel, (abs_err / denom).max().item())
            equivalent = equivalent and torch.allclose(r32, g32, rtol=rtol, atol=atol)

        result.update(
            {
                "equivalent": equivalent,
                "max_abs_err": max_abs,
                "max_rel_err": max_rel,
                "rtol": rtol,
                "atol": atol,
                "dtype": str(dtype).replace("torch.", ""),
                "device": device,
                "num_outputs": len(refs),
            }
        )
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-2000:]

    print(json.dumps(result, indent=2))
    return 0 if result.get("equivalent") else 1


if __name__ == "__main__":
    sys.exit(main())
