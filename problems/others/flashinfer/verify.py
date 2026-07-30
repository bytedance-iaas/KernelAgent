#!/usr/bin/env python3
"""Standalone verifier for KernelAgent-generated flashinfer-bench solutions.

Mirrors the behaviour of verify.py in mit-han-lab/mlsys2026-flashinfer-contest:
  - Loads a solution.json
  - Extracts kernel sources to a temp directory
  - Imports kernel_function
  - Generates inputs from our DefinitionSpec
  - Runs the reference Model and the kernel
  - Checks correctness (torch.allclose)
  - Benchmarks both and reports speedup_factor

Usage:
    python flashinfer/verify.py --solution flashinfer_solutions/<def>/solution.json
    python flashinfer/verify.py --solution flashinfer_solutions/<def>/solution.json \\
        --warmup 25 --iters 100

Exit code 0 = passed, 1 = failed.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_kernel_function(sources: list[dict[str, str]], entry_point: str):
    """Extract sources to a temp dir and import the entry function."""
    file_name, func_name = entry_point.split("::")
    with tempfile.TemporaryDirectory(prefix="fib_verify_") as tmpdir:
        pkg_dir = Path(tmpdir)
        for src in sources:
            out = pkg_dir / src["path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(src["content"])

        # Add pkg_dir to sys.path so cross-file imports work
        sys.path.insert(0, str(pkg_dir))
        try:
            spec = importlib.util.spec_from_file_location(
                "fib_kernel", pkg_dir / file_name
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = getattr(mod, func_name)
        finally:
            sys.path.pop(0)

    return fn


def _time_fn(fn, args: list, warmup: int, iters: int, device: str = "cuda") -> float:
    """Return mean latency in ms over `iters` runs after `warmup` warmup runs."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()

    # Measure
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            with torch.no_grad():
                fn(*args)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                fn(*args)
        return (time.perf_counter() - t0) / iters * 1000


def _get_definition_spec(definition_key: str):
    """Look up our DefinitionSpec by key."""
    # Allow running from any working directory
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from flashinfer.definitions import DEFINITIONS
    if definition_key not in DEFINITIONS:
        raise ValueError(
            f"Unknown definition '{definition_key}'.\n"
            f"Available: {list(DEFINITIONS.keys())}"
        )
    return DEFINITIONS[definition_key]


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

def verify(
    solution_path: Path,
    warmup: int = 25,
    iters: int = 100,
    verbose: bool = True,
) -> bool:
    solution_path = Path(solution_path)
    sol = json.loads(solution_path.read_text())

    definition_key  = sol["definition"]
    author          = sol.get("author", "?")
    name            = sol.get("name", definition_key)
    entry_point     = sol["spec"]["entry_point"]
    dst_passing     = sol["spec"].get("destination_passing_style", False)
    sources         = sol["sources"]

    if verbose:
        print("=" * 70)
        print(f"FlashInfer Solution Verifier")
        print("=" * 70)
        print(f"  Solution : {name}")
        print(f"  Author   : {author}")
        print(f"  Definition: {definition_key}")
        print(f"  Entry    : {entry_point}")
        print(f"  Hardware : {sol['spec'].get('target_hardware', '?')}")
        print(f"  DPS      : {dst_passing}")
        print()

    # -----------------------------------------------------------------------
    # Load definition & generate inputs
    # -----------------------------------------------------------------------
    spec = _get_definition_spec(definition_key)

    from flashinfer.adapter import FlashInferProblemAdapter
    adapter = FlashInferProblemAdapter(spec)

    # problem.py get_inputs() generates tensors on cuda
    with tempfile.TemporaryDirectory(prefix="fib_problem_") as tmpdir:
        adapter.write_problem_dir(Path(tmpdir))
        sys.path.insert(0, tmpdir)
        try:
            import importlib as _il
            prob = _il.import_module("problem")
            inputs = prob.get_inputs()
            model  = prob.Model(*prob.get_init_inputs()).cuda().eval()
        finally:
            sys.path.pop(0)
            # Remove cached module so next import is fresh
            for key in [k for k in sys.modules if k.startswith("problem")]:
                del sys.modules[key]

    device = "cuda" if inputs and isinstance(inputs[0], torch.Tensor) and inputs[0].is_cuda else "cpu"

    # -----------------------------------------------------------------------
    # Reference output
    # -----------------------------------------------------------------------
    if verbose:
        print("Computing reference output...")
    with torch.no_grad():
        ref_out = model(*inputs)
    if isinstance(ref_out, torch.Tensor):
        ref_outs = [ref_out]
    else:
        ref_outs = list(ref_out)

    # -----------------------------------------------------------------------
    # Load kernel function
    # -----------------------------------------------------------------------
    if verbose:
        print("Loading kernel function...")

    # Re-extract sources fresh (avoids tmpdir-gone issues)
    with tempfile.TemporaryDirectory(prefix="fib_kernel_") as kdir:
        pkg = Path(kdir)
        file_name, func_name = entry_point.split("::")
        for src in sources:
            out = pkg / src["path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(src["content"])

        sys.path.insert(0, str(pkg))
        try:
            kspec = importlib.util.spec_from_file_location("fib_kernel", pkg / file_name)
            kmod  = importlib.util.module_from_spec(kspec)
            kspec.loader.exec_module(kmod)
            kernel_fn = getattr(kmod, func_name)
        finally:
            sys.path.pop(0)

        # -----------------------------------------------------------------------
        # Correctness check
        # -----------------------------------------------------------------------
        if verbose:
            print("Checking correctness...")
        try:
            with torch.no_grad():
                kernel_out = kernel_fn(*inputs)
        except Exception as e:
            print(f"  FAIL — kernel raised exception: {e}")
            return False

        if isinstance(kernel_out, torch.Tensor):
            kernel_outs = [kernel_out]
        elif kernel_out is None:
            kernel_outs = [inputs[0]]
        else:
            kernel_outs = list(kernel_out)

        passed = True
        for i, (r, o) in enumerate(zip(ref_outs, kernel_outs)):
            r_f = r.float()
            o_f = o.float()
            if not torch.allclose(r_f, o_f, atol=spec.atol, rtol=spec.rtol):
                abs_err = (r_f - o_f).abs()
                max_err = abs_err.max().item()
                match_pct = (abs_err <= (spec.atol + spec.rtol * r_f.abs())).float().mean().item()
                print(f"  FAIL output[{i}]: max_err={max_err:.4f}, matched={match_pct:.1%} "
                      f"(atol={spec.atol}, rtol={spec.rtol})")
                passed = False
            else:
                if verbose:
                    print(f"  PASS output[{i}]: shape={o.shape} dtype={o.dtype}")

        if not passed:
            return False

        # -----------------------------------------------------------------------
        # Benchmark
        # -----------------------------------------------------------------------
        if verbose:
            print()
            print(f"Benchmarking (warmup={warmup}, iters={iters})...")

        ref_ms    = _time_fn(lambda *a: model(*a),     inputs, warmup, iters, device)
        kernel_ms = _time_fn(kernel_fn,                inputs, warmup, iters, device)
        speedup   = ref_ms / kernel_ms if kernel_ms > 0 else float("inf")

        print()
        print("=" * 70)
        print("Results")
        print("=" * 70)
        print(f"  Reference latency : {ref_ms:.4f} ms")
        print(f"  Kernel latency    : {kernel_ms:.4f} ms")
        print(f"  Speedup factor    : {speedup:.3f}x  {'✓ faster' if speedup >= 1.0 else '✗ slower'}")
        print("=" * 70)

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a KernelAgent flashinfer-bench solution.json"
    )
    parser.add_argument(
        "--solution", "-s",
        required=True,
        type=Path,
        help="Path to solution.json",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup iterations (default: 25)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    ok = verify(
        solution_path=args.solution,
        warmup=args.warmup,
        iters=args.iters,
        verbose=not args.quiet,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
