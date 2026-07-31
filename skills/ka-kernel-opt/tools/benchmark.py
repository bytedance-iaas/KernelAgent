#!/usr/bin/env python3
"""
Benchmark a GPU kernel or its PyTorch reference.

Standalone: needs torch (and triton for --timing do_bench) at runtime; no
repo dependency. Ported from
triton_kernel_agent/opt_worker_component/benchmarking/ — keep semantics in
sync (CUDA-event timing with L2 clears; do_bench for Triton kernels).

Run one measurement per invocation; invoking this script IS the subprocess
isolation (a crashing kernel cannot take the orchestrating session down).

Usage:
    # Time the kernel (kernel.py defines kernel_function)
    python benchmark.py --mode kernel --kernel kernel.py --problem problem.py

    # Time the PyTorch eager reference
    python benchmark.py --mode eager --problem problem.py

    # Time the torch.compile reference
    python benchmark.py --mode compile --problem problem.py

Output (JSON to stdout):
    {"mode": ..., "time_ms": <mean>, "stats": {mean_ms, std_ms, min_ms, max_ms,
     num_trials}, "dtype": ..., "timing": ...}
    or {"error": "..."} with exit code 1.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

L2_CLEAR_BYTES = 256 * 1024 * 1024


def _clear_l2_cache(device: str = "cuda"):
    import torch

    cache = torch.empty(L2_CLEAR_BYTES // 8, dtype=torch.int64, device=device)
    cache.zero_()
    del cache


def time_with_cuda_events(fn, args, warmup: int, repeat: int, clear_cache: bool = True):
    """Return per-trial times in ms using CUDA events, clearing L2 between trials."""
    import torch

    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        if clear_cache:
            _clear_l2_cache()
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def time_with_do_bench(fn, args, warmup: int, repeat: int):
    """Triton's adaptive benchmark. warmup/repeat are time budgets in ms."""
    from triton.testing import do_bench

    times = do_bench(lambda: fn(*args), warmup=warmup, rep=repeat, return_mode="all")
    return list(times) if isinstance(times, (list, tuple)) else [float(times)]


def _stats(times: list[float]) -> dict:
    return {
        "mean_ms": statistics.fmean(times),
        "std_ms": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
        "num_trials": len(times),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark a kernel or PyTorch reference")
    p.add_argument("--mode", required=True, choices=["kernel", "eager", "compile"])
    p.add_argument("--problem", required=True, help="Path to problem.py")
    p.add_argument("--kernel", default=None, help="Path to kernel.py (mode=kernel)")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"]
    )
    p.add_argument(
        "--timing",
        default="auto",
        choices=["auto", "cuda_event", "do_bench"],
        help="auto: do_bench for mode=kernel when triton is available, else cuda_event",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None, help="Write JSON here instead of stdout")
    p.add_argument(
        "--workload-index", type=int, default=None,
        help="Benchmark a specific WORKLOADS[i] instead of the canonical "
             "get_inputs() -- requires the problem to define "
             "build_workload_inputs(i) (unified problem.md contract). "
             "Used by lazy niche scan.")
    args = p.parse_args(argv)

    result: dict
    try:
        import torch

        from kernel_io import (
            build_model,
            detect_dtype,
            load_problem,
            prepare_inputs,
            prepare_kernel_callable,
        )

        if args.mode == "kernel" and not args.kernel:
            raise ValueError("--kernel is required for --mode kernel")

        Model, get_inputs, get_init_inputs = load_problem(
            args.problem, workload_index=args.workload_index)

        if args.dtype != "auto":
            dtype = getattr(torch, args.dtype)
        elif args.kernel:
            dtype = detect_dtype(Path(args.kernel).read_text(encoding="utf-8"))
        else:
            dtype = torch.bfloat16

        inputs = prepare_inputs(get_inputs(), device=args.device, dtype=dtype)
        model = build_model(Model, get_init_inputs, args.device, dtype)

        if args.mode == "kernel":
            fn, call_args = prepare_kernel_callable(args.kernel, inputs, model)
        elif args.mode == "eager":
            fn, call_args = model, inputs
        else:  # compile
            compiled = torch.compile(model)
            with torch.no_grad():
                for _ in range(3):
                    compiled(*inputs)
            torch.cuda.synchronize()
            fn, call_args = compiled, inputs

        timing = args.timing
        if timing == "auto":
            if args.mode == "kernel":
                try:
                    import triton  # noqa: F401

                    timing = "do_bench"
                except ImportError:
                    timing = "cuda_event"
            else:
                timing = "cuda_event"

        with torch.no_grad():
            if timing == "do_bench":
                times = time_with_do_bench(fn, call_args, args.warmup, args.repeat)
            else:
                times = time_with_cuda_events(fn, call_args, args.warmup, args.repeat)

        stats = _stats(times)
        result = {
            "mode": args.mode,
            "time_ms": stats["mean_ms"],
            "stats": stats,
            "dtype": str(dtype).replace("torch.", ""),
            "timing": timing,
            "workload_index": args.workload_index,
        }
        exit_code = 0
    except Exception as e:  # report every failure as structured JSON
        import traceback

        result = {
            "mode": args.mode,
            "time_ms": None,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-2000:],
        }
        exit_code = 1

    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
