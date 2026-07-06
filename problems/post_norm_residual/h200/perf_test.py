"""Auto-generated from problem.md by scripts/problem_md.py - do not edit.

Performance goal gate for GPU spec 'h200' (this file lives in the
<problem>/h200/ subfolder and is pinned to that spec). Benchmarks
`kernel.kernel_function` from the PARENT problem directory on every
workload carrying a 'h200' latency target (ms). Exit 0 iff every
target is met; exit 2 when run on a different GPU. This is the goal
gate for the ka-kernel-opt pipeline; it is NOT a correctness test -
run test.py for accuracy.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import problem
from kernel import kernel_function

GPU_KEY = 'h200'


def bench_ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]  # median


def main() -> int:
    name = torch.cuda.get_device_name(0)
    if GPU_KEY.lower() not in name.lower():
        print(f"this perf test is pinned to {GPU_KEY!r} but the current "
              f"GPU is {name!r} - refusing to judge the goal")
        return 2
    results, all_met = [], True
    for i, wl in enumerate(problem.WORKLOADS):
        lat = problem.workload_latency(i, GPU_KEY)
        if lat is None:
            continue
        baseline_ms, target_ms = lat
        args = problem.build_workload_inputs(i)
        with torch.no_grad():
            ms = bench_ms(lambda: kernel_function(*args))
        met = ms <= target_ms
        all_met &= met
        results.append(dict(workload=i, axes=wl.get("axes", {}), gpu=GPU_KEY,
                            measured_ms=ms, baseline_ms=baseline_ms,
                            target_ms=target_ms, target_met=met))
        print(f"{'ok  ' if met else 'MISS'} workload[{i}] {wl.get('axes', {})}: "
              f"{ms:.4f} ms (baseline {baseline_ms}, target {target_ms}, "
              f"{baseline_ms / ms:.2f}x vs baseline)")
        del args
    print(json.dumps(dict(gpu=GPU_KEY, all_targets_met=all_met, results=results)))
    print("PERF PASS" if all_met else "PERF MISS")
    return 0 if all_met else 1


if __name__ == "__main__":
    sys.exit(main())
