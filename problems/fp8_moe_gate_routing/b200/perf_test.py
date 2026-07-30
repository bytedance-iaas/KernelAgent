"""Auto-generated from problem.md by scripts/problem_md.py - do not edit.

Performance goal gate for GPU spec 'b200' (this file lives in the
<problem>/b200/ subfolder and is pinned to that spec). Benchmarks
`kernel.kernel_function` from the PARENT problem directory on every
workload carrying a 'b200' latency spec (ms). Per workload the
pass criterion is: measured <= `target` when the spec pins a hard
target; otherwise SOL-Score >= `min_score` (default 0.5) when it pins
`sol`, where S = 1/(1+(t-sol)/(baseline-sol)) is 1.0 at speed-of-light
and 0.5 at the baseline; otherwise measured <= `baseline`. Exit 0 iff
every workload passes; exit 2 when run on a different GPU. This is the
goal gate for the ka-kernel-opt pipeline; it is NOT a correctness
test - run test.py for accuracy.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import problem
from fp8_moe_gate_routing import kernel_function
# from submission import kernel_function
# from submission_final import kernel_function

GPU_KEY = 'b200'


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


def sol_score(t_k, t_b, t_sol):
    """SOL-ExecBench anchored score: 1.0 at speed-of-light, 0.5 at baseline."""
    denom = t_b - t_sol
    if denom <= 0:
        return 1.0 if t_k <= t_sol else 0.0
    return 1.0 / (1.0 + (t_k - t_sol) / denom)


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
        args = problem.build_workload_inputs(i)
        with torch.no_grad():
            ms = bench_ms(lambda: kernel_function(*args))
        baseline_ms = lat["baseline"]
        rec = dict(workload=i, axes=wl.get("axes", {}), gpu=GPU_KEY,
                   measured_ms=ms, baseline_ms=baseline_ms)
        detail = f"baseline {baseline_ms}, {baseline_ms / ms:.2f}x vs baseline"
        if lat["target"] is not None:
            met = ms <= lat["target"]
            rec.update(target_ms=lat["target"])
            detail += f", target {lat['target']}"
        elif lat["sol"] is not None:
            score = sol_score(ms, baseline_ms, lat["sol"])
            met = score >= lat["min_score"]
            rec.update(sol_ms=lat["sol"], sol_score=score,
                       min_score=lat["min_score"])
            detail += (f", sol {lat['sol']}, {100.0 * lat['sol'] / ms:.1f}% of SOL"
                       f", score {score:.3f} (need >= {lat['min_score']})")
        else:
            met = ms <= baseline_ms
        rec["target_met"] = met
        all_met &= met
        results.append(rec)
        print(f"{'ok  ' if met else 'MISS'} workload[{i}] {wl.get('axes', {})}: "
              f"{ms:.4f} ms ({detail})")
        del args
    print(json.dumps(dict(gpu=GPU_KEY, all_targets_met=all_met, results=results)))
    print("PERF PASS" if all_met else "PERF MISS")
    return 0 if all_met else 1


if __name__ == "__main__":
    sys.exit(main())
