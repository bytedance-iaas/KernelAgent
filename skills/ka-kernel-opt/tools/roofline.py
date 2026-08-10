#!/usr/bin/env python3
"""
Roofline / Speed-of-Light analysis from NCU metrics.

Standalone: stdlib only. Vendored from
kernel_perf_agent/kernel_opt/roofline/ncu_roofline.py and the grid-analysis /
config-extraction helpers in
triton_kernel_agent/opt_worker_component/orchestrator/optimization_orchestrator.py
— keep in sync.

Classifies the kernel bottleneck from the two NCU SOL metrics:
  - sm__throughput.avg.pct_of_peak_sustained_elapsed                → compute SOL
  - gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed → memory SOL

Usage:
    python roofline.py --metrics ncu_metrics.json \
        [--kernel-name SUBSTRING] \
        [--gpu-name "NVIDIA H100 NVL 94GB"] \
        [--kernel kernel.py] [--kernel-language triton]

Input:
    ncu_metrics.json — either the output of profile_ncu.py
    ({"kernels": {name: {metric: value}}, ...}) or a flat {metric: value} dict.

Output (JSON to stdout):
    {
      "roofline": {compute_sol_pct, memory_sol_pct, efficiency_pct,
                   at_roofline, headroom_pct, bottleneck, uses_tensor_cores,
                   warnings},
      "grid_analysis": {...} | null,     # requires --gpu-name
      "kernel_config": {...} | null      # requires --kernel
    }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

COMPUTE_SOL_KEY = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
MEMORY_SOL_KEY = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
TENSOR_CORE_KEY = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"

DEFAULT_THRESHOLD_PCT = 95.0
DEFAULT_UNDERUTILIZED_PCT = 60.0
DEFAULT_TC_THRESHOLD_PCT = 5.0


def select_target_metrics(data: dict, kernel_name: str | None = None) -> tuple[str, dict]:
    """Pick the target kernel's flat metrics dict from a metrics JSON.

    Accepts profile_ncu.py output ({"kernels": {...}}), a name-keyed dict,
    or an already-flat {metric: value} dict. PyTorch-internal kernels
    (``at::*``) are filtered out unless they are the only ones present.
    Among the remaining candidates, the most expensive one by
    sm__cycles_active.avg is picked (falling back to launch order if no
    candidate has a usable cycle count) — the bottleneck kernel isn't
    necessarily the first one launched.
    """
    if "kernels" in data and isinstance(data["kernels"], dict):
        kernels = data["kernels"]
    elif data and all(isinstance(v, dict) for v in data.values()):
        kernels = data
    else:
        return "<flat>", data  # already flat

    if kernel_name:
        for name, metrics in kernels.items():
            if kernel_name in name:
                return name, metrics
        raise SystemExit(f"error: no kernel matching '{kernel_name}' in metrics")

    non_torch = {
        n: m
        for n, m in kernels.items()
        if not n.startswith("at::") and not n.startswith("void at::")
    }
    pool = non_torch or kernels

    scored = [
        (n, m.get("sm__cycles_active.avg"))
        for n, m in pool.items()
        if isinstance(m.get("sm__cycles_active.avg"), (int, float))
    ]
    name = max(scored, key=lambda item: item[1])[0] if scored else next(iter(pool))
    return name, pool[name]


def _get_float(metrics: dict, key: str) -> float | None:
    v = metrics.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def analyze_roofline(
    metrics: dict,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    underutilized_pct: float = DEFAULT_UNDERUTILIZED_PCT,
    tc_threshold_pct: float = DEFAULT_TC_THRESHOLD_PCT,
) -> dict:
    warnings = []
    compute_sol = _get_float(metrics, COMPUTE_SOL_KEY)
    memory_sol = _get_float(metrics, MEMORY_SOL_KEY)

    if compute_sol is None:
        warnings.append(f"missing metric: {COMPUTE_SOL_KEY}")
    if memory_sol is None:
        warnings.append(f"missing metric: {MEMORY_SOL_KEY}")

    if compute_sol is None and memory_sol is None:
        return {
            "compute_sol_pct": 0.0,
            "memory_sol_pct": 0.0,
            "efficiency_pct": 0.0,
            "at_roofline": False,
            "headroom_pct": 100.0,
            "bottleneck": "unknown",
            "uses_tensor_cores": False,
            "warnings": warnings,
        }

    compute_sol = compute_sol or 0.0
    memory_sol = memory_sol or 0.0
    efficiency = max(compute_sol, memory_sol)

    if compute_sol < underutilized_pct and memory_sol < underutilized_pct:
        bottleneck = "underutilized"
    elif memory_sol >= compute_sol:
        bottleneck = "memory"
    else:
        bottleneck = "compute"

    tc = _get_float(metrics, TENSOR_CORE_KEY) or 0.0

    return {
        "compute_sol_pct": compute_sol,
        "memory_sol_pct": memory_sol,
        "efficiency_pct": efficiency,
        "at_roofline": efficiency >= threshold_pct,
        "headroom_pct": max(0.0, 100.0 - efficiency),
        "bottleneck": bottleneck,
        "uses_tensor_cores": tc > tc_threshold_pct,
        "warnings": warnings,
    }


def compute_grid_analysis(metrics: dict, gpu_specs: dict | None) -> dict | None:
    """Grid utilization heuristics for diagnosing occupancy issues."""
    grid_x = metrics.get("launch__grid_dim_x")
    if grid_x is None:
        return None
    try:
        grid_x = int(float(grid_x))
        grid_y = int(float(metrics.get("launch__grid_dim_y") or 1))
        grid_z = int(float(metrics.get("launch__grid_dim_z") or 1))
    except (ValueError, TypeError):
        return None

    warp_active_pct = _get_float(metrics, "sm__warps_active.avg.pct_of_peak_sustained_active") or 0.0
    total_blocks = grid_x * grid_y * grid_z
    sm_count = (gpu_specs or {}).get("sm_count", 132)

    bps = _get_float(metrics, "launch__blocks_per_multiprocessor")
    if bps is None:
        bps = max(1.0, total_blocks / sm_count)

    concurrent_blocks = sm_count * bps
    estimated_waves = (
        total_blocks / concurrent_blocks if concurrent_blocks > 0 else float("inf")
    )

    if total_blocks < sm_count:
        assessment = (
            f"CRITICAL: only {total_blocks} blocks for {sm_count} SMs — "
            f"{100 * total_blocks / sm_count:.0f}% SM utilization. "
            "Most SMs are idle; HBM latency cannot be hidden."
        )
        recommendation = (
            f"Reduce tile size (e.g. BLOCK_M) to increase total blocks to at least "
            f"{sm_count * 4}–{sm_count * 8}. "
            "Example: BLOCK_M=1 gives grid=M blocks, fully utilizing all SMs."
        )
    elif estimated_waves > 20:
        assessment = (
            f"WARNING: {total_blocks} blocks → ~{estimated_waves:.0f} sequential waves. "
            f"Each wave incurs ~15 µs scheduling overhead "
            f"(≈ {estimated_waves * 15 / 1000:.1f} ms total wasted)."
        )
        recommendation = (
            f"Reduce total blocks to {sm_count * 4}–{sm_count * 8} by increasing tile "
            "size or reducing SPLIT_K. Wave overhead is invisible in per-kernel NCU "
            "metrics but dominates wall-clock time."
        )
    elif warp_active_pct < 30.0:
        assessment = (
            f"LOW OCCUPANCY: warp active = {warp_active_pct:.1f}%. "
            "Too few warps in flight to hide memory latency."
        )
        recommendation = (
            "Increase grid size or num_warps to raise occupancy above 50%. "
            "Check launch__occupancy_limit_* to find the bottleneck resource."
        )
    else:
        assessment = (
            f"Grid OK: {total_blocks} blocks, ~{estimated_waves:.1f} waves, "
            f"{warp_active_pct:.1f}% warp active."
        )
        recommendation = ""

    return {
        "total_blocks": total_blocks,
        "sm_count": sm_count,
        "blocks_per_sm": bps,
        "estimated_waves": estimated_waves,
        "warp_active_pct": float(warp_active_pct),
        "assessment": assessment,
        "recommendation": recommendation,
    }


def extract_kernel_config(kernel_code: str, kernel_language: str = "triton") -> dict:
    """Extract tunable configuration parameters from kernel code.

    Triton: parses triton.Config({...}, num_warps=..., num_stages=...).
    CuteDSL/TileLang: module-level UPPER_CASE integer constants.
    """
    if kernel_language != "triton":
        matches = re.findall(
            r"^([A-Z_][A-Z0-9_]*)\s*=\s*(\d+)", kernel_code, re.MULTILINE
        )
        return {name: int(val) for name, val in matches}

    config: dict = {}
    config_pattern = r"triton\.Config\s*\(\s*\{([^}]*)\}(?:\s*,\s*([^)]*))?\)"
    matches = re.findall(config_pattern, kernel_code)
    if not matches:
        # Fall back to module-level constants (kernels without autotune)
        const_matches = re.findall(
            r"^([A-Z_][A-Z0-9_]*)\s*=\s*(\d+)", kernel_code, re.MULTILINE
        )
        return {name: int(val) for name, val in const_matches}

    block_params, extra_params = matches[0]
    for name, value in re.findall(r"['\"](\w+)['\"]\s*:\s*(\d+)", block_params):
        config[name] = int(value)
    if extra_params:
        for name, value in re.findall(r"(\w+)\s*=\s*(\d+)", extra_params):
            config[name] = int(value)
    return config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Roofline analysis from NCU metrics")
    p.add_argument("--metrics", required=True, help="Path to NCU metrics JSON")
    p.add_argument("--kernel-name", default=None, help="Substring to select the kernel")
    p.add_argument("--gpu-name", default=None, help="GPU name for grid analysis")
    p.add_argument("--gpu-specs", default=None, help="Path to GPU specs JSON (overrides --gpu-name)")
    p.add_argument("--kernel", default=None, help="Kernel file for config extraction")
    p.add_argument("--kernel-language", default="triton", choices=["triton", "tilelang", "cutedsl"])
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PCT, help="At-roofline threshold %%")
    p.add_argument("--underutilized-threshold", type=float, default=DEFAULT_UNDERUTILIZED_PCT)
    p.add_argument("--tc-threshold", type=float, default=DEFAULT_TC_THRESHOLD_PCT)
    args = p.parse_args(argv)

    data = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    kernel_name, metrics = select_target_metrics(data, args.kernel_name)

    gpu_specs = None
    if args.gpu_specs:
        gpu_specs = json.loads(Path(args.gpu_specs).read_text(encoding="utf-8"))
    elif args.gpu_name:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from gpu_specs import lookup

        gpu_specs = lookup(args.gpu_name)

    result = {
        "kernel_name": kernel_name,
        "roofline": analyze_roofline(
            metrics,
            threshold_pct=args.threshold,
            underutilized_pct=args.underutilized_threshold,
            tc_threshold_pct=args.tc_threshold,
        ),
        "grid_analysis": compute_grid_analysis(metrics, gpu_specs),
        "kernel_config": (
            extract_kernel_config(
                Path(args.kernel).read_text(encoding="utf-8"), args.kernel_language
            )
            if args.kernel
            else None
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
