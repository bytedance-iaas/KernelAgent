#!/usr/bin/env python3
"""
Auto-select representative workloads from a problem's WORKLOADS list.

Standalone: stdlib only (imports the problem file only for its WORKLOADS
list, no torch/GPU needed). Used by lazy niche scan (steps/07_lazy_niche_scan.md)
to pick which workload indices to scan against instead of hand-picking them
by guessing where the interesting regime split is; this runs a rule against
the problem's own WORKLOADS list (the unified problem.md / SOL-ExecBench-style
contract -- see docs/PROBLEM_MD_FORMAT.md).

Rule: rank workloads by total problem "size" (product of their numeric axis
values, e.g. batch_size * seq_len), then pick N indices evenly spread across
that ranking by rank position -- always including the smallest and largest.
This is a size-based proxy for "different regime," not a guarantee (two
workloads with similar size can still stress a kernel differently) -- it is
a default, not a claim that size is the only axis that matters. Pick a
larger --n or run again with a different axis if a problem is known to vary
along a non-size dimension.

Usage:
    python select_workloads.py --problem problem.py --n 3
    python select_workloads.py --problem problem.py --n 3 --axis seq_len
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_workloads(problem_path: str | Path):
    spec = importlib.util.spec_from_file_location("problem", str(problem_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {problem_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "WORKLOADS"):
        raise SystemExit(
            "error: problem.py has no WORKLOADS list -- this needs the "
            "unified problem.md contract (WORKLOADS + build_workload_inputs); "
            "see docs/PROBLEM_MD_FORMAT.md. This problem only has a single "
            "canonical get_inputs() workload, so there's nothing to select."
        )
    return mod.WORKLOADS


def _size(workload: dict, axis: str | None) -> float:
    axes = workload.get("axes", {})
    if axis is not None:
        v = axes.get(axis)
        if not isinstance(v, (int, float)):
            raise SystemExit(f"error: axis {axis!r} missing or non-numeric in {axes}")
        return float(v)
    numeric = [v for v in axes.values() if isinstance(v, (int, float))]
    if not numeric:
        raise SystemExit(f"error: no numeric axes to rank by in {axes}")
    size = 1.0
    for v in numeric:
        size *= v
    return size


def select(workloads: list[dict], n: int, axis: str | None) -> dict:
    sized = sorted(
        ((_size(wl, axis), i, wl) for i, wl in enumerate(workloads)),
        key=lambda t: t[0],
    )
    n = max(1, min(n, len(sized)))
    if n == 1:
        rank_positions = [len(sized) // 2]
    else:
        # Evenly spaced by rank, always including both ends.
        rank_positions = sorted({
            round(k * (len(sized) - 1) / (n - 1)) for k in range(n)
        })
    picked = [sized[r] for r in rank_positions]
    return {
        "axis": axis or "size (product of numeric axes)",
        "n_requested": n,
        "n_selected": len(picked),
        "workload_indices": [i for _, i, _ in picked],
        "sizes": [s for s, _, _ in picked],
        "axes": [wl.get("axes", {}) for _, _, wl in picked],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Auto-select representative workloads for lazy niche scan")
    p.add_argument("--problem", required=True, help="Path to problem.py")
    p.add_argument("--n", type=int, default=3, help="Number of representative workloads (default 3)")
    p.add_argument("--axis", default=None,
                    help="Rank by this single axis name instead of the product "
                         "of all numeric axes (e.g. --axis seq_len)")
    args = p.parse_args(argv)

    workloads = _load_workloads(args.problem)
    result = select(workloads, args.n, args.axis)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
