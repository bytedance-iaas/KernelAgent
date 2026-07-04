#!/usr/bin/env python3
"""
Deduplicate subgraphs by shape signature and merge counts.

Takes a raw subgraphs JSON array and deduplicates entries that have
identical shape signatures (ops + input/output/weight shapes).

Usage:
    python dedup_subgraphs.py --input raw_subgraphs.json --output deduped.json

Logic ported from: Fuser/subgraph_extractor.py (sig_of, grouping logic)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _sig_of(it: dict[str, Any]) -> str:
    """Build a robust signature from ops + shapes + weights."""
    ops = it.get("ops") or []
    ops_norm = []
    for op in ops:
        if isinstance(op, dict):
            ops_norm.append(json.loads(json.dumps(op, sort_keys=True)))
        else:
            ops_norm.append(op)

    inputs_single = it.get("input_shape")
    inputs_multi = it.get("inputs")
    outputs = it.get("output_shape")
    weights = it.get("weights") or {}
    weights_fused = it.get("weights_fused") or {}
    weights_original = it.get("weights_original") or {}

    def sort_w(obj: Any) -> dict[str, Any]:
        if isinstance(obj, dict):
            return {k: obj[k] for k in sorted(obj.keys())}
        return {}

    sig_obj = {
        "ops": ops_norm,
        "in": inputs_multi if inputs_multi is not None else inputs_single,
        "out": outputs,
        "w": sort_w(weights),
        "wf": sort_w(weights_fused),
        "wo": sort_w(weights_original),
        "layout": it.get("data_layout"),
        "dtype": it.get("dtype"),
    }
    return json.dumps(sig_obj, sort_keys=True)


def dedup_subgraphs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate subgraphs by shape signature, summing counts."""
    grouped: dict[str, dict[str, Any]] = {}

    for it in items:
        s = _sig_of(it)
        if s not in grouped:
            # Ensure id exists
            if not it.get("id"):
                it["id"] = f"sg_{hash(s) & 0xFFFFFFFF:08x}"
            # Normalize count
            c = it.get("count")
            try:
                count_val = int(c) if c is not None else 1
            except Exception:
                count_val = 1
            it["count"] = count_val
            grouped[s] = it
        else:
            try:
                grouped[s]["count"] += int(it.get("count") or 1)
            except Exception:
                grouped[s]["count"] += 1

    return list(grouped.values())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Deduplicate subgraphs JSON by shape signature"
    )
    p.add_argument("--input", required=True, help="Path to raw subgraphs JSON")
    p.add_argument(
        "--output",
        default=None,
        help="Output path for deduplicated JSON (prints to stdout if not set)",
    )
    args = p.parse_args(argv)

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(json.dumps({"error": f"File not found: {input_path}"}))
        return 2

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print(json.dumps({"error": "Input must be a JSON array"}))
        return 1

    deduped = dedup_subgraphs(data)
    result = json.dumps(deduped, indent=2)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(json.dumps({
            "input_count": len(data),
            "output_count": len(deduped),
            "output_path": args.output,
        }))
    else:
        print(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
