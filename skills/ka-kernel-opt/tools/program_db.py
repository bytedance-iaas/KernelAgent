#!/usr/bin/env python3
"""
Program database for kernel optimization runs.

Standalone: stdlib only. Ported from
triton_kernel_agent/opt_worker_component/searching/history/{json_db,models}.py
(JSONProgramDatabase / ProgramEntry) — keep the entry schema in sync.

Persists every verified kernel candidate with its runtime and lineage
(parent_id, generation) to a single JSON file, and answers the queries the
search strategies need: current best (greedy parent), top-K (beam parents and
the final report).

Usage:
    # Round 0: register the initial kernel (time may be updated later)
    python program_db.py init --db $RUN_DIR/program_db.json \
        --kernel-file kernel_round_0.py --problem-id problem.py --time-ms 1.234

    # After each verified+benchmarked candidate:
    python program_db.py add --db $RUN_DIR/program_db.json \
        --kernel-file kernel_round_3.py --time-ms 0.98 --round 3 \
        --parent-id initial --sol-pct 72.5 \
        --category memory --fix "PID swizzling for L2 locality"

    python program_db.py best --db $RUN_DIR/program_db.json
    python program_db.py top  --db $RUN_DIR/program_db.json --k 5

Entry schema (mirrors ProgramEntry + ProgramMetrics):
    {"program_id", "kernel_file", "metrics": {"time_ms", "sol_pct"},
     "problem_id", "parent_id", "generation", "round", "category", "fix",
     "created_at"}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _load(db_path: Path) -> dict:
    if db_path.exists():
        return json.loads(db_path.read_text(encoding="utf-8"))
    return {"programs": []}


def _save(db_path: Path, db: dict) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text(json.dumps(db, indent=2), encoding="utf-8")


def _sorted_programs(db: dict) -> list[dict]:
    return sorted(
        db["programs"],
        key=lambda p: (
            p["metrics"]["time_ms"]
            if p["metrics"]["time_ms"] is not None
            else float("inf")
        ),
    )


def cmd_init(args) -> dict:
    db_path = Path(args.db)
    db = _load(db_path)
    entry = {
        "program_id": "initial",
        "kernel_file": str(Path(args.kernel_file).resolve()),
        "metrics": {"time_ms": args.time_ms, "sol_pct": args.sol_pct},
        "problem_id": args.problem_id,
        "parent_id": None,
        "generation": 0,
        "round": 0,
        "category": "",
        "fix": "",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    db["programs"] = [p for p in db["programs"] if p["program_id"] != "initial"]
    db["programs"].insert(0, entry)
    _save(db_path, db)
    return entry


def cmd_add(args) -> dict:
    db_path = Path(args.db)
    db = _load(db_path)

    parent = next(
        (p for p in db["programs"] if p["program_id"] == args.parent_id), None
    )
    generation = (parent["generation"] + 1) if parent else 1

    program_id = f"r{args.round}_{len(db['programs'])}"
    entry = {
        "program_id": program_id,
        "kernel_file": str(Path(args.kernel_file).resolve()),
        "metrics": {"time_ms": args.time_ms, "sol_pct": args.sol_pct},
        "problem_id": args.problem_id or (parent or {}).get("problem_id", ""),
        "parent_id": args.parent_id,
        "generation": generation,
        "round": args.round,
        "category": args.category,
        "fix": args.fix,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    db["programs"].append(entry)
    _save(db_path, db)
    return entry


def cmd_best(args) -> dict:
    db = _load(Path(args.db))
    programs = _sorted_programs(db)
    if not programs:
        raise SystemExit("error: database is empty")
    return programs[0]


def cmd_top(args) -> list[dict]:
    db = _load(Path(args.db))
    return _sorted_programs(db)[: args.k]


def cmd_list(args) -> list[dict]:
    db = _load(Path(args.db))
    return db["programs"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Kernel optimization program database")
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Register the initial kernel")
    p_init.add_argument("--db", required=True)
    p_init.add_argument("--kernel-file", required=True)
    p_init.add_argument("--problem-id", default="")
    p_init.add_argument("--time-ms", type=float, default=None)
    p_init.add_argument("--sol-pct", type=float, default=None)
    p_init.set_defaults(fn=cmd_init)

    p_add = sub.add_parser("add", help="Add a verified candidate")
    p_add.add_argument("--db", required=True)
    p_add.add_argument("--kernel-file", required=True)
    p_add.add_argument("--time-ms", type=float, required=True)
    p_add.add_argument("--round", type=int, required=True)
    p_add.add_argument("--parent-id", required=True)
    p_add.add_argument("--sol-pct", type=float, default=None)
    p_add.add_argument("--problem-id", default=None)
    p_add.add_argument("--category", default="")
    p_add.add_argument("--fix", default="")
    p_add.set_defaults(fn=cmd_add)

    p_best = sub.add_parser("best", help="Best program by time_ms")
    p_best.add_argument("--db", required=True)
    p_best.set_defaults(fn=cmd_best)

    p_top = sub.add_parser("top", help="Top-K programs by time_ms")
    p_top.add_argument("--db", required=True)
    p_top.add_argument("--k", type=int, default=5)
    p_top.set_defaults(fn=cmd_top)

    p_list = sub.add_parser("list", help="All programs in insertion order")
    p_list.add_argument("--db", required=True)
    p_list.set_defaults(fn=cmd_list)

    args = p.parse_args(argv)
    print(json.dumps(args.fn(args), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
