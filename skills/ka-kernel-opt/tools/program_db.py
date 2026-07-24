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
     "lineage_id", "created_at"}

`lineage_id` tracks which *beam slot* an entry descends from (not which
kernel state it descends from) -- ported from the diversity-floor patch in
triton_kernel_agent/opt_worker_component/searching/strategy/beam_search.py
(local sandbox experiment, uncommitted). For beam mode's round-1 candidates,
pass distinct `--lineage-id slot0` / `slot1` per parent slot explicitly
(both slots' parent_id is the same "initial" entry, so lineage can't be
inferred from parent_id alone at that point); from round 2 onward, a child
with no explicit --lineage-id inherits its parent's lineage_id automatically.

`top-diverse` (lineage-floor-from-leader selection) was tried and retired --
see insights/DIVERSITY-AWARE SEARCH (DIRECTION 1) POSTMORTEM.md. It survives
here as a documented dead end; do not wire it into the default beam workflow.

`metrics_by_workload` (optional, on `add`) records a candidate's time_ms on
each of a small set of *representative* workloads, not just the single
round-proxy metric -- for problems with a wide size/shape axis where one
proxy workload can hide a regime-dependent trade-off (see the postmortem's
GEMM crossover test: a candidate tuned for large-M can lose 6-7x at small-M
on the exact same problem). `top-niche` selects the best candidate *per
representative workload* instead of by a single scalar ranking, so a
regime specialist survives by being someone's best rather than by being
close to the single leader -- this is "direction 3" in the postmortem.
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
        "lineage_id": "initial",
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
    lineage_id = args.lineage_id or (parent or {}).get("lineage_id") or args.parent_id
    metrics_by_workload = (
        json.loads(args.metrics_by_workload) if args.metrics_by_workload else {}
    )

    program_id = f"r{args.round}_{len(db['programs'])}"
    entry = {
        "program_id": program_id,
        "kernel_file": str(Path(args.kernel_file).resolve()),
        "metrics": {"time_ms": args.time_ms, "sol_pct": args.sol_pct},
        "metrics_by_workload": metrics_by_workload,
        "problem_id": args.problem_id or (parent or {}).get("problem_id", ""),
        "parent_id": args.parent_id,
        "generation": generation,
        "round": args.round,
        "category": args.category,
        "fix": args.fix,
        "lineage_id": lineage_id,
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


def cmd_top_diverse(args) -> list[dict]:
    """Lineage-floor selection: fill slots with the best not-yet-represented
    lineage within `floor_factor` of the leader, before falling back to pure
    fitness for any remaining slots. Ported from the diversity-floor patch
    in the legacy BeamSearchStrategy (see module docstring)."""
    db = _load(Path(args.db))
    programs = _sorted_programs(db)
    if not programs:
        raise SystemExit("error: database is empty")

    best_time = programs[0]["metrics"]["time_ms"]
    selected: list[dict] = []
    used_lineages: set[str] = set()

    # Pass 1: one per lineage, within the floor.
    for p in programs:
        if len(selected) >= args.k:
            break
        lid = p.get("lineage_id") or p["parent_id"]
        if lid in used_lineages:
            continue
        t = p["metrics"]["time_ms"]
        if t is not None and best_time is not None and t <= best_time * args.floor_factor:
            selected.append(p)
            used_lineages.add(lid)

    # Pass 2: fill remaining slots by pure fitness (lineage repeats allowed).
    selected_ids = {p["program_id"] for p in selected}
    for p in programs:
        if len(selected) >= args.k:
            break
        if p["program_id"] in selected_ids:
            continue
        selected.append(p)
        selected_ids.add(p["program_id"])

    return selected


def cmd_top_niche(args) -> list[dict]:
    """Per-workload niching: for each representative workload, return the
    candidate with the best (lowest) time_ms on that workload. A candidate
    survives by being someone's best, not by being close to the single
    overall leader -- see module docstring / direction 3 in the postmortem.
    """
    db = _load(Path(args.db))
    programs = [p for p in db["programs"] if p.get("metrics_by_workload")]
    if not programs:
        raise SystemExit(
            "error: no entries have metrics_by_workload -- register candidates "
            "with `add --metrics-by-workload '{...}'` first"
        )

    workloads = sorted({wl for p in programs for wl in p["metrics_by_workload"]})
    champions: dict[str, dict] = {}
    for wl in workloads:
        candidates = [p for p in programs if wl in p["metrics_by_workload"]]
        best = min(candidates, key=lambda p: p["metrics_by_workload"][wl])
        champions[wl] = best

    seen: set[str] = set()
    selected: list[dict] = []
    for wl in workloads:
        p = champions[wl]
        if p["program_id"] in seen:
            continue
        seen.add(p["program_id"])
        selected.append(dict(p, niche_workload=wl))
    return selected


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
    p_add.add_argument("--lineage-id", default=None,
                        help="Beam slot this entry descends from; inherits "
                             "from the parent's lineage_id if omitted.")
    p_add.add_argument("--metrics-by-workload", default=None,
                        help='JSON object of {workload_key: time_ms}, e.g. '
                             '\'{"M1": 0.0356, "M8192": 2.0125}\' -- optional, '
                             'used by top-niche.')
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

    p_diverse = sub.add_parser(
        "top-diverse",
        help="Top-K with a lineage-diversity floor (see module docstring)")
    p_diverse.add_argument("--db", required=True)
    p_diverse.add_argument("--k", type=int, default=2)
    p_diverse.add_argument("--floor-factor", type=float, default=2.0)
    p_diverse.set_defaults(fn=cmd_top_diverse)

    p_niche = sub.add_parser(
        "top-niche",
        help="Best candidate per representative workload (direction 3)")
    p_niche.add_argument("--db", required=True)
    p_niche.set_defaults(fn=cmd_top_niche)

    args = p.parse_args(argv)
    print(json.dumps(args.fn(args), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
