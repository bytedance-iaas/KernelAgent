#!/usr/bin/env python3
"""
Create and manage run directories for kernel generation artifacts.

Handles the directory structure that mirrors the existing pipeline:
  .fuse/<run_id>/
    orchestrator/
    workers/
    subgraphs.json
    kernels_out/
    compose_out/

Usage:
    python manage_artifacts.py create-run-dir [--base .fuse]
    python manage_artifacts.py write-artifact --run-dir <path> --name code.py --content <text>
    python manage_artifacts.py list-runs [--base .fuse]

Logic ported from: Fuser/paths.py, Fuser/config.py (new_run_id, make_run_dirs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    """Generate a unique run ID."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    micro = int(time.time() * 1000000) % 1000000
    pid = os.getpid()
    return f"run_{ts}_{micro}_{pid}"


def make_run_dirs(base_dir: Path, run_id: str) -> dict[str, Path]:
    """Create the standard run directory structure."""
    run_dir = base_dir / run_id
    orchestrator_dir = run_dir / "orchestrator"
    workers_dir = run_dir / "workers"
    kernels_dir = run_dir / "kernels_out"
    compose_dir = run_dir / "compose_out"

    for d in [run_dir, orchestrator_dir, workers_dir, kernels_dir, compose_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "orchestrator": orchestrator_dir,
        "workers": workers_dir,
        "kernels_out": kernels_dir,
        "compose_out": compose_dir,
    }


def create_run_dir(base: str = ".fuse") -> dict[str, str]:
    """Create a new run directory and return paths."""
    base_dir = Path.cwd() / base
    base_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id()
    dirs = make_run_dirs(base_dir, run_id)
    return {k: str(v) for k, v in dirs.items()}


def write_artifact(run_dir: str, name: str, content: str) -> dict[str, str]:
    """Write an artifact file to the run directory."""
    rd = Path(run_dir)
    if not rd.is_dir():
        return {"error": f"Run directory not found: {run_dir}"}
    path = rd / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "size": len(content)}


def read_artifact(run_dir: str, name: str) -> dict[str, Any]:
    """Read an artifact file from the run directory."""
    rd = Path(run_dir)
    path = rd / name
    if not path.is_file():
        return {"error": f"Artifact not found: {path}"}
    content = path.read_text(encoding="utf-8")
    return {"path": str(path), "content": content, "size": len(content)}


def list_runs(base: str = ".fuse") -> list[dict[str, Any]]:
    """List all run directories."""
    base_dir = Path.cwd() / base
    if not base_dir.is_dir():
        return []
    runs = []
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith("run_"):
            has_subgraphs = (d / "subgraphs.json").is_file()
            has_composed = (d / "compose_out" / "composed_kernel.py").is_file()
            runs.append({
                "run_id": d.name,
                "path": str(d),
                "has_subgraphs": has_subgraphs,
                "has_composed": has_composed,
            })
    return runs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Manage run directories for kernel generation artifacts"
    )
    sub = p.add_subparsers(dest="command", help="Command to run")

    # create-run-dir
    cr = sub.add_parser("create-run-dir", help="Create a new run directory")
    cr.add_argument("--base", default=".fuse", help="Base directory (default: .fuse)")

    # write-artifact
    wa = sub.add_parser("write-artifact", help="Write an artifact file")
    wa.add_argument("--run-dir", required=True, help="Path to run directory")
    wa.add_argument("--name", required=True, help="Artifact filename")
    wa.add_argument(
        "--content",
        default=None,
        help="Content to write (reads from stdin if not provided)",
    )
    wa.add_argument(
        "--content-file",
        default=None,
        help="Path to file with content to write",
    )

    # read-artifact
    ra = sub.add_parser("read-artifact", help="Read an artifact file")
    ra.add_argument("--run-dir", required=True, help="Path to run directory")
    ra.add_argument("--name", required=True, help="Artifact filename")

    # list-runs
    lr = sub.add_parser("list-runs", help="List all run directories")
    lr.add_argument("--base", default=".fuse", help="Base directory (default: .fuse)")

    args = p.parse_args(argv)

    if args.command == "create-run-dir":
        result = create_run_dir(base=args.base)
        print(json.dumps(result, indent=2))
        return 0

    elif args.command == "write-artifact":
        if args.content_file:
            content = Path(args.content_file).read_text(encoding="utf-8")
        elif args.content:
            content = args.content
        else:
            content = sys.stdin.read()
        result = write_artifact(args.run_dir, args.name, content)
        print(json.dumps(result, indent=2))
        return 0 if "error" not in result else 1

    elif args.command == "read-artifact":
        result = read_artifact(args.run_dir, args.name)
        print(json.dumps(result, indent=2))
        return 0 if "error" not in result else 1

    elif args.command == "list-runs":
        runs = list_runs(base=args.base)
        print(json.dumps(runs, indent=2))
        return 0

    else:
        p.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
