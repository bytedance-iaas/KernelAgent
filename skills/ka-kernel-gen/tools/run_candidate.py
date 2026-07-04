#!/usr/bin/env python3
"""
Execute a candidate Python file and classify PASS/FAIL.

Runs the file in a subprocess with timeout, captures stdout/stderr,
and classifies the result based on exit code + sentinel detection.

Usage:
    python run_candidate.py --code-path /path/to/candidate.py --timeout 60

Output:
    JSON to stdout with execution results.

Logic ported from: Fuser/runner.py (run_candidate)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

STDOUT_MAX_TAIL = 20000  # bytes
STDERR_MAX_TAIL = 20000  # bytes
MAX_SCAN_BYTES = 512 * 1024

_SENTINEL = "ALL_TESTS_PASSED"
_PASS_REGEX = re.compile(r"\bPASS\b")


@dataclass(frozen=True)
class RunResult:
    rc: int
    passed: bool
    validator: str
    reason: str
    stdout_tail: str
    stderr_tail: str
    stdout_path: str
    stderr_path: str
    elapsed_s: float


def _tail_bytes(p: Path, max_bytes: int) -> bytes:
    try:
        with p.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            take = min(size, max_bytes)
            f.seek(size - take)
            return f.read()
    except FileNotFoundError:
        return b""


def _read_all_text_bounded(p: Path, max_bytes: int) -> tuple[str, bool]:
    try:
        with p.open("rb") as f:
            data = f.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        if truncated:
            data = data[:max_bytes]
        return data.decode("utf-8", errors="replace"), truncated
    except FileNotFoundError:
        return "", False


def _allowlist_env() -> dict[str, str]:
    """Build a minimal environment for subprocess execution."""
    allow: dict[str, str] = {}
    for k, v in os.environ.items():
        if k == "PATH":
            allow[k] = v
        elif k == "PYTHONPATH":
            parts = [p for p in v.split(os.pathsep) if p]
            keep: list[str] = []
            for p in parts:
                try:
                    pp = os.path.abspath(p)
                    if os.path.isabs(pp) and os.path.isdir(pp):
                        keep.append(pp)
                except Exception:
                    continue
            if keep:
                allow["PYTHONPATH"] = os.pathsep.join(keep)
        elif k.startswith("LANG") or k.startswith("LC_"):
            allow[k] = v
    allow["PYTHONHASHSEED"] = "0"
    allow.setdefault("OMP_NUM_THREADS", "1")
    allow.setdefault("MKL_NUM_THREADS", "1")
    allow.setdefault("OPENBLAS_NUM_THREADS", "1")
    return allow


def run_candidate(
    code_path: Path,
    timeout_s: int = 60,
    run_dir: Path | None = None,
) -> RunResult:
    """Execute a candidate Python file and classify the result."""
    if run_dir is None:
        run_dir = (
            code_path.parent
            / f".run_{int(time.time() * 1000)}_{os.getpid()}_{random.randint(0, 9999):04d}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy candidate to run directory
    exec_filename = "candidate_main.py"
    code_dst = run_dir / exec_filename
    shutil.copy2(code_path, code_dst)

    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"

    argv = [sys.executable, "-u", exec_filename]
    env = _allowlist_env()

    t_started = time.time()

    f_out = stdout_path.open("wb")
    f_err = stderr_path.open("wb")
    try:
        p = subprocess.Popen(
            argv,
            cwd=str(run_dir),
            stdin=subprocess.DEVNULL,
            stdout=f_out,
            stderr=f_err,
            start_new_session=True,
            env=env,
        )
    except Exception:
        f_out.close()
        f_err.close()
        raise

    rc: int
    try:
        while True:
            try:
                rc = p.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                if time.time() - t_started > timeout_s:
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except Exception:
                        pass
                    try:
                        p.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(p.pid, signal.SIGKILL)
                        except Exception:
                            pass
                        p.wait(timeout=2.0)
                    rc = p.returncode if p.returncode is not None else -9
                    break
    finally:
        t_finished = time.time()
        try:
            f_out.close()
        except Exception:
            pass
        try:
            f_err.close()
        except Exception:
            pass

    # Read bounded scan for classification
    out_text, scan_truncated = _read_all_text_bounded(stdout_path, MAX_SCAN_BYTES)

    # Classification
    passed = False
    validator = "unknown"
    reason = ""
    if rc == 0:
        if _PASS_REGEX.search(out_text):
            passed = True
            validator = "run_tests"
            reason = "run_tests printed PASS and exited 0"
        elif _SENTINEL in out_text:
            passed = True
            validator = "sentinel"
            reason = "sentinel ALL_TESTS_PASSED found and exited 0"
        else:
            passed = False
            validator = "unknown"
            reason = "rc==0 but neither PASS nor sentinel found"
            if scan_truncated:
                reason += " (scan_truncated=true)"
    else:
        passed = False
        reason = f"nonzero exit code: {rc}"

    # Read tails for output
    stdout_tail = _tail_bytes(stdout_path, STDOUT_MAX_TAIL).decode(
        "utf-8", errors="replace"
    )
    stderr_tail = _tail_bytes(stderr_path, STDERR_MAX_TAIL).decode(
        "utf-8", errors="replace"
    )

    return RunResult(
        rc=rc,
        passed=passed,
        validator=validator,
        reason=reason,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        elapsed_s=round(t_finished - t_started, 3),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Execute a candidate Python file and classify PASS/FAIL"
    )
    p.add_argument("--code-path", required=True, help="Path to the Python file to run")
    p.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")
    p.add_argument("--run-dir", default=None, help="Working directory for execution")
    args = p.parse_args(argv)

    code_path = Path(args.code_path).resolve()
    if not code_path.is_file():
        print(json.dumps({"error": f"File not found: {code_path}"}))
        return 2

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    result = run_candidate(code_path, timeout_s=args.timeout, run_dir=run_dir)

    print(json.dumps(asdict(result), indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
