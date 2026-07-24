#!/usr/bin/env python3
"""
Profile a GPU kernel with NVIDIA Nsight Compute (ncu) and emit metrics JSON.

Standalone: stdlib only in this process (torch/triton run inside the profiled
subprocess). Ported from kernel_perf_agent/kernel_opt/profiler/ncu_profiler.py
and triton_kernel_agent/opt_worker_component/profiling/ — keep the metric
list and command shape in sync.

Pipeline:
    1. Generate a wrapper script that loads problem.py + kernel.py, does
       warmup launches, then repeated kernel_function calls.
    2. Run: ncu --csv --page=raw --kernel-name-base=demangled
             --target-processes=all --replay-mode=kernel
             --profile-from-start=on --log-file=<csv> --metrics=<...>
             --launch-skip=N --launch-count=M python wrapper.py
    3. Parse the CSV (stdlib), keep the LAST profiled launch per kernel,
       and emit JSON keyed by kernel name.

Usage:
    python profile_ncu.py --kernel kernel.py --problem problem.py \
        --workdir ./artifacts [--out ncu_metrics.json] \
        [--launch-skip 3] [--launch-count 20] [--timeout 360] [--sudo]

Output (JSON, also written to --out):
    {"kernels": {"<name>": {"<metric>": value, ...}},
     "target_kernel": "<first non-PyTorch kernel>",
     "csv_path": "..."}

Notes:
    - NCU requires exclusive GPU access; do not run it concurrently with
      benchmarks. If it fails with ERR_NVGPUCTRPERM, either run with --sudo
      (or KERNELAGENT_NCU_USE_SUDO=1) or enable user profiling permissions.
    - --save-report additionally runs `ncu --set full -o <workdir>/<name>`
      to produce a .ncu-rep file for deep analysis (per-line stall
      attribution, timeline, ncu_report Python API — see the
      ncu-report-skill for the methodology). Slower: full sections replay
      the kernel many more times.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Metric selection inspired by the CudaForge team
# (https://github.com/OptimAI-Lab/CudaForge) — same list as upstream.
METRICS = [
    "sm__cycles_active.avg",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__registers_per_thread",
    "launch__block_dim_x",
    "launch__block_dim_y",
    "launch__block_dim_z",
    "launch__grid_dim_x",
    "launch__grid_dim_y",
    "launch__grid_dim_z",
    "launch__blocks_per_multiprocessor",
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum.per_second",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__shared_mem_per_block_allocated",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "lts__throughput.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    # SOL metrics consumed by roofline.py
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
]

_WRAPPER_TEMPLATE = '''\
"""Auto-generated NCU wrapper — launches the target repeatedly for profiling."""
import sys
sys.path.insert(0, {tools_dir!r})

import torch
from kernel_io import (
    build_model, detect_dtype, load_problem, prepare_inputs,
    prepare_kernel_callable,
)

TARGET = {target!r}          # "kernel" or "eager"
KERNEL_PATH = {kernel_path!r}
PROBLEM_PATH = {problem_path!r}
NUM_WARMUP = {num_warmup}
NUM_LAUNCHES = {num_launches}

Model, get_inputs, get_init_inputs = load_problem(PROBLEM_PATH)
source = open(KERNEL_PATH, encoding="utf-8").read() if KERNEL_PATH else ""
dtype = detect_dtype(source)
inputs = prepare_inputs(get_inputs(), device="cuda", dtype=dtype)
model = build_model(Model, get_init_inputs, "cuda", dtype)
if TARGET == "kernel":
    fn, args = prepare_kernel_callable(KERNEL_PATH, inputs, model)
else:
    fn, args = model, inputs

with torch.no_grad():
    for _ in range(NUM_WARMUP):
        out = fn(*args)
    torch.cuda.synchronize()
    for _ in range(NUM_LAUNCHES):
        out = fn(*args)
    torch.cuda.synchronize()

shape = out.shape if hasattr(out, "shape") else type(out).__name__
print(f"wrapper done, output: {{shape}}")
'''

# A row is the units header (not data) if a metric cell matches one of these.
_UNIT_TOKENS = (
    "%", "inst", "cycle", "block", "register", "byte", "second", "warp", "thread",
)


def _find_ncu(ncu_bin: str | None) -> str:
    if ncu_bin:
        return ncu_bin
    found = shutil.which("ncu")
    if found:
        return found
    fallback = "/usr/local/cuda/bin/ncu"
    if Path(fallback).exists():
        return fallback
    raise FileNotFoundError(
        "ncu binary not found. Install NVIDIA Nsight Compute or pass --ncu-bin."
    )


def run_ncu(
    wrapper: Path,
    csv_path: Path,
    workdir: Path,
    ncu_bin: str,
    launch_skip: int,
    launch_count: int,
    timeout: int,
    use_sudo: bool,
    python_executable: str,
) -> None:
    cmd = []
    if use_sudo:
        cmd += ["sudo", "-E", "--preserve-env=PYTHONPATH,TRITON_CACHE_DIR,PATH"]
    cmd += [
        ncu_bin,
        "--csv",
        "--page=raw",
        "--kernel-name-base=demangled",
        "--target-processes=all",
        "--replay-mode=kernel",
        "--profile-from-start=on",
        f"--log-file={csv_path}",
        f"--metrics={','.join(METRICS)}",
        f"--launch-skip={launch_skip}",
        f"--launch-count={launch_count}",
        python_executable,
        str(wrapper),
    ]

    env = os.environ.copy()
    tools_dir = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [tools_dir, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env.setdefault("TRITON_CACHE_DIR", str(workdir / ".triton_cache"))

    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-2000:]
        if "ERR_NVGPUCTRPERM" in (proc.stderr or "") or "permission" in (
            proc.stderr or ""
        ).lower():
            raise RuntimeError(
                "NCU lacks GPU performance-counter permission. Re-run with --sudo "
                "(or set KERNELAGENT_NCU_USE_SUDO=1), or enable user access: "
                "https://developer.nvidia.com/ERR_NVGPUCTRPERM\n" + stderr_tail
            )
        raise RuntimeError(
            f"ncu exited with code {proc.returncode}:\n{stderr_tail}"
        )
    if not csv_path.exists() or csv_path.stat().st_size < 100:
        raise RuntimeError(
            f"NCU produced no usable CSV at {csv_path}. stderr:\n"
            + (proc.stderr or "")[-2000:]
        )


def _coerce(value: str):
    v = value.strip().replace(",", "").rstrip("%")
    if not v:
        return None
    try:
        f = float(v)
        return f
    except ValueError:
        return value.strip()


def parse_ncu_csv(csv_path: Path) -> dict[str, dict]:
    """Parse the NCU raw-page CSV; return {kernel_name: {metric: value}} with
    the LAST profiled launch kept per kernel name."""
    lines = [
        ln
        for ln in csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip() and not ln.startswith("==")
    ]
    if not lines:
        raise ValueError(f"empty NCU CSV: {csv_path}")

    reader = csv.reader(lines)
    rows = list(reader)
    header = rows[0]
    if "Kernel Name" not in header:
        raise ValueError(f"'Kernel Name' column not found in {csv_path}")

    name_idx = header.index("Kernel Name")
    metric_idx = {m: header.index(m) for m in METRICS if m in header}
    if not metric_idx:
        raise ValueError(f"no requested metrics found in {csv_path}")

    kernels: dict[str, dict] = {}
    for row in rows[1:]:
        if len(row) <= name_idx:
            continue
        # Skip the units row (metric cells hold unit names, not numbers)
        sample = [row[i].strip().lower() for i in list(metric_idx.values())[:5] if i < len(row)]
        if sample and all(
            any(tok in cell for tok in _UNIT_TOKENS) or cell == ""
            for cell in sample
        ) and not any(_is_number(c) for c in sample):
            continue
        name = row[name_idx].strip()
        if not name:
            continue
        metrics = {}
        for m, i in metric_idx.items():
            if i < len(row):
                metrics[m] = _coerce(row[i])
        kernels[name] = metrics  # later rows overwrite -> LAST launch wins
    return kernels


def _is_number(s: str) -> bool:
    try:
        float(s.replace(",", "").rstrip("%"))
        return True
    except ValueError:
        return False


def pick_target_kernel(kernels: dict[str, dict]) -> str | None:
    """The most expensive non-PyTorch-internal (at::*) kernel, ranked by
    sm__cycles_active.avg -- NOT simply the first one launched. A kernel
    launching several real sub-kernels per call (quant, GEMM, routing, ...)
    can have its true bottleneck be anything but the first launch; picking
    launch order over cost silently profiles/diagnoses the wrong kernel on
    any multi-kernel problem (found live: a 4-sub-kernel MoE routing
    problem where the first-launched kernel had 44x less DRAM traffic than
    the actual bottleneck). Falls back to launch order only when no
    candidate kernel has a usable cycle count."""
    candidates = [
        name for name in kernels
        if not name.startswith("at::") and not name.startswith("void at::")
    ]
    if not candidates:
        return next(iter(kernels), None)

    def _cost(name: str) -> float:
        v = kernels[name].get("sm__cycles_active.avg")
        return v if isinstance(v, (int, float)) else -1.0

    if any(_cost(name) >= 0 for name in candidates):
        return max(candidates, key=_cost)
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Profile a kernel with NCU")
    p.add_argument(
        "--target",
        default="kernel",
        choices=["kernel", "eager"],
        help="Profile kernel_function (kernel) or the PyTorch reference (eager)",
    )
    p.add_argument("--kernel", default=None, help="Path to kernel.py (target=kernel)")
    p.add_argument("--problem", required=True, help="Path to problem.py")
    p.add_argument("--workdir", required=True, help="Directory for artifacts")
    p.add_argument("--out", default=None, help="Output JSON path (default: <workdir>/ncu_metrics.json)")
    p.add_argument("--csv", default="ncu_output.csv", help="CSV filename inside workdir")
    p.add_argument("--ncu-bin", default=None)
    p.add_argument("--launch-skip", type=int, default=3)
    p.add_argument("--launch-count", type=int, default=20)
    p.add_argument("--timeout", type=int, default=360)
    p.add_argument("--sudo", action="store_true", help="Run ncu under sudo -E")
    p.add_argument("--python", default=sys.executable, help="Python for the profiled subprocess")
    p.add_argument(
        "--save-report",
        default=None,
        metavar="NAME",
        help="Also run 'ncu --set full -o <workdir>/NAME' to save a .ncu-rep "
        "for deep analysis (ncu_report Python API / ncu-report-skill)",
    )
    args = p.parse_args(argv)

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    csv_path = workdir / args.csv
    out_path = Path(args.out) if args.out else workdir / "ncu_metrics.json"

    if args.target == "kernel" and not args.kernel:
        p.error("--kernel is required for --target kernel")

    wrapper = workdir / "ncu_wrapper.py"
    wrapper.write_text(
        _WRAPPER_TEMPLATE.format(
            tools_dir=str(Path(__file__).resolve().parent),
            target=args.target,
            kernel_path=str(Path(args.kernel).resolve()) if args.kernel else "",
            problem_path=str(Path(args.problem).resolve()),
            num_warmup=args.launch_skip,
            num_launches=args.launch_count + 10,
        ),
        encoding="utf-8",
    )

    use_sudo = args.sudo or os.environ.get("KERNELAGENT_NCU_USE_SUDO") == "1"
    ncu_bin = _find_ncu(args.ncu_bin)

    run_ncu(
        wrapper=wrapper,
        csv_path=csv_path,
        workdir=workdir,
        ncu_bin=ncu_bin,
        launch_skip=args.launch_skip,
        launch_count=args.launch_count,
        timeout=args.timeout,
        use_sudo=use_sudo,
        python_executable=args.python,
    )

    kernels = parse_ncu_csv(csv_path)
    result = {
        "kernels": kernels,
        "target_kernel": pick_target_kernel(kernels),
        "csv_path": str(csv_path),
    }

    if args.save_report:
        report_base = workdir / args.save_report
        cmd = []
        if use_sudo:
            cmd += ["sudo", "-E", "--preserve-env=PYTHONPATH,TRITON_CACHE_DIR,PATH"]
        cmd += [
            ncu_bin,
            "--set", "full",
            "-o", str(report_base),
            "--force-overwrite",
            f"--launch-skip={args.launch_skip}",
            "--launch-count=1",
            args.python,
            str(wrapper),
        ]
        env = os.environ.copy()
        tools_dir = str(Path(__file__).resolve().parent)
        env["PYTHONPATH"] = os.pathsep.join(
            [tools_dir, env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        env.setdefault("TRITON_CACHE_DIR", str(workdir / ".triton_cache"))
        proc = subprocess.run(
            cmd, cwd=str(workdir), env=env, capture_output=True, text=True,
            timeout=args.timeout,
        )
        if proc.returncode == 0:
            result["ncu_rep_path"] = str(report_base) + ".ncu-rep"
        else:
            result["ncu_rep_error"] = (proc.stderr or "")[-500:]

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
