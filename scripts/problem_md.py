#!/usr/bin/env python3
"""Unified single-markdown problem format for KernelAgent.

One `problem.md` replaces the SOL-ExecBench triplet (definition.json +
reference.py + workload.jsonl) and the KernelBench triplet
(problem.py + test.py + input.py):

    ---                       <- YAML front matter: name, hf_id, ...
    name: 033_post_norm_residual
    ---
    # Title / free-form description prose

    ## Axes                   <- table: axis | type(var|const) | value | description
    ## Inputs                 <- table: name | shape | dtype | role | description
    ## Outputs                <- table: name | shape | dtype | description
    ## Reference              <- ONE ```python block defining `run(...)` (PyTorch)
    ## Workloads              <- ONE ```jsonl block, SOL-ExecBench workload lines

Shapes are `[axis_or_int, ...]` or `scalar`. Roles: `input` (default,
runtime tensor), `weight` (constructor/init input), `scalar` (python
scalar; value comes from the workload line).

Subcommands:
    from-solbench <problem_dir> -o problem.md   convert a SOL-ExecBench problem
    materialize <problem.md> -o <dir>           emit problem.py + test.py
    check <problem.md>                          parse + validate only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

_DTYPES = {
    "float32", "float", "float16", "half", "bfloat16", "float64",
    "float8_e4m3fn", "float8_e5m2", "float4_e2m1fn_x2",
    "int8", "int16", "int32", "int64", "uint8", "bool",
}


def _parse_front_matter(text: str):
    meta, body = {}, text
    if text.startswith("---"):
        end = text.index("\n---", 3)
        for line in text[3:end].strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = text[end + 4:]
    return meta, body


def _sections(body: str):
    """Split on `## ` headings; returns {heading_lower: content}."""
    parts = re.split(r"^##\s+(.+?)\s*$", body, flags=re.M)
    out = {"_preamble": parts[0]}
    for head, content in zip(parts[1::2], parts[2::2]):
        out[head.strip().lower()] = content
    return out


def _parse_table(content: str):
    rows = []
    header = None
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue  # separator row
        if header is None:
            header = [c.lower() for c in cells]
        else:
            rows.append(dict(zip(header, cells)))
    return rows


def _parse_shape(text: str):
    text = text.strip().strip("`")
    if text.lower() in ("scalar", "null", "none", "-", ""):
        return None
    inner = text.strip("[]").strip()
    if not inner:
        return []  # 0-d tensor
    dims = []
    for d in inner.split(","):
        d = d.strip()
        dims.append(int(d) if re.fullmatch(r"-?\d+", d) else d)
    return dims


def _code_block(content: str, lang: str):
    m = re.search(rf"```{lang}[^\n]*\n(.*?)```", content, flags=re.S)
    if not m:
        raise ValueError(f"missing ```{lang} block")
    return m.group(1)


def parse_problem_md(path: Path) -> dict:
    meta, body = _parse_front_matter(path.read_text())
    sec = _sections(body)
    for required in ("axes", "inputs", "outputs", "reference", "workloads"):
        if required not in sec:
            raise ValueError(f"{path}: missing `## {required.title()}` section")

    axes = {}
    for r in _parse_table(sec["axes"]):
        raw = (r.get("value") or "").strip().strip("`")
        if r["type"] == "const":
            value = int(raw)
        elif r["type"] == "expr":
            value = raw  # arithmetic over other axes, resolved at build time
        else:
            value = None
        axes[r["axis"]] = {
            "type": r["type"],
            "value": value,
            "description": r.get("description", ""),
        }

    def parse_io(rows, with_role):
        out = {}
        for r in rows:
            dtype = r["dtype"].strip("`")
            if dtype not in _DTYPES:
                raise ValueError(f"unknown dtype {dtype!r}")
            entry = {
                "shape": _parse_shape(r["shape"]),
                "dtype": dtype,
                "description": r.get("description", ""),
            }
            if with_role:
                entry["role"] = (r.get("role") or "input").strip() or "input"
                if entry["shape"] is None:
                    entry["role"] = "scalar"
            out[r["name"].strip("`")] = entry
        return out

    inputs = parse_io(_parse_table(sec["inputs"]), with_role=True)
    outputs = parse_io(_parse_table(sec["outputs"]), with_role=False)

    reference = _code_block(sec["reference"], "python")
    if not re.search(r"^def run\(|^\s*def run\(", reference, flags=re.M):
        raise ValueError("reference block must define `run(...)`")

    workloads = []
    for line in _code_block(sec["workloads"], "jsonl").splitlines():
        line = line.strip()
        if line:
            workloads.append(json.loads(line))
    if not workloads:
        raise ValueError("at least one workload line required")

    # free-form description: preamble prose minus the title line
    desc_lines = [ln for ln in sec["_preamble"].splitlines() if not ln.startswith("#")]
    description = "\n".join(desc_lines).strip()

    return {
        "meta": meta,
        "description": description,
        "axes": axes,
        "inputs": inputs,
        "outputs": outputs,
        "reference": reference,
        "workloads": workloads,
    }


# --------------------------------------------------------------------------- #
# from-solbench: definition.json + workload.jsonl -> problem.md
# --------------------------------------------------------------------------- #

def _fmt_shape(shape):
    if shape is None:
        return "scalar"
    return "[" + ", ".join(str(d) for d in shape) + "]"


def from_solbench(problem_dir: Path, out_path: Path):
    definition = json.loads((problem_dir / "definition.json").read_text())
    reference = definition["reference"]
    ref_file = problem_dir / "reference.py"
    if ref_file.exists():
        reference = ref_file.read_text()
    workloads = [
        line for line in (problem_dir / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]

    lines = ["---", f"name: {definition['name']}"]
    if definition.get("hf_id"):
        lines.append(f"hf_id: {definition['hf_id']}")
    if definition.get("custom_inputs_entrypoint"):
        lines.append(f"custom_inputs_entrypoint: {definition['custom_inputs_entrypoint']}")
    if any(g.get("type") == "safetensors"
           for w in map(json.loads, workloads) for g in w.get("inputs", {}).values()):
        # blob paths in workload lines are relative to the benchmark repo root
        lines.append(f"blob_root: {problem_dir.resolve().parents[3]}")
    lines += ["---", "", f"# {definition['name']}", "", definition.get("description", ""), ""]

    lines += ["## Axes", "", "| axis | type | value | description |",
              "|------|------|-------|-------------|"]
    for name, ax in definition.get("axes", {}).items():
        if ax.get("type") == "const":
            val = ax.get("value", "-")
        elif ax.get("type") == "expr":
            val = ax.get("expression", "-")
        else:
            val = "-"
        lines.append(f"| {name} | {ax['type']} | {val} | {ax.get('description', '')} |")

    lines += ["", "## Inputs", "",
              "| name | shape | dtype | role | description |",
              "|------|-------|-------|------|-------------|"]
    for name, spec in definition["inputs"].items():
        role = "scalar" if spec["shape"] is None else "input"
        lines.append(f"| {name} | {_fmt_shape(spec['shape'])} | {spec['dtype']} "
                     f"| {role} | {spec.get('description', '')} |")

    lines += ["", "## Outputs", "",
              "| name | shape | dtype | description |",
              "|------|-------|-------|-------------|"]
    for name, spec in definition["outputs"].items():
        lines.append(f"| {name} | {_fmt_shape(spec['shape'])} | {spec['dtype']} "
                     f"| {spec.get('description', '')} |")

    lines += ["", "## Reference", "", "```python", reference.rstrip(), "```", ""]
    lines += ["## Workloads", "", "```jsonl", *workloads, "```", ""]

    out_path.write_text("\n".join(lines))
    parse_problem_md(out_path)  # validate round-trip
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# materialize: problem.md -> problem.py + test.py (KernelBench-compatible)
# --------------------------------------------------------------------------- #

_PROBLEM_TEMPLATE = '''"""Auto-generated from {md_name} by scripts/problem_md.py - do not edit.

KernelBench-format problem: `Model` wraps the reference `run(...)`;
`get_inputs()` builds the canonical (first) workload; `WORKLOADS` +
`build_workload_inputs(i)` expose every workload for the test harness.
"""

import json

import torch
import torch.nn as nn

PROBLEM_NAME = {name!r}
CUSTOM_INPUTS_ENTRYPOINT = {entrypoint!r}
BLOB_ROOT = {blob_root!r}   # override with env SOLBENCH_BLOB_ROOT
AXES = json.loads(r"""{axes_json}""")
INPUT_SPECS = json.loads(r"""{inputs_json}""")
WORKLOADS = [json.loads(line) for line in r"""
{workloads_json}
""".strip().splitlines()]

_DTYPES = {{
    "float32": torch.float32, "float": torch.float32,
    "float16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "float64": torch.float64,
    "float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2,
    "int8": torch.int8, "int16": torch.int16, "int32": torch.int32,
    "int64": torch.int64, "uint8": torch.uint8, "bool": torch.bool,
}}

_INT_RANGES = {{
    torch.int8: (-128, 128), torch.int16: (-1024, 1024),
    torch.int32: (-1024, 1024), torch.int64: (-1024, 1024),
    torch.uint8: (0, 256),
}}


def _resolve_shape(shape, axis_values):
    return [d if isinstance(d, int) else int(axis_values[d]) for d in shape]


def _rand_tensor(shape, dtype, device):
    # Mirrors SOL-ExecBench core/bench/io.py:_rand_tensor
    if dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.randn(shape, dtype=torch.float32, device=device) \\
            .clamp_(-2.0, 2.0).to(dtype)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    low, high = _INT_RANGES[dtype]
    return torch.randint(low, high, shape, dtype=dtype, device=device)




def _heuristic_tensor(name, shape, dtype, device, description=""):
    """Name-based input heuristics, ported from SOL-ExecBench
    core/bench/io.py:_generate_heuristic_tensor (same check order).
    Returns None when no heuristic applies."""
    import math
    if not dtype.is_floating_point or dtype in (
            torch.float8_e4m3fn, torch.float8_e5m2):
        return None
    desc = (description or "").lower()
    base = name.rstrip("0123456789")

    def _is_norm_param(suffix):
        if name == "norm" + suffix:
            return True
        if name.endswith(suffix):
            prefix = name[: -len(suffix)].rstrip("0123456789")
            return prefix.endswith(("norm", "layernorm"))
        return False

    if _is_norm_param("_weight"):
        return torch.ones(shape, dtype=dtype, device=device)
    if _is_norm_param("_bias"):
        return torch.zeros(shape, dtype=dtype, device=device)
    if (len(shape) >= 2 and shape[-1] == shape[-2]
            and (name in ("attention_mask", "causal_mask")
                 or ("attention mask" in desc and "causal" in desc))):
        mask = torch.zeros(shape, dtype=dtype, device=device)
        causal = torch.triu(torch.ones(shape[-1], shape[-1], device=device),
                            diagonal=1).bool()
        mask[..., causal] = torch.finfo(dtype).min
        return mask
    if name in ("x_mask", "text_mask", "aspect_ratio_mask", "drop_mask",
                "attention_mask") or (name.endswith("_mask") and "mask" in desc):
        return torch.randint(0, 2, shape, device=device).to(dtype)
    if name in ("cos", "sin", "cos_cached", "sin_cached", "rope_cos", "rope_sin"):
        t = torch.randn(shape, dtype=torch.float32, device=device) \
            .clamp_(-math.pi, math.pi)
        fn = torch.cos if name in ("cos", "cos_cached", "rope_cos") else torch.sin
        return fn(t).to(dtype)
    if (name in ("rstd", "std", "variance", "var")
            or name.endswith(("_rstd", "_std", "_variance", "_var"))
            or base in ("rstd", "std", "variance", "var")
            or base.endswith(("_rstd", "_std", "_var"))):
        return torch.randn(shape, dtype=dtype, device=device).abs() + 0.1
    if name in ("A", "A_log", "A_cumsum", "g"):
        if name == "A_cumsum":
            return torch.empty(shape, dtype=dtype, device=device) \
                .uniform_(-0.1, 0.0).cumsum(dim=-1)
        if name == "A_log":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, -1.0)
        if name == "A":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-1.0, -0.001)
        return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, 0.0)
    if name in ("attn_weights", "attention_weights", "routing_weights") \
            or "softmax" in desc:
        logits = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.softmax(logits, dim=-1).to(dtype)
    if len(shape) >= 2 and (name == "weight" or name.endswith(
            ("_weight", "_weights", "_proj", "_projs", "_proj_weight",
             "_proj_weights", "_weight_matrix"))):
        return torch.randn(shape, dtype=dtype, device=device) \
            / math.sqrt(shape[-1])
    return None


_ST_CACHE = {{}}


def _load_safetensor(rel_path: str, tensor_key: str, device: str):
    import os
    from safetensors.torch import load_file
    root = os.environ.get("SOLBENCH_BLOB_ROOT", BLOB_ROOT)
    if root is None:
        raise RuntimeError("safetensors input needs SOLBENCH_BLOB_ROOT")
    full = os.path.join(root, rel_path)
    if full not in _ST_CACHE:
        _ST_CACHE[full] = load_file(full)
    return _ST_CACHE[full][tensor_key].to(device)


def build_workload_inputs(workload_idx: int = 0, device: str = "cuda"):
    """Build the ordered input list for one workload (deterministic)."""
    wl = WORKLOADS[workload_idx]
    axis_values = dict(wl.get("axes", {{}}))
    for name, ax in AXES.items():
        if ax["type"] == "const":
            axis_values[name] = ax["value"]
    for name, ax in AXES.items():  # exprs resolve in declaration order
        if ax["type"] == "expr":
            axis_values[name] = int(eval(ax["value"], {{"__builtins__": {{}}}},
                                         dict(axis_values)))
    torch.manual_seed(workload_idx)

    # SOL-ExecBench custom inputs: fn(axes_and_scalars, device) -> dict
    custom = None
    if CUSTOM_INPUTS_ENTRYPOINT:
        scalars = {{n: g["value"] for n, g in wl["inputs"].items()
                   if g.get("type") == "scalar"}}
        custom = _CUSTOM_INPUTS_FN({{**axis_values, **scalars}},
                                    torch.device(device))

    args = []
    for name, spec in INPUT_SPECS.items():
        gen = wl["inputs"].get(name, {{"type": "random"}})
        kind = gen.get("type", "random")
        if kind == "custom":
            val = custom[name]
            args.append(val.to(device) if torch.is_tensor(val) else val)
            continue
        if kind == "safetensors":
            args.append(_load_safetensor(gen["path"], gen["tensor_key"], device))
            continue
        if spec["shape"] is None or kind == "scalar":
            args.append(gen.get("value"))
            continue
        shape = _resolve_shape(spec["shape"], axis_values)
        dtype = _DTYPES[spec["dtype"]]
        if kind == "random":
            t = _heuristic_tensor(name, tuple(shape), dtype, torch.device(device),
                                  spec.get("description"))
            args.append(t if t is not None else _rand_tensor(shape, dtype, device))
        elif kind == "zeros":
            args.append(torch.zeros(shape, dtype=dtype, device=device))
        else:
            raise ValueError(f"unsupported input generator: {{gen}}")
    return args


def workload_tolerance(workload_idx: int = 0):
    tol = WORKLOADS[workload_idx].get("tolerance", {{}})
    return float(tol.get("max_atol", 1e-2)), float(tol.get("max_rtol", 5e-2))


def gpu_key():
    """Normalized GPU key for latency-target lookup (e.g. 'h200', 'b200')."""
    name = torch.cuda.get_device_name(0).lower()
    for wl in WORKLOADS:
        for key in wl.get("latency", {{}}):
            if key.lower() in name:
                return key
    return None


def workload_latency(workload_idx: int = 0, key: str = None):
    """(baseline_ms, target_ms) for this workload on the given/current GPU,
    or None when the workload carries no target for it."""
    key = key or gpu_key()
    if key is None:
        return None
    spec = WORKLOADS[workload_idx].get("latency", {{}}).get(key)
    if spec is None:
        return None
    return float(spec["baseline"]), float(spec["target"])


# ---- reference implementation (verbatim from problem.md) ---- #

{reference}

# Snapshot the custom-inputs entrypoint BEFORE the KernelBench wrappers
# below shadow names like `get_inputs`.
_CUSTOM_INPUTS_FN = (globals()[CUSTOM_INPUTS_ENTRYPOINT]
                     if CUSTOM_INPUTS_ENTRYPOINT else None)


class Model(nn.Module):
    """Reference model: forwards straight to run()."""

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return run(*args)


def get_init_inputs():
    return []


def get_inputs():
    return build_workload_inputs(0)
'''

_TEST_TEMPLATE = '''"""Auto-generated from {md_name} by scripts/problem_md.py - do not edit.

Checks `kernel.kernel_function` against the reference over every workload
using SOL-ExecBench correctness semantics (core/bench/correctness.py):
an element matches when |out - ref| <= max_atol + max_rtol * |ref|; the
workload passes when matched_ratio >= required_matched_ratio (default
0.99), no nan/inf anywhere, output is not spuriously all-zero, and
max_abs_err <= max_error_cap when a cap is set. Exits 0 on PASS.
"""

import sys

import torch

import problem
from kernel import kernel_function


def check_one(out: torch.Tensor, ref: torch.Tensor, tol: dict):
    x, y = out.to(torch.float32), ref.to(torch.float32)
    if (~torch.isfinite(x)).any() or (~torch.isfinite(y)).any():
        return False, float("nan"), 0.0
    if y.abs().sum() > 0 and x.abs().sum() == 0:
        return False, float(y.abs().max()), 0.0
    abs_err = (x - y).abs()
    max_abs = float(abs_err.max())
    atol = float(tol.get("max_atol", 1e-2))
    rtol = float(tol.get("max_rtol", 1e-2))
    bound = atol + rtol * y.abs()
    matched = 1.0 - float((abs_err > bound).sum()) / abs_err.numel()
    ok = matched >= float(tol.get("required_matched_ratio", 0.99))
    cap = tol.get("max_error_cap")
    if cap is not None and max_abs > float(cap):
        ok = False
    return ok, max_abs, matched


def check(out, ref, tol: dict):
    """Handle single- and multi-output references uniformly."""
    outs = out if isinstance(out, (tuple, list)) else (out,)
    refs = ref if isinstance(ref, (tuple, list)) else (ref,)
    if len(outs) != len(refs):
        return False, float("nan"), 0.0
    ok, worst_err, worst_matched = True, 0.0, 1.0
    for o, r in zip(outs, refs):
        ok_i, err_i, matched_i = check_one(o, r, tol)
        ok &= ok_i
        worst_err = max(worst_err, err_i) if err_i == err_i else float("nan")
        worst_matched = min(worst_matched, matched_i)
    return ok, worst_err, worst_matched


def main() -> int:
    failures = 0
    for i, wl in enumerate(problem.WORKLOADS):
        args = problem.build_workload_inputs(i)
        model = problem.Model().cuda()
        with torch.no_grad():
            ref = model(*args)
            out = kernel_function(*args)
        ok, max_err, matched = check(out, ref, wl.get("tolerance", {{}}))
        axes = wl.get("axes", {{}})
        print(f"{{'ok  ' if ok else 'FAIL'}} workload[{{i}}] {{axes}}: "
              f"max_abs_err={{max_err:.3e}} matched={{matched:.4f}}")
        failures += (not ok)
        del args, model, ref, out
    print("PASS" if failures == 0 else f"{{failures}} FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
'''


_PERF_TEST_TEMPLATE = '''"""Auto-generated from {md_name} by scripts/problem_md.py - do not edit.

Performance goal gate for GPU spec {gpu_key!r} (this file lives in the
<problem>/{gpu_key}/ subfolder and is pinned to that spec). Benchmarks
`kernel.kernel_function` from the PARENT problem directory on every
workload carrying a {gpu_key!r} latency target (ms). Exit 0 iff every
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

GPU_KEY = {gpu_key!r}


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
        print(f"this perf test is pinned to {{GPU_KEY!r}} but the current "
              f"GPU is {{name!r}} - refusing to judge the goal")
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
        results.append(dict(workload=i, axes=wl.get("axes", {{}}), gpu=GPU_KEY,
                            measured_ms=ms, baseline_ms=baseline_ms,
                            target_ms=target_ms, target_met=met))
        print(f"{{'ok  ' if met else 'MISS'}} workload[{{i}}] {{wl.get('axes', {{}})}}: "
              f"{{ms:.4f}} ms (baseline {{baseline_ms}}, target {{target_ms}}, "
              f"{{baseline_ms / ms:.2f}}x vs baseline)")
        del args
    print(json.dumps(dict(gpu=GPU_KEY, all_targets_met=all_met, results=results)))
    print("PERF PASS" if all_met else "PERF MISS")
    return 0 if all_met else 1


if __name__ == "__main__":
    sys.exit(main())
'''


def materialize(md_path: Path, out_dir: Path):
    prob = parse_problem_md(md_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = [n for n, s in prob["inputs"].items() if s.get("role") == "weight"]
    if weights:
        print(f"note: weight-role inputs {weights} are passed as runtime args "
              "(KernelBench init-input split not yet emitted)", file=sys.stderr)

    problem_py = _PROBLEM_TEMPLATE.format(
        md_name=md_path.name,
        name=prob["meta"].get("name", md_path.stem),
        entrypoint=prob["meta"].get("custom_inputs_entrypoint") or None,
        blob_root=prob["meta"].get("blob_root") or None,
        axes_json=json.dumps(prob["axes"], indent=4),
        inputs_json=json.dumps(prob["inputs"], indent=4),
        workloads_json="\n".join(json.dumps(w) for w in prob["workloads"]),
        reference=textwrap.dedent(prob["reference"]).strip(),
    )
    (out_dir / "problem.py").write_text(problem_py)
    (out_dir / "test.py").write_text(_TEST_TEMPLATE.format(md_name=md_path.name))
    wrote = [out_dir / 'problem.py', out_dir / 'test.py']
    # one pinned perf gate per GPU spec, in a <gpu>/ subfolder
    gpu_keys = sorted({k for w in prob["workloads"]
                       for k in w.get("latency", {})})
    for gk in gpu_keys:
        gpu_dir = out_dir / gk
        gpu_dir.mkdir(exist_ok=True)
        (gpu_dir / "perf_test.py").write_text(
            _PERF_TEST_TEMPLATE.format(md_name=md_path.name, gpu_key=gk))
        wrote.append(gpu_dir / 'perf_test.py')
    print("wrote " + ", ".join(str(w) for w in wrote))


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("from-solbench", help="SOL-ExecBench problem dir -> problem.md")
    p1.add_argument("problem_dir", type=Path)
    p1.add_argument("-o", "--out", type=Path, default=None)

    p2 = sub.add_parser("materialize", help="problem.md -> problem.py + test.py")
    p2.add_argument("problem_md", type=Path)
    p2.add_argument("-o", "--out-dir", type=Path, default=None)

    p3 = sub.add_parser("check", help="parse and validate a problem.md")
    p3.add_argument("problem_md", type=Path)

    args = ap.parse_args()
    if args.cmd == "from-solbench":
        out = args.out or (args.problem_dir / "problem.md")
        from_solbench(args.problem_dir, out)
    elif args.cmd == "materialize":
        out_dir = args.out_dir or args.problem_md.parent
        materialize(args.problem_md, out_dir)
    elif args.cmd == "check":
        prob = parse_problem_md(args.problem_md)
        print(f"OK: {prob['meta'].get('name')} - "
              f"{len(prob['inputs'])} inputs, {len(prob['outputs'])} outputs, "
              f"{len(prob['workloads'])} workloads")


if __name__ == "__main__":
    main()
