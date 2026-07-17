#!/usr/bin/env python3
"""
Static PTX/SASS analysis of a GPU kernel — the compiler's-eye complement to
NCU profiling (analyze_ptx.py runs BETWEEN profiling and diagnosis so the
diagnosis can cross-check dynamic counters against what was actually
compiled).

Standalone: stdlib only in this process (torch/the DSL run inside a dump
subprocess). Mirrors the harness shape of profile_ncu.py.

Pipeline:
    1. Run the kernel once in a subprocess with DSL-specific dump hooks so
       the JIT artifacts land in <workdir>/ptx_dump_<round>/:
         - triton:  TRITON_CACHE_DIR=<dump> (+ TRITON_ALWAYS_COMPILE=1)
         - cutedsl: CUTE_DSL_KEEP_PTX=1 CUTE_DSL_KEEP_CUBIN=1
                    CUTE_DSL_DUMP_DIR=<dump>
         - tilelang: TILELANG_CACHE_DIR=<dump> (best effort)
       plus a generic scan for *.ptx / *.cubin / *.so under the dump dir.
    2. For each PTX file: parse launch directives (.reqntid/.maxntid/
       .maxnreg), local-memory decls, global load/store vector widths; run
       `ptxas --verbose` for registers / spills / barriers when available.
    3. For each cubin (or cuobjdump-extractable .so): resource usage
       (`--dump-resource-usage`) and a SASS census (`-sass`): opcode
       histogram, spill ops (LDL/STL), load/store width mix, FSETP+FSEL
       min/max expansions, SHFL/BAR counts.
    4. Emit JSON: per-artifact analysis + a deduplicated `flags` list of
       detected inefficiencies, each with evidence — ready to be read next
       to the NCU metrics JSON in the diagnose step.

Usage:
    python analyze_ptx.py --kernel kernel.py --problem problem.py \
        --workdir ./artifacts --kernel-language triton \
        [--out ptx_analysis.json] [--arch sm_100a] [--timeout 300]

Exit code 0 even when some sub-analyses are unavailable (missing ptxas /
cuobjdump / no cubin): the JSON records what was and wasn't collected —
diagnosis degrades gracefully, it must never block the round.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_WRAPPER_TEMPLATE = '''\
"""Auto-generated PTX-dump wrapper — runs the kernel once so the JIT
artifacts land in the dump directory (env hooks set by analyze_ptx.py)."""
import sys
sys.path.insert(0, {tools_dir!r})

import torch
from kernel_io import (
    build_model, detect_dtype, load_problem, prepare_inputs,
    prepare_kernel_callable,
)

KERNEL_PATH = {kernel_path!r}
PROBLEM_PATH = {problem_path!r}

Model, get_inputs, get_init_inputs = load_problem(PROBLEM_PATH)
source = open(KERNEL_PATH, encoding="utf-8").read()
dtype = detect_dtype(source)
inputs = prepare_inputs(get_inputs(), device="cuda", dtype=dtype)
model = build_model(Model, get_init_inputs, "cuda", dtype)
fn, args = prepare_kernel_callable(KERNEL_PATH, inputs, model)

with torch.no_grad():
    for _ in range(3):   # a few launches: let autotuners settle & compile
        out = fn(*args)
    torch.cuda.synchronize()

shape = out.shape if hasattr(out, "shape") else type(out).__name__
print(f"ptx-dump wrapper done, output: {{shape}}")
'''

# SASS opcodes we track explicitly (prefix match on the mnemonic).
_SASS_TRACKED = [
    "LDG", "STG", "LDS", "STS", "LDL", "STL", "LDSM",
    "FFMA", "FMUL", "FADD", "FMNMX", "FSETP", "FSEL",
    "HFMA2", "HMUL2", "HADD2", "IMAD", "LOP3", "SHF", "PRMT",
    "F2F", "F2FP", "I2F", "MUFU",
    "BRA", "BAR", "SHFL", "VOTE", "MEMBAR", "ERRBAR",
    "MMA", "HMMA", "IMMA", "QGMMA", "UTCGEN", "USETMAXREG",
    "CP.ASYNC", "UBLKCP", "ELECT", "REDUX", "ATOM", "RED",
]


def _mnemonic(line: str) -> str | None:
    """Extract the SASS mnemonic from a cuobjdump -sass line."""
    m = re.search(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.@]*)", line)
    return m.group(1) if m else None


def detect_arch(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    try:
        cap = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()[0].strip()
        major, _, minor = cap.partition(".")
        sm = f"sm_{major}{minor or '0'}"
        # Hopper/Blackwell kernels are usually built for the 'a' variant.
        if int(major) >= 9:
            sm += "a"
        return sm
    except Exception:
        return None


def _find_bin(name: str, extra: list[str]) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for cand in extra:
        if Path(cand).exists():
            return cand
    return None


def run_dump(kernel: Path, problem: Path, workdir: Path, dump_dir: Path,
             language: str, timeout: int, python_executable: str) -> dict:
    """Run the kernel once with dump hooks; returns {'ok': bool, 'log': str}."""
    wrapper = workdir / "ptx_dump_wrapper.py"
    wrapper.write_text(
        _WRAPPER_TEMPLATE.format(
            tools_dir=str(Path(__file__).resolve().parent),
            kernel_path=str(kernel.resolve()),
            problem_path=str(problem.resolve()),
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    tools_dir = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [tools_dir, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    if language == "triton":
        env["TRITON_CACHE_DIR"] = str(dump_dir / "triton")
        env["TRITON_ALWAYS_COMPILE"] = "1"
    elif language == "cutedsl":
        env["CUTE_DSL_KEEP_PTX"] = "1"
        env["CUTE_DSL_KEEP_CUBIN"] = "1"
        env["CUTE_DSL_DUMP_DIR"] = str(dump_dir / "cutedsl")
    elif language == "tilelang":
        env["TILELANG_CACHE_DIR"] = str(dump_dir / "tilelang")
        # tilelang also honors triton-style caches for fused paths
        env["TRITON_CACHE_DIR"] = str(dump_dir / "triton")
    dump_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [python_executable, str(wrapper)],
        cwd=str(workdir), env=env, capture_output=True, text=True,
        timeout=timeout,
    )
    return {
        "ok": proc.returncode == 0,
        "log": (proc.stdout + "\n" + proc.stderr)[-2000:],
    }


def analyze_ptx_text(path: Path) -> dict:
    """Parse launch directives and access-width signals out of a PTX file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    entries = re.findall(r"\.visible\s+\.entry\s+(\w+)", text)
    info: dict = {"path": str(path), "entries": entries}
    for directive in ("reqntid", "maxntid", "maxnreg", "minnctapersm"):
        m = re.search(rf"\.{directive}\s+([\d,\s]+)", text)
        if m:
            info[directive] = m.group(1).strip()
    # implied per-thread register budget from the launch width
    ntid = info.get("reqntid") or info.get("maxntid")
    if ntid:
        try:
            threads = 1
            for d in ntid.split(","):
                threads *= int(d.strip())
            info["threads_per_cta"] = threads
            info["ptxas_reg_budget"] = min(255, (65536 // threads) // 8 * 8)
        except ValueError:
            pass
    info["local_bytes_decls"] = len(re.findall(r"\.local\s+\.align", text))
    # global access width census
    widths = collections.Counter()
    for m in re.finditer(r"\b(ld|st)\.global(?:\.nc)?(?:\.\w+)*?\.(v[24])?\.?([usbf](?:8|16|32|64))", text):
        vec = {"v2": 2, "v4": 4}.get(m.group(2) or "", 1)
        bits = int(re.search(r"\d+", m.group(3)).group())
        widths[f"{m.group(1)}.{vec * bits}b"] += 1
    info["global_access_widths"] = dict(widths)
    return info


def run_ptxas(path: Path, arch: str | None, ptxas: str | None) -> dict:
    if not (ptxas and arch):
        return {"skipped": "ptxas or arch unavailable"}
    try:
        proc = subprocess.run(
            [ptxas, "--verbose", f"--gpu-name={arch}", "-o", os.devnull, str(path)],
            capture_output=True, text=True, timeout=300,
        )
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}
    out: dict = {"returncode": proc.returncode}
    txt = proc.stderr
    m = re.search(r"Used (\d+) registers", txt)
    if m:
        out["registers"] = int(m.group(1))
    m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", txt)
    if m:
        out["stack_frame_bytes"] = int(m.group(1))
        out["spill_store_bytes"] = int(m.group(2))
        out["spill_load_bytes"] = int(m.group(3))
    m = re.search(r"used (\d+) barriers", txt)
    if m:
        out["barriers"] = int(m.group(1))
    for warn in re.findall(r"\(C\d+\)[^\n]*", txt):
        out.setdefault("warnings", []).append(warn.strip()[:160])
    if proc.returncode != 0:
        out["stderr_tail"] = txt[-500:]
    return out


def analyze_sass(binary: Path, cuobjdump: str) -> dict:
    """cuobjdump resource usage + SASS census for a cubin/.so."""
    out: dict = {"path": str(binary)}
    res = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(binary)],
        capture_output=True, text=True, timeout=300,
    )
    usage = {}
    fn = None
    for line in res.stdout.splitlines():
        m = re.search(r"Function\s+(\S+)", line)
        if m:
            fn = m.group(1).rstrip(":")
        m = re.search(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)", line)
        if m and fn:
            usage[fn] = {
                "registers": int(m.group(1)),
                "stack_bytes": int(m.group(2)),
                "shared_static_bytes": int(m.group(3)),
                "local_bytes": int(m.group(4)),
            }
    out["resource_usage"] = usage

    sass = subprocess.run(
        [cuobjdump, "-sass", str(binary)],
        capture_output=True, text=True, timeout=600,
    )
    if sass.returncode != 0 or not sass.stdout:
        out["sass_error"] = (sass.stderr or "no SASS output")[-300:]
        return out

    hist: collections.Counter = collections.Counter()
    tracked: collections.Counter = collections.Counter()
    ldg_widths: collections.Counter = collections.Counter()
    sts_lds_widths: collections.Counter = collections.Counter()
    fsetp_fsel_pairs = 0
    prev = None
    total = 0
    for line in sass.stdout.splitlines():
        mn = _mnemonic(line)
        if not mn:
            continue
        total += 1
        base = mn.split(".")[0]
        hist[base] += 1
        for t in _SASS_TRACKED:
            if mn.startswith(t):
                tracked[t] += 1
                break
        if base in ("LDG", "STG"):
            w = re.search(r"\.(\d+)\b", mn)
            ldg_widths[f"{base}.{w.group(1) if w else '32'}"] += 1
        if base in ("LDS", "STS"):
            w = re.search(r"\.(\d+)\b", mn)
            sts_lds_widths[f"{base}.{w.group(1) if w else '32'}"] += 1
        if base == "FSEL" and prev in ("FSETP", "FSEL"):
            fsetp_fsel_pairs += 1
        prev = base
    out["sass_total_instructions"] = total
    out["sass_top_opcodes"] = dict(hist.most_common(25))
    out["sass_tracked"] = {k: v for k, v in tracked.items() if v}
    out["global_widths"] = dict(ldg_widths)
    out["shared_widths"] = dict(sts_lds_widths)
    out["fsetp_fsel_pairs"] = fsetp_fsel_pairs
    return out


def derive_flags(ptx_infos: list[dict], sass_infos: list[dict]) -> list[dict]:
    """Turn raw analyses into a deduplicated list of inefficiency flags with
    evidence, ordered most-severe first."""
    flags: list[dict] = []

    def add(name, severity, evidence, hint):
        flags.append({"flag": name, "severity": severity,
                      "evidence": evidence, "hint": hint})

    for s in sass_infos:
        for fn, u in (s.get("resource_usage") or {}).items():
            if u.get("stack_bytes", 0) > 0 or u.get("local_bytes", 0) > 0:
                add("register_spills", "high",
                    f"{fn[:60]}: STACK={u['stack_bytes']}B LOCAL={u['local_bytes']}B",
                    "Reduce live values / tile sizes, or rebalance per-warp "
                    "registers (setmaxnreg) — spills turn into LDL/STL traffic.")
        t = s.get("sass_tracked", {})
        if t.get("LDL", 0) + t.get("STL", 0) > 0:
            add("local_memory_ops", "high",
                f"LDL={t.get('LDL', 0)} STL={t.get('STL', 0)} in SASS",
                "Local-memory ops confirm spills or dynamically-indexed "
                "arrays — check rmem indexing and register pressure.")
        if s.get("fsetp_fsel_pairs", 0) >= 8:
            add("nan_propagating_minmax", "medium",
                f"{s['fsetp_fsel_pairs']} FSETP/FSEL pairs (NaN-safe min/max "
                "expansion, 3-4 instrs each)",
                "If NaN semantics are not required, use plain max.f32/FMNMX "
                "(e.g. cute.arch.fmax instead of cutlass.max, "
                "tl.maximum(propagate_nan=False)).")
        gw = s.get("global_widths", {})
        narrow = sum(v for k, v in gw.items() if k.endswith((".32", ".16", ".8")))
        wide = sum(v for k, v in gw.items() if k.endswith((".64", ".128")))
        if narrow > 4 and narrow > 2 * max(wide, 1):
            add("narrow_global_access", "medium",
                f"global width mix {gw} — mostly <=32-bit",
                "Vectorize global loads/stores to 128-bit (contiguous "
                "per-thread elements, alignment 16).")
        sw = s.get("shared_widths", {})
        narrow_s = sum(v for k, v in sw.items() if k.endswith((".32", ".16", ".8")))
        wide_s = sum(v for k, v in sw.items() if k.endswith((".64", ".128")))
        if narrow_s > 8 and narrow_s > 2 * max(wide_s, 1):
            add("narrow_shared_access", "low",
                f"shared width mix {sw}",
                "Widen shared-memory accesses; check for bank conflicts "
                "in the NCU l1tex conflict counters.")
        if t.get("SHFL", 0) > 16:
            add("shuffle_heavy", "low",
                f"SHFL={t['SHFL']} — convergent ops limit scheduler reordering",
                "Batch shuffles together; consider smem or layout changes "
                "that avoid cross-lane exchanges in hot loops.")
        if t.get("BAR", 0) > 8:
            add("barrier_heavy", "low",
                f"BAR={t['BAR']}",
                "Check whether __syncthreads/named barriers can be reduced "
                "via double buffering or producer/consumer pipelines.")
        hist = s.get("sass_top_opcodes", {})
        cvt = hist.get("F2F", 0) + hist.get("F2FP", 0) + hist.get("I2F", 0)
        if cvt > 0.15 * max(s.get("sass_total_instructions", 1), 1):
            add("conversion_heavy", "medium",
                f"{cvt} conversion ops of {s.get('sass_total_instructions')} total",
                "Datatype conversions dominate — keep data in the compute "
                "dtype longer or use packed 2-element conversions.")

    for pinfo in ptx_infos:
        pa = pinfo.get("ptxas", {})
        if pa.get("spill_store_bytes", 0) or pa.get("spill_load_bytes", 0):
            add("register_spills", "high",
                f"{Path(pinfo['path']).name}: spill stores "
                f"{pa['spill_store_bytes']}B loads {pa['spill_load_bytes']}B "
                f"at {pa.get('registers', '?')} regs",
                "See register_spills above.")
        for w in pa.get("warnings", []):
            add("ptxas_warning", "medium", w,
                "ptxas performance warnings usually mean an ignored "
                "directive (e.g. C7508 setmaxnreg without .maxnreg).")
        budget = pinfo.get("ptxas_reg_budget")
        regs = pa.get("registers")
        if budget and regs and regs >= budget - 8 and budget < 200:
            add("reg_budget_ceiling", "medium",
                f"{Path(pinfo['path']).name}: {regs} regs vs launch-width "
                f"budget {budget} ({pinfo.get('threads_per_cta')} thr/CTA)",
                "The per-thread register budget is 65536/threads_per_cta: "
                "wide CTAs starve the scheduler (>8 warps/CTA cliff). "
                "Consider fewer threads/CTA or per-warpgroup setmaxnreg "
                "redistribution.")
        if pinfo.get("local_bytes_decls", 0) > 0:
            add("ptx_local_decls", "medium",
                f"{Path(pinfo['path']).name}: {pinfo['local_bytes_decls']} "
                ".local declarations",
                "PTX-level local memory: dynamically indexed thread arrays "
                "or ABI stack — usually worth eliminating.")

    # dedupe by (flag, evidence), keep order high->low severity
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    seen = set()
    unique = []
    for f in sorted(flags, key=lambda f: sev_rank.get(f["severity"], 3)):
        key = (f["flag"], f["evidence"])
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _pick_target(paths: list[Path], kernel_source: str) -> list[Path]:
    """Order artifacts so the most-likely target kernel comes first: prefer
    files whose name shares a token with the kernel source's function names."""
    tokens = set(re.findall(r"def\s+(\w+)", kernel_source)) | set(
        re.findall(r"(\w+_kernel|kernel_\w+)", kernel_source)
    )

    def score(p: Path) -> tuple:
        name = p.stem.lower()
        hit = any(t.lower()[:24] in name for t in tokens if len(t) > 3)
        return (0 if hit else 1, -p.stat().st_size)

    return sorted(paths, key=score)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Static PTX/SASS kernel analysis")
    p.add_argument("--kernel", required=True)
    p.add_argument("--problem", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--kernel-language", default="triton",
                   choices=["triton", "tilelang", "cutedsl"])
    p.add_argument("--out", default=None,
                   help="Output JSON (default <workdir>/ptx_analysis.json)")
    p.add_argument("--dump-dir", default=None,
                   help="Artifact dump dir (default <workdir>/ptx_dump)")
    p.add_argument("--arch", default=None, help="e.g. sm_100a (auto-detect)")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--max-artifacts", type=int, default=6,
                   help="Analyze at most N PTX and N binary artifacts")
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args(argv)

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    dump_dir = Path(args.dump_dir) if args.dump_dir else workdir / "ptx_dump"
    out_path = Path(args.out) if args.out else workdir / "ptx_analysis.json"
    kernel = Path(args.kernel)
    result: dict = {"language": args.kernel_language, "dump_dir": str(dump_dir)}

    dump = run_dump(kernel, Path(args.problem), workdir, dump_dir,
                    args.kernel_language, args.timeout, args.python)
    result["dump_ok"] = dump["ok"]
    if not dump["ok"]:
        result["dump_log"] = dump["log"]

    ptx_files = sorted(dump_dir.rglob("*.ptx"))
    bin_files = sorted(dump_dir.rglob("*.cubin")) + sorted(dump_dir.rglob("*.so"))
    kernel_source = kernel.read_text(encoding="utf-8", errors="replace")
    ptx_files = _pick_target(ptx_files, kernel_source)[: args.max_artifacts]
    bin_files = _pick_target(bin_files, kernel_source)[: args.max_artifacts]
    result["ptx_files"] = [str(f) for f in ptx_files]
    result["binary_files"] = [str(f) for f in bin_files]

    arch = detect_arch(args.arch)
    result["arch"] = arch
    ptxas = _find_bin("ptxas", ["/usr/local/cuda/bin/ptxas"])
    cuobjdump = _find_bin("cuobjdump", ["/usr/local/cuda/bin/cuobjdump"])
    result["tools"] = {"ptxas": ptxas, "cuobjdump": cuobjdump}

    ptx_infos = []
    for f in ptx_files:
        info = analyze_ptx_text(f)
        info["ptxas"] = run_ptxas(f, arch, ptxas)
        ptx_infos.append(info)
    result["ptx"] = ptx_infos

    sass_infos = []
    if cuobjdump:
        for f in bin_files:
            try:
                info = analyze_sass(f, cuobjdump)
            except Exception as e:  # noqa: BLE001
                info = {"path": str(f), "error": str(e)[:200]}
            # host-only .so files (launchers/utils) carry no device code
            if "does not contain device code" in info.get("sass_error", ""):
                continue
            sass_infos.append(info)
    result["sass"] = sass_infos

    result["flags"] = derive_flags(ptx_infos, sass_infos)
    if not ptx_files and not bin_files:
        result["flags"].append({
            "flag": "no_artifacts", "severity": "info",
            "evidence": f"nothing dumped under {dump_dir}",
            "hint": "The DSL's dump hooks produced no PTX/cubin — diagnosis "
                    "proceeds on NCU data alone.",
        })

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
