#!/usr/bin/env python3
"""
Lazy/reactive niching: post-hoc scan of an already-completed vanilla run.

Standalone: stdlib only, except importing the problem file directly for its
WORKLOADS list (same convention as select_workloads.py -- no active GPU
needed for that part, but the environment needs torch installed since
problem.py itself imports it). Actual timing shells out to benchmark.py
subprocesses.

Prototype + reference implementation for "direction 3, improvement #2" (see
insights/DIRECTION 3 PROPOSAL - PER-WORKLOAD NICHING.md, "Improvement #2:
lazy/reactive niching"). Always-on niching forces the agent to carry
per-round bookkeeping (tracking NUM_NICHES workloads, checking top-niche,
watching for divergence) in its own reasoning -- measured, in a controlled
3x-replicated test, to cost more in fix quality than it recovered through
protection. This is the opposite design: run the *exact* vanilla
single-proxy loop untouched -- the agent never knows this exists -- and only
after the round budget is spent, do a purely mechanical, scripted post-hoc
pass over candidates the run already produced, including rounds that were
*rejected* as regressions on the canonical workload but still passed
correctness and are still sitting on disk. No extra kernel generation, no
extra agent rounds, no extra LLM reasoning -- only extra `benchmark.py`
calls and, if a rescue is found, mechanical dispatcher synthesis (decision-
stump threshold search over the problem's own WORKLOADS axes -- no agent
involvement there either).

Tested against 3 real controlled vanilla runs (060_chunk_gated_delta_rule_
linear_attention): 2/3 found nothing to rescue (canonical champion already
dominant on all 16 real workloads -- cost was ~96 cheap benchmark calls,
no change). 1/3 found a real one: a round rejected for regressing 45% on
the canonical workload turned out to win 6/16 workloads by 1.09x-1.88x;
the resulting 2-kernel dispatcher beat shipping the canonical champion
alone by a geomean of 1.167x across all 16 workloads with zero regressions
anywhere (it's min(champions) per workload by construction).

Also exercised live, end-to-end (full Setup->Rounds->Finalize, not just
against saved program databases) on a 4th, real problem
(fp8_moe_gate_routing). That run found the dispatch safety gate itself was
noise-sensitive: the identical `dispatch` command, run 5x sequentially on
a real shared GPU box, flipped between "refuse" and "ship" 2/5 times from
ordinary contention noise (the deciding gap was only a couple percent).
Fixed by making `dispatch` repeat the full sweep `--repeats` times
(default 3) and requiring the WORST regret across all repeats -- not a
single pass -- to clear the safety threshold before shipping.

Usage:
    # Screen: scan every registered candidate against a handful of
    # workloads (from select_workloads.py, or the full WORKLOADS list if
    # small enough)
    python lazy_niche_scan.py scan --db program_db.json --problem problem.py \
        --tools-dir /path/to/tools --workloads 0 1 2 ...

    # If the screen found 2 distinct champions, synthesize a dispatcher
    # (mechanical decision-stump routing-axis search, no agent needed):
    python lazy_niche_scan.py dispatch --db program_db.json --problem problem.py \
        --tools-dir /path/to/tools --champions r3_3 r4_4 \
        --workloads 0 1 2 ... (all real workloads, for the routing-axis
        search and the final honest benchmark table) \
        -o dispatcher.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path


def _bench(tools_dir: Path, kernel_file: str, problem: str, workload_index: int,
           warmup: int = 15, repeat: int = 50) -> float | None:
    cmd = [
        sys.executable, str(tools_dir / "benchmark.py"),
        "--mode", "kernel", "--kernel", kernel_file, "--problem", problem,
        "--workload-index", str(workload_index), "--warmup", str(warmup), "--repeat", str(repeat),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        data = json.loads(out.stdout)
        if data.get("error"):
            # benchmark.py reports failures as structured JSON with
            # time_ms=None rather than a nonzero-exit crash -- surface that
            # error text here instead of letting it collapse into a silent
            # None (e.g. a missing build_workload_inputs would otherwise
            # look identical to "no data", not "this is broken").
            print(f"  benchmark ERROR ({kernel_file}, wl{workload_index}): "
                  f"{data['error']}", file=sys.stderr)
            return None
        return data.get("time_ms")
    except Exception as e:  # noqa: BLE001
        print(f"  benchmark failed ({kernel_file}, wl{workload_index}): {e}", file=sys.stderr)
        return None


def _import_problem_module(problem_path: str):
    spec = importlib.util.spec_from_file_location("problem", problem_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_workloads(problem_path: str) -> list[dict]:
    mod = _import_problem_module(problem_path)
    if not hasattr(mod, "WORKLOADS"):
        raise SystemExit(
            f"error: {problem_path} has no WORKLOADS list -- lazy niche "
            f"scan needs the unified problem.md contract (WORKLOADS + "
            f"build_workload_inputs); see docs/PROBLEM_MD_FORMAT.md. "
            f"Nothing to do for this problem -- treat as a no-op, not a "
            f"failure."
        )
    if not hasattr(mod, "build_workload_inputs"):
        raise SystemExit(
            f"error: {problem_path} defines WORKLOADS but no "
            f"build_workload_inputs(i) -- per-workload benchmarking would "
            f"silently re-measure the canonical input under different "
            f"labels instead of actually varying the workload (see "
            f"kernel_io.load_problem). Add build_workload_inputs (see "
            f"docs/PROBLEM_MD_FORMAT.md) before running this tool on this "
            f"problem. Nothing to do for now -- treat as a no-op, not a "
            f"failure."
        )
    return mod.WORKLOADS


def _verify_axis_is_arg0_shape0(problem_path: str, axis: str, workloads: list[dict],
                                 wl_indices: list[int]) -> tuple[bool, str]:
    """The generated dispatcher can only extract its routing axis via
    `args[0].shape[0]` (see cmd_dispatch's dispatcher_src) -- it has no
    generic way to map an arbitrary axis name to a runtime expression.
    Rather than assume that mapping is correct (found live: it isn't,
    whenever the winning split is on an axis that isn't the first
    positional tensor's leading dimension -- e.g. seq_len, hidden_size, a
    non-leading dim, or an axis on a later argument), build the *actual*
    runtime inputs for every scanned workload (CPU, shape-only -- no GPU
    needed) and check the axis's declared value against what
    args[0].shape[0] really is. A candidate that fails this check would
    still pass correctness tests (both routed kernels are individually
    correct; only the routing decision would be silently wrong), so this
    has to be checked explicitly rather than assumed from the axis name
    alone."""
    try:
        mod = _import_problem_module(problem_path)
    except Exception as e:  # noqa: BLE001
        return False, f"could not import {problem_path} to verify: {e}"
    if not hasattr(mod, "build_workload_inputs"):
        return False, "problem has no build_workload_inputs to verify against"

    for i in wl_indices:
        declared = workloads[i].get("axes", {}).get(axis)
        if declared is None:
            return False, f"workload {i} has no declared value for axis {axis!r}"
        try:
            call_args = mod.build_workload_inputs(i, device="cpu")
        except TypeError:
            call_args = mod.build_workload_inputs(i)
        except Exception as e:  # noqa: BLE001
            return False, f"build_workload_inputs({i}) raised: {e}"
        if not call_args or not hasattr(call_args[0], "shape") or len(call_args[0].shape) == 0:
            return False, f"workload {i}: args[0] has no usable .shape[0]"
        actual = call_args[0].shape[0]
        if actual != declared:
            return False, (f"workload {i}: axis {axis!r} declared {declared}, "
                            f"but args[0].shape[0] is {actual}")
    return True, "verified: args[0].shape[0] matches the declared axis value on every scanned workload"


def cmd_scan(args) -> int:
    # Preflight the workload contract (WORKLOADS + build_workload_inputs)
    # before spending a single benchmark call -- `dispatch` already does
    # this via `_load_workloads`, but `scan` was calling `_bench` directly
    # without it, so a broken contract here produced silent all-missing
    # data rather than a clear refusal (see the "no valid data" check
    # below for the other half of this fix).
    workloads = _load_workloads(args.problem)
    out_of_range = [wl for wl in args.workloads if not (0 <= wl < len(workloads))]
    if out_of_range:
        print(f"error: workload index/indices {out_of_range} out of range "
              f"for {args.problem} (has {len(workloads)} workloads, valid "
              f"range 0..{len(workloads) - 1})", file=sys.stderr)
        return 1

    db = json.loads(Path(args.db).read_text())
    programs = db["programs"]
    tools_dir = Path(args.tools_dir)

    canonical_champion = min(programs, key=lambda pr: pr["metrics"]["time_ms"] or float("inf"))
    print(f"Canonical champion (min time_ms in db): {canonical_champion['program_id']} "
          f"({canonical_champion['metrics']['time_ms']:.4f}ms)")

    results: dict[int, dict[str, float]] = {wl: {} for wl in args.workloads}
    for pr in programs:
        pid = pr["program_id"]
        kf = pr["kernel_file"]
        for wl in args.workloads:
            t = _bench(tools_dir, kf, args.problem, wl)
            if t is not None:
                results[wl][pid] = t
                print(f"  {pid:>10}  wl{wl:<3}  {t:.4f}ms")

    # A workload with zero collected timings is missing data, not evidence
    # of "nothing to rescue" -- every candidate's benchmark call for it
    # failed (broken contract, invalid index passed through some other
    # path, transient GPU error, ...). Surfacing which workloads have no
    # data at all lets a caller tell "genuinely nothing here" apart from
    # "this workload never got measured."
    empty_workloads = [wl for wl in args.workloads if not results[wl]]
    if empty_workloads:
        print(f"\nWARNING: zero valid benchmark results for workload(s) "
              f"{empty_workloads} -- every candidate's benchmark call "
              f"failed for these. Excluded from champion selection below; "
              f"do not read their absence as \"canonical already wins "
              f"here\".", file=sys.stderr)

    print("\n=== Per-workload champions ===")
    champions: dict[int, str] = {}
    for wl in args.workloads:
        times = results[wl]
        if not times:
            continue
        best_pid = min(times, key=lambda k: times[k])
        champions[wl] = best_pid
        canon_t = times.get(canonical_champion["program_id"])
        best_t = times[best_pid]
        flag = "" if best_pid == canonical_champion["program_id"] else "  <-- RESCUE OPPORTUNITY"
        gap = (f" ({canon_t / best_t:.2f}x better than canonical)"
               if canon_t and best_pid != canonical_champion["program_id"] else "")
        print(f"  wl{wl}: champion={best_pid} ({best_t:.4f}ms){gap}{flag}")

    if not champions:
        # Every requested workload came back empty -- this is total
        # benchmark failure, not "canonical dominates everywhere". Treat
        # it as an error, not a silent, false "opportunity found" (which
        # is exactly what `sorted(set({}.values())) == [canonical_id]`
        # being False used to fall through to before this check existed).
        print(f"\nerror: zero valid benchmark results across all "
              f"{len(args.workloads)} requested workload(s) -- cannot "
              f"determine whether there is anything to rescue. This is a "
              f"failure, not \"nothing to rescue\" -- do not proceed to "
              f"`dispatch`.", file=sys.stderr)
        return 1

    distinct = sorted(set(champions.values()))
    print(f"\n{len(distinct)} distinct champion(s) across {len(champions)}/"
          f"{len(args.workloads)} workloads with valid data: {distinct}")
    if distinct == [canonical_champion["program_id"]]:
        print("Nothing to rescue -- canonical champion already dominates every scanned workload.")
    else:
        print(f"Dispatcher opportunity found -- run `dispatch --champions {' '.join(distinct)}` next.")

    print(json.dumps({"canonical_champion": canonical_champion["program_id"],
                       "results": results, "champions": champions}, indent=2))
    return 0


def _best_split(workloads: list[dict], wl_indices: list[int],
                 assignment: dict[int, str],
                 per_wl_times: dict[int, dict[str, float]]) -> tuple[str, float, int, float, str, str] | None:
    """Decision-stump search: find the (axis, threshold, orientation) that
    minimizes worst-case regret -- the largest slowdown any scanned
    workload would suffer versus its own best-available champion -- NOT
    the raw misclassification count. Minimizing count alone can pick a
    split that's right most of the time but catastrophically wrong
    occasionally (found on real data: a count-minimizing split left a 44%
    regression on one workload while a regret-minimizing split does
    better on that exact metric). Scans every numeric axis the problem
    defines, and **both label orientations** for every (axis, threshold)
    pair -- which champion should own the "low" side of a split is a real
    question the data has to answer, not something to fix in advance by
    label sort order. (Found live: fixing the orientation via
    `sorted(set(assignment.values()))` silently made the search blind to
    exactly the split it exists to find whenever the low-axis champion's
    program_id happened to sort after the high-axis champion's -- an
    arbitrary, data-independent coincidence, not a real signal.) Returns
    (axis, threshold, misclassified_count, worst_regret, lo_label,
    hi_label) or None if no numeric axis exists -- lo_label/hi_label are
    part of the return value now, since the caller can no longer assume
    lexicographic order is the chosen orientation.

    SAFETY INVARIANT THIS FUNCTION RELIES ON THE CALLER TO HOLD: if
    `per_wl_times[i]` is missing an expected champion, that (workload,
    prediction) pair is silently skipped rather than penalized -- a
    missing measurement is NOT the same as "this prediction was correct",
    and treating it as skippable would make an incomplete sweep look
    artificially safe (a workload with only one champion's timing present
    trivially has zero regret, since there's no second option to have
    been worse than). `cmd_dispatch` enforces a complete (repeat x
    workload x champion) matrix *before* calling this, specifically so
    this skip path is never actually exercised in practice. Any other
    caller MUST provide equivalently complete data, not rely on this
    function to catch incompleteness for them."""
    labels = sorted(set(assignment.values()))
    if len(labels) != 2:
        return None

    axis_names = {k for wl in workloads for k, v in wl.get("axes", {}).items()
                  if isinstance(v, (int, float))}
    best = None  # (axis, threshold, misclassified, worst_regret, lo_label, hi_label)
    for axis in axis_names:
        points = sorted({(workloads[i]["axes"][axis], i) for i in wl_indices
                          if axis in workloads[i].get("axes", {})})
        if len(points) < 2:
            continue
        values = [v for v, _ in points]
        thresholds = [(values[k] + values[k + 1]) / 2 for k in range(len(values) - 1)]
        for thresh in thresholds:
            for lo_label, hi_label in (labels, labels[::-1]):
                miscls = 0
                worst_regret = 1.0
                for v, i in points:
                    predicted = lo_label if v <= thresh else hi_label
                    times = per_wl_times[i]
                    if predicted not in times:
                        continue
                    routed_t = times[predicted]
                    best_t = min(times.values())
                    regret = routed_t / best_t
                    worst_regret = max(worst_regret, regret)
                    if assignment[i] != predicted:
                        miscls += 1
                key = (worst_regret, miscls)
                if best is None or key < (best[3], best[2]):
                    best = (axis, thresh, miscls, worst_regret, lo_label, hi_label)
    return best


def _regret_for_rule(workloads: list[dict], wl_indices: list[int], axis: str,
                      threshold: float, lo_label: str, hi_label: str,
                      per_wl_times: dict[int, dict[str, float]]) -> float:
    """Worst-case regret of a FIXED (axis, threshold) rule against one
    measurement set -- no search, just evaluation. Used to re-check a
    chosen rule against each independent repeat's own numbers.

    Same safety invariant as `_best_split` above: relies on the caller
    (`cmd_dispatch`, via its upfront complete-matrix check) to guarantee
    `per_wl_times` has every champion for every workload -- a missing
    entry is skipped here, not penalized, so incomplete data fed to this
    function would understate regret rather than correctly refuse."""
    worst = 1.0
    for i in wl_indices:
        axes = workloads[i].get("axes", {})
        if axis not in axes:
            continue
        times = per_wl_times.get(i, {})
        if not times:
            continue
        predicted = lo_label if axes[axis] <= threshold else hi_label
        if predicted not in times:
            continue
        worst = max(worst, times[predicted] / min(times.values()))
    return worst


def cmd_dispatch(args) -> int:
    db = json.loads(Path(args.db).read_text())
    programs = {pr["program_id"]: pr for pr in db["programs"]}
    tools_dir = Path(args.tools_dir)
    workloads = _load_workloads(args.problem)

    if len(args.champions) != 2:
        print("error: this MVP only synthesizes a dispatcher for exactly 2 "
              "champions (found more in the scan? extend this tool before "
              "using it on that case)", file=sys.stderr)
        return 1
    for c in args.champions:
        if c not in programs:
            print(f"error: {c} not found in {args.db}", file=sys.stderr)
            return 1

    # Repeat the full sweep independently rather than trusting one pass.
    # Found live, on a real (shared) GPU box: the same dispatch command run
    # 5x sequentially flipped between "refuse" and "ship" 2/5 times purely
    # from ordinary contention noise, because the deciding gap between
    # champions can be only a couple percent -- right at the noise floor.
    # A safety gate that a single noisy sample can fool isn't a safety
    # gate. Rule *selection* uses the per-workload median across repeats
    # (robust to one outlier repeat); the *safety decision* re-checks that
    # chosen rule against every individual repeat and requires the WORST
    # of them to clear the threshold, matching this tool's own worst-case
    # framing -- one bad repeat is enough to correctly refuse.
    print(f"Full sweep (all requested workloads) for the two champions, "
          f"x{args.repeats} independent repeats:")
    all_repeats: list[dict[int, dict[str, float]]] = []
    for rep in range(args.repeats):
        per_wl_times: dict[int, dict[str, float]] = {}
        for wl in args.workloads:
            per_wl_times[wl] = {}
            for c in args.champions:
                t = _bench(tools_dir, programs[c]["kernel_file"], args.problem, wl)
                if t is not None:
                    per_wl_times[wl][c] = t
                    print(f"  rep{rep}  {c:>10}  wl{wl:<3}  {t:.4f}ms")
        all_repeats.append(per_wl_times)

    # Require a COMPLETE matrix -- every repeat x every requested workload
    # x every champion -- before any regret is computed from it. Missing
    # cells were previously dropped silently (median_times only includes
    # what exists; _best_split/_regret_for_rule skip missing predicted
    # labels), which lets a partially-failed sweep report an artificially
    # low, meaningless regret instead of failing loudly: a workload with
    # only one champion's timing present trivially "routes correctly"
    # with regret 1.0, because there's no second option to have been
    # worse than -- not because the routed branch was actually measured
    # against its alternative. A safety gate computed from data that was
    # never fully collected isn't a safety gate.
    missing_cells = [
        (rep_idx, wl, c)
        for rep_idx, per_wl_times in enumerate(all_repeats)
        for wl in args.workloads
        for c in args.champions
        if c not in per_wl_times.get(wl, {})
    ]
    if missing_cells:
        print(f"\nerror: incomplete benchmark matrix -- {len(missing_cells)} "
              f"missing (repeat, workload, champion) cell(s), e.g. "
              f"{missing_cells[:5]}{'...' if len(missing_cells) > 5 else ''}. "
              f"Refusing to compute regret from partial data -- this is "
              f"not a safe basis for a ship/no-ship decision. Check "
              f"benchmark.py's stderr output above for the underlying "
              f"failure (a broken workload contract, an invalid index, or "
              f"a transient GPU error) and rerun once it's fixed.",
              file=sys.stderr)
        return 1

    median_times: dict[int, dict[str, float]] = {}
    for wl in args.workloads:
        median_times[wl] = {}
        for c in args.champions:
            vals = [rep[wl][c] for rep in all_repeats if c in rep.get(wl, {})]
            if vals:
                median_times[wl][c] = statistics.median(vals)

    assignment = {wl: min(times, key=lambda k: times[k])
                  for wl, times in median_times.items() if times}

    split = _best_split(workloads, list(assignment.keys()), assignment, median_times)
    if split is None:
        print("error: could not find a numeric axis to split on -- cannot "
              "synthesize a mechanical dispatcher for this problem "
              "(needs an agent-authored routing rule instead)", file=sys.stderr)
        return 1
    axis, threshold, miscls, median_worst_regret, lo_champion, hi_champion = split

    per_repeat_regret = [
        _regret_for_rule(workloads, list(assignment.keys()), axis, threshold,
                          lo_champion, hi_champion, rep)
        for rep in all_repeats
    ]
    worst_regret = max(per_repeat_regret)

    print(f"\nBest routing split (minimizing worst-case regret on the "
          f"median measurements, not just misclassification count): "
          f"axis={axis!r}, threshold={threshold}")
    print(f"  misclassifies {miscls}/{len(assignment)} scanned workloads on "
          f"the median; per-repeat worst-case regret: "
          f"{[f'{r:.3f}x' for r in per_repeat_regret]} "
          f"-> using the worst across all {args.repeats} repeats: "
          f"{worst_regret:.3f}x (1.0 = never worse than the best available "
          f"choice)")
    SAFE_REGRET_THRESHOLD = 1.10
    if worst_regret > SAFE_REGRET_THRESHOLD:
        print(f"  UNSAFE: worst-case regret {worst_regret:.2f}x (across "
              f"{args.repeats} independent repeats) exceeds the "
              f"{SAFE_REGRET_THRESHOLD}x safety threshold -- some workload "
              f"gets a real, non-trivial regression under any single-axis "
              f"routing rule this search could find, on at least one "
              f"repeat. Refusing to write a dispatcher automatically (this "
              f"MVP only auto-ships when it's actually safe on every "
              f"repeat, not just on average). The rescue opportunity is "
              f"real (see the per-workload champions above) but needs "
              f"either a smarter routing rule than a single-axis "
              f"threshold, or a hand-authored one -- flag for follow-up "
              f"rather than ship blind.")
        return 1

    axis_ok, axis_detail = _verify_axis_is_arg0_shape0(args.problem, axis, workloads,
                                                         list(assignment.keys()))
    if not axis_ok:
        print(f"  UNSAFE: the chosen routing axis {axis!r} does not match "
              f"args[0].shape[0] at runtime ({axis_detail}) -- the generated "
              f"dispatcher can only extract the routing axis from "
              f"args[0].shape[0] (see cmd_dispatch's generated "
              f"kernel_function), so a mismatch here means the dispatcher "
              f"would route on the wrong value, silently, while still "
              f"passing correctness tests (both branches are valid "
              f"kernels; only which one gets picked would be wrong).  "
              f"Refusing to write a dispatcher automatically. The rescue "
              f"opportunity is real (see the per-workload champions above) "
              f"but this problem's routing axis needs a hand-authored "
              f"dispatcher that knows how to extract it -- flag for "
              f"follow-up rather than ship a dispatcher that silently "
              f"routes on the wrong runtime value.")
        return 1

    lo_kernel = Path(programs[lo_champion]["kernel_file"])
    hi_kernel = Path(programs[hi_champion]["kernel_file"])

    dispatcher_src = f'''"""Mechanically-synthesized dispatcher (lazy/reactive niching --
tools/lazy_niche_scan.py). Routes between two candidates a normal vanilla
{{"greedy"}} search already produced -- {lo_champion!r} (registered time
{programs[lo_champion]["metrics"]["time_ms"]:.4f}ms) was NOT the canonical
best (that was {hi_champion!r} vs. this file's other branch, or vice
versa) -- but a post-hoc scan found it wins on workloads where {axis!r}
<= {threshold}. No agent authored this file; the routing rule was found by
an exhaustive decision-stump search over the problem's own WORKLOADS axes,
minimizing worst-case regret (not just misclassification count).
Misclassified {miscls}/{len(assignment)} scanned workloads at this
threshold; worst-case regret {worst_regret:.3f}x (1.0 = never worse than
the best available choice at that workload -- if this is above ~1.1-1.15,
this dispatcher is NOT risk-free: it trades a real regression at some
workload for a larger average win elsewhere. Re-run the scan with more
workloads, or fall back to hand-authored routing (which can look past a
single axis) if that's not acceptable.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_lo = None
_hi = None


def _get_lo():
    global _lo
    if _lo is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_lazy_niche_lo", {str(lo_kernel)!r})
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _lo = mod.kernel_function
    return _lo


def _get_hi():
    global _hi
    if _hi is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_lazy_niche_hi", {str(hi_kernel)!r})
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _hi = mod.kernel_function
    return _hi


def kernel_function(*args, **kwargs):
    # Route by reading {axis!r} off the actual input shapes at call time.
    # This assumes {axis!r} is inferable from args[0]'s leading dims --
    # adjust the extraction below if the axis maps differently for this
    # problem (e.g. via kwargs or a later positional argument).
    axis_value = args[0].shape[0]
    if axis_value <= {threshold}:
        return _get_lo()(*args, **kwargs)
    return _get_hi()(*args, **kwargs)
'''
    Path(args.output).write_text(dispatcher_src)
    print(f"\nWrote dispatcher to {args.output}")
    print("NOTE: the axis-extraction line (`axis_value = args[0].shape[0]`) was "
          "already verified above to match this problem's actual argument "
          "layout on every scanned workload before this file was written -- "
          "no manual check needed for that. Still run test.py on the full "
          "WORKLOADS list before shipping -- that's the one thing this tool "
          "doesn't verify (functional correctness of the underlying "
          "candidate kernels, not routing/performance).")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Lazy/reactive niching: post-hoc scan + mechanical dispatch")
    sub = p.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Screen an existing run's candidates for rescue opportunities")
    p_scan.add_argument("--db", required=True)
    p_scan.add_argument("--problem", required=True)
    p_scan.add_argument("--tools-dir", required=True)
    p_scan.add_argument("--workloads", type=int, nargs="+", required=True)
    p_scan.set_defaults(fn=cmd_scan)

    p_dispatch = sub.add_parser("dispatch", help="Synthesize a mechanical dispatcher between 2 champions")
    p_dispatch.add_argument("--db", required=True)
    p_dispatch.add_argument("--problem", required=True)
    p_dispatch.add_argument("--tools-dir", required=True)
    p_dispatch.add_argument("--champions", nargs="+", required=True)
    p_dispatch.add_argument("--workloads", type=int, nargs="+", required=True)
    p_dispatch.add_argument("--repeats", type=int, default=3,
                             help="Independent full-sweep repeats; the safety "
                                  "gate uses the worst regret across all of "
                                  "them, not a single noisy sample (default 3)")
    p_dispatch.add_argument("-o", "--output", required=True)
    p_dispatch.set_defaults(fn=cmd_dispatch)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
