# Step: Lazy/Reactive Niche Scan (direction 3, improvement #2)

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-opt skill directory; the
> tool scripts live at `${CLAUDE_SKILL_DIR}/tools/`.

## Purpose

Alternative to `06_dispatch_synthesis.md`'s always-on niching, motivated by
a controlled, 3x-replicated finding: agent-managed niching (tracking
multiple workloads every round, checking `top-niche`, watching for
divergence) cost more in per-round fix quality than it recovered through
protection (see `insights/DIRECTION 3 PROPOSAL - PER-WORKLOAD NICHING.md`).
This step never touches the agent's reasoning. It runs the *exact* vanilla
single-proxy loop untouched, then — only at Finalize, only as a scripted,
non-agent pass — checks whether any candidate the run already produced,
including rounds that were **rejected** as regressions on the canonical
workload but still passed correctness and are still sitting on disk, is
secretly the best choice for some other real workload. No extra kernel
generation, no extra rounds, no extra LLM reasoning.

**Gated by `LAZY_NICHE_SCAN=true`** (default `true` — validated in offline
replay against 3 real controlled runs, and exercised live end-to-end on a
4th, real problem (see the proposal doc for the current evidence). Requires
the problem to define `WORKLOADS` + `build_workload_inputs(i)` (the unified
`problem.md` contract) — no-ops otherwise, so it's safe to leave on for
single-workload problems too. Mutually exclusive with `NICHE_WORKLOADS=true`
(that mode already does its own Finalize-time niche check via `top-niche`;
running both is redundant, not additive).

## Step 1: Scan (cheap, always safe to run)

After the round budget is exhausted (or any other stop condition), before
the normal single-kernel copy:

```bash
python "${CLAUDE_SKILL_DIR}/tools/select_workloads.py" \
  --problem $RUN_DIR/problem.py --n ${NUM_NICHES:-3}
```

Take the `workload_indices` from that output (or use the full `WORKLOADS`
list directly if it's small — under ~20 entries, per the cost measured
below) and run:

```bash
python "${CLAUDE_SKILL_DIR}/tools/lazy_niche_scan.py" scan \
  --db $RUN_DIR/program_db.json --problem $RUN_DIR/problem.py \
  --tools-dir "${CLAUDE_SKILL_DIR}/tools" --workloads <indices>
```

This benchmarks **every** candidate already registered in `program_db.json`
(the initial kernel plus every round, accepted or rejected-but-correct)
against every selected workload — real GPU time, but no agent reasoning
and no new kernels. Cost measured on a real 6-candidate, 16-workload run:
~96 `benchmark.py` calls, a few minutes wall-clock. **Run scans for
different runs sequentially, not concurrently** — running multiple scans
in parallel on a shared GPU produced 2–3.4x contention noise in testing,
enough to fabricate spurious "rescue opportunities" that don't survive a
clean, isolated rerun. If in doubt, rerun once cleanly before trusting a
result.

If the output says "Nothing to rescue" (the canonical champion —
`program_db.py best`'s own choice — already dominates every scanned
workload), stop here and finalize normally (copy the canonical champion as
`optimized_kernel.py`). This is the common case (2 of 3 real controlled
runs tested): the extra cost was small and nothing changes.

## Step 2: Dispatch (only if a rescue was found)

If the scan reports 2 distinct champions:

```bash
python "${CLAUDE_SKILL_DIR}/tools/lazy_niche_scan.py" dispatch \
  --db $RUN_DIR/program_db.json --problem $RUN_DIR/problem.py \
  --tools-dir "${CLAUDE_SKILL_DIR}/tools" \
  --champions <champion_1> <champion_2> \
  --workloads <all real WORKLOADS indices, not just the scan subset> \
  -o $RUN_DIR/lazy_dispatcher.py
```

This does a full sweep of both champions across every real workload —
**repeated 3x independently by default (`--repeats`)**, not a single pass —
then runs an exhaustive decision-stump search over the problem's own
numeric `WORKLOADS` axes, using the per-workload *median* across repeats,
for the routing rule that minimizes **worst-case regret** (the largest
slowdown any workload would suffer versus its own best available champion)
— not just raw misclassification count, which can hide a large regression
behind a good-looking average. The chosen rule is then re-checked against
*each individual repeat's own numbers*, and **the tool refuses to write a
dispatcher unless the worst regret across all repeats stays at or below
1.10x** — not just the median. This repeat-and-take-worst-case design
exists because a single-pass version of this gate was found, live on a
real shared GPU box, to flip between "refuse" and "ship" on 2 of 5
identical sequential runs purely from ordinary contention noise (the
deciding gap between champions was only a couple percent, near the
benchmark noise floor) — a safety gate a single noisy sample can fool
isn't a safety gate. Separately, a real, measured failure mode from
earlier testing: a rejected round-4 candidate rescued 6/16 workloads by
1.09-1.88x on average, but the *best available* single-axis routing rule
for that exact case still left a 1.478x regression on one workload,
because the true separating structure wasn't fully explained by any one
axis. When the tool refuses (for either reason), treat this the same as no
rescue — finalize with the canonical champion alone. Do not force a
dispatcher through by hand-picking a threshold the tool rejected, and
don't lower `--repeats` to make a borderline case pass; both defeat
exactly the unquantified-risk case this gate exists to catch.

**If more than 2 distinct champions exist**, this tool's MVP doesn't
handle it — fall back to `06_dispatch_synthesis.md`'s agent-authored
dispatch (which can reason about 3+ champions and axes jointly) instead of
forcing this mechanical path.

If the tool succeeds, it writes a working dispatcher module. Before doing
so, it already checked — automatically, not as an agent judgment call —
that the chosen routing axis actually equals `args[0].shape[0]` at
runtime on every scanned workload (building real inputs via the problem's
own `build_workload_inputs` to confirm this); if that check fails, it
refuses to write a dispatcher at all, the same way it refuses on an
unsafe regret score. So a dispatcher only ever gets written when the
extraction line is already verified correct — no manual check is load-
bearing for trust here. Still required before shipping:
- Run `test.py` (iterates the full `WORKLOADS` list) — must PASS on every
  workload, not just the ones scanned. This is the one thing the tool
  itself doesn't verify (it checks routing/performance, not functional
  correctness of the underlying candidate kernels).
- Copy as `$KERNEL_DIR/optimized_kernel.py`.

## Output

- `optimized_kernel.py`: either the canonical champion alone (nothing to
  rescue, or a rescue existed but no safe routing rule was found), or a
  mechanically-synthesized dispatcher between two intact, already-verified
  candidates.
- Report which case occurred, the champions and their per-workload
  numbers, the routing axis/threshold and its worst-case regret if a
  dispatcher was built, and the real cost (number of extra `benchmark.py`
  calls) — same honesty standard as every other step in this skill.
