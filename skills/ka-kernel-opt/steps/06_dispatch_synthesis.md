# Step: Per-Workload Niching + Dispatch Synthesis (direction 3)

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-opt skill directory; the
> tool scripts live at `${CLAUDE_SKILL_DIR}/tools/`.

## Purpose

Gated alternative to single-proxy selection, for problems whose `WORKLOADS`
span a wide regime (see `insights/DIRECTION 3 PROPOSAL - PER-WORKLOAD
NICHING.md` for the validation this is based on: 4 independent real-hardware
tests — GEMM, `fp8_group_gemm`, GDN hand-authored, GDN generated x3 — all
converging on the same finding). A single round-proxy metric has no way to
notice that its "best" candidate is unsafe or badly regressed on a workload
it never looked at; this tracks a champion **per representative workload**
instead of one global best, then routes between the intact, independently-
verified champions by input shape.

**Only active when `NICHE_WORKLOADS=true`** (default `false` — this changes
cost and only helps problems with real regime diversity; see "Ground Rules"
in `SKILL.md`). Requires the problem to define `WORKLOADS` +
`build_workload_inputs(i)` (the unified `problem.md` contract) — problems
with only a single canonical `get_inputs()` workload have nothing to niche
on; fall back to normal single-proxy optimization and note why.

## Step 1: Select Representative Workloads (once, at Round 0)

```bash
python "${CLAUDE_SKILL_DIR}/tools/select_workloads.py" \
  --problem $RUN_DIR/problem.py --n ${NUM_NICHES:-3}
```

Record `NICHE_INDICES` (the `workload_indices` list from the output) — fixed
for the whole run. Default ranking is by total problem size (product of
numeric axes); pass `--axis <name>` instead if the problem is known to vary
along one specific axis rather than overall size (e.g. `--axis seq_len` when
batch_size is expected to be a weaker driver of regime behavior — check this
against the problem's own description/axes table before assuming).

## Step 2: Benchmark Every Candidate Against Every Niche Workload

Replaces step 05's single `benchmark.py` call. For each verified candidate,
benchmark against **every** index in `NICHE_INDICES` (not just the round's
canonical workload):

```bash
declare -A WL_TIMES
for i in $NICHE_INDICES; do
  T=$(python "${CLAUDE_SKILL_DIR}/tools/benchmark.py" \
    --mode kernel --kernel $RUN_DIR/kernel.py --problem $RUN_DIR/problem.py \
    --workload-index $i --warmup $WARMUP --repeat $REPEAT | \
    python3 -c "import json,sys; print(json.load(sys.stdin)['time_ms'])")
  WL_TIMES[$i]=$T
done
```

This is real, measured extra cost — roughly `len(NICHE_INDICES)x` the
benchmarking work of a single-proxy round (the realistic GEMM test measured
~1.6x total, since not every lineage needs re-benchmarking every round; see
the proposal doc). That's the whole tradeoff this gate exists to make
explicit — don't enable it by default.

Register with `--metrics-by-workload` (JSON `{"wl<i>": time_ms, ...}`) in
addition to the normal `--time-ms` (canonical workload, first index in
`NICHE_INDICES`, for backward-compatible reporting):

```bash
python "${CLAUDE_SKILL_DIR}/tools/program_db.py" add \
  --db $RUN_DIR/program_db.json --kernel-file $RUN_DIR/kernel_round_$ROUND.py \
  --time-ms $NEW_TIME_MS --round $ROUND --parent-id $PARENT_ID \
  --category "$BOTTLENECK_CATEGORY" --fix "$FIX_ONE_LINER" \
  --metrics-by-workload "$(python3 -c "import json; print(json.dumps({f'wl{i}': t for i, t in WL_TIMES.items()}))")"
```

## Step 3: Select Parents Via Niching

Replace step 1's `program_db.py best`/`top --k 2` with:

```bash
python "${CLAUDE_SKILL_DIR}/tools/program_db.py" top-niche --db $RUN_DIR/program_db.json
```

Returns one champion per niche workload (deduplicated by program_id). Each
distinct champion is a parent for this round — diagnose + rewrite one
candidate per parent, same fan-out pattern as beam mode. This is what
protects a workload-1 specialist that a single-proxy ranking on workload-2
would silently drop, exactly the failure mode direction 1 (lineage-floor
diversity) tried and failed to fix — see the postmortem for why that
approach doesn't work and this one does.

## Step 4: Dispatch Synthesis (at Finalize, replacing the single-kernel copy)

After the round budget (or a stop condition) is reached:

```bash
python "${CLAUDE_SKILL_DIR}/tools/program_db.py" top-niche --db $RUN_DIR/program_db.json
```

- **If every niche's champion is the same `program_id`**: niching found
  nothing to combine (one kernel is simply best everywhere tested) —
  finalize normally, copy that kernel as `optimized_kernel.py`, note in the
  report that niching was enabled but had nothing to do.
- **Otherwise**: build a dispatcher.
  1. For each pair of champions adjacent in the workload-size ranking
     (`select_workloads.py`'s ordering), locate a routing threshold: either
     the midpoint between their two representative workloads' sizes (cheap,
     used in every test so far), or a real sweep across intermediate
     `WORKLOADS` entries if the problem's own size range is coarse enough
     that the midpoint is a poor proxy — prefer the sweep when time allows,
     it's what caught the true crossover point in the GEMM/`fp8_group_gemm`
     tests rather than assuming linearity.
  2. Write a thin dispatcher module: import each intact, unmodified champion
     kernel under its own name, route by whichever axis actually drives the
     regime split — **check this against a real sweep of a few intermediate
     `WORKLOADS` entries before trusting `select_workloads.py`'s selection
     axis** (its default, total size, is a reasonable *starting point* for
     picking representative workloads, not a guarantee it's the axis that
     explains why champions differ — confirmed on the real GDN run: the
     actual driver turned out to be `batch_size` alone, not the size
     product, found only by sweeping and comparing). Read the routing axis
     directly off the input tensors' shapes at call time, not off a
     workload index — the dispatcher receives raw tensors, not a
     `WORKLOADS` entry. Call the matching champion's `kernel_function`
     unchanged. **Lazy-import** any champion built on `torch.compile` or
     similar — eagerly importing a compiled module measurably taxes an
     unrelated eager code path even when never called (real bug, found and
     fixed in the GDN test; see the proposal doc). **Add
     `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`
     before importing sibling champion modules** — `benchmark.py`/
     `profile_ncu.py` load a kernel file via
     `importlib.util.spec_from_file_location`, which does not add the
     file's own directory to `sys.path`, so a plain `import champion_rN`
     resolves under `run_candidate.py`'s test execution but fails under
     benchmarking/profiling otherwise (real bug hit on first use of this
     step). Never modify a champion's internals — only route between them.
  3. Verify: run `test.py` (iterates the full `WORKLOADS` list per the
     unified contract) — must PASS on every workload, not just the niched
     ones.
  4. Benchmark the dispatcher against every workload in the problem's full
     `WORKLOADS` list (not just `NICHE_INDICES`) to report real coverage,
     including workloads between the niched points — this is where a
     too-coarse `NUM_NICHES` or a bad routing threshold would show up as a
     regression neither champion alone has.
  5. Save as `$KERNEL_DIR/optimized_kernel.py`.

## Output

- `optimized_kernel.py`: either a single champion (niching had nothing to
  combine) or a dispatcher between 2+ intact champions.
- Report which case occurred, the champions and their niche workloads, the
  routing threshold(s) chosen and how (midpoint vs. swept), and the full
  `WORKLOADS` benchmark table — including any workload that regressed
  relative to what a single-proxy run would have produced, honestly, the
  same as any other round in this skill.
