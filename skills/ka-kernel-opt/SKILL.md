---
name: ka-kernel-opt
description: "KernelAgent kernel optimization. Profile a working GPU kernel with NCU, diagnose its bottleneck (roofline/SOL analysis), and iteratively rewrite it for speed with verified accept/reject rounds. Also supports profile-only and diagnose-only runs. Supports triton, tilelang, and cutedsl kernels."
allowed-tools:
  - Bash
---

# Skill: KernelAgent Kernel Optimization

The user's input arguments are: `$ARGUMENTS`

> **Paths:** `${CLAUDE_SKILL_DIR}` is the directory containing this SKILL.md
> (resolve it with `echo "${CLAUDE_SKILL_DIR}"` in bash if needed). Tool
> scripts live at `${CLAUDE_SKILL_DIR}/tools/`, step instructions in
> `${CLAUDE_SKILL_DIR}/steps/`, and the optimization pattern library in
> `${CLAUDE_SKILL_DIR}/reference/`.

## Purpose
Single entry point for hardware-guided kernel optimization: profile → roofline
→ diagnose → rewrite → verify → benchmark, looping until the kernel reaches
the roofline, converges, or the round budget runs out. Replaces
`triton_kernel_agent.opt_manager.OptimizationManager` /
`OptimizationWorker` / `OptimizationOrchestrator` and the `kernel_perf_agent`
pipeline, with Claude as the diagnosis/rewrite brain.

## Requirements
- A GPU machine with `torch` (+ the kernel's DSL: `triton`, `tilelang`, or
  `cutlass`), and NVIDIA Nsight Compute (`ncu`) on PATH for profiling.
- The kernel must already be **correct** — use the `ka-kernel-gen` skill to
  generate/fix kernels first.

**Companion knowledge sources** (used when installed in `.claude/skills/`;
each has a graceful fallback if absent):
- **KernelWiki** — Hopper/Blackwell optimization techniques with real PR
  references (used by the rewrite step).
- **cuda** — general CUDA optimization/debugging/profiling practice
  (used by the rewrite step).
- **ncu-report-skill** — deep NCU analysis methodology: full-set reports,
  per-line stall attribution, B200 metric names (escalation path of the
  profile step).
- **Real kernel repos** per platform at `${CLAUDE_SKILL_DIR}/reference/cuda/`
  (flash-attention, flashinfer, sgl-DeepGEMM — git submodules; run
  `git submodule update --init` if empty).

## Mode Dispatch — interpret the arguments first

| Arguments look like | Mode |
|---|---|
| a kernel directory (or kernel+problem+test paths) | **optimize** — full loop |
| `profile <dir>` or the user only wants metrics/roofline | **profile** — steps 01–02 once, report |
| `diagnose <dir>` or the user wants analysis without changes | **diagnose** — steps 01–03 once, report |

## Inputs
- `KERNEL_DIR`: directory with `input.py` (initial kernel), `problem.py`
  (defines `Model`, `get_inputs()`, optional `get_init_inputs()`), and
  `test.py` (imports `from kernel import kernel_function`; exits 0 on PASS).
  Individual paths may be given instead — copy them into this layout.
- `EXTRA_TESTS` (optional): additional test files; `test.py` is the primary
  test, extras must also pass for a candidate to count as verified.
- `<gpu>/perf_test.py` (optional, in a GPU-spec subfolder of `KERNEL_DIR`,
  e.g. `b200/perf_test.py`): a **performance goal** gate — benchmarks
  `kernel.kernel_function` (from `KERNEL_DIR`) against that spec's latency
  goals and exits 0 iff every workload passes (hard `target` ms when
  pinned, else SOL-Score >= `min_score` when the spec pins `sol`, see
  `docs/PROBLEM_MD_FORMAT.md`), 2 when run on a different GPU (emitted
  by ka-kernel-parser when the problem's workloads carry `latency`
  entries; its last-but-one stdout line is a JSON report with
  `measured_ms` / `baseline_ms` and `target_ms` or
  `sol_ms`/`sol_score`/`min_score` per workload). Pick the
  subfolder whose name matches the current GPU
  (`nvidia-smi --query-gpu=name`); if none matches, there is no goal.
  It is a GOAL, not a correctness gate: never use it to accept/reject
  candidates for correctness, and a candidate that fails it is still
  recorded normally.
- `GPU_NAME` (optional): e.g. `"NVIDIA H100 NVL 94GB"` — auto-detected if omitted
- `KERNEL_LANGUAGE`: `triton` (default), `tilelang`, or `cutedsl`
- `MAX_ROUNDS`: optimization rounds (default 5)
- `WARMUP` / `REPEAT`: benchmark iterations (defaults 25 / 100)
- `DIVERGENCE_THRESHOLD`: % slower than best before reverting (default 50)
- `MAX_NO_IMPROVEMENT`: greedy early-stop plateau (default 5 rounds)
- `BOTTLENECK_OVERRIDE` (optional): force `memory`/`compute`/`underutilized`
- `STRATEGY`: `greedy` (default) or `beam` (see Variants)
- `NICHE_WORKLOADS` (optional, default `false`): enable per-workload niching
  + dispatch synthesis (direction 3, see Variants) instead of single-proxy
  selection. Requires the problem to define `WORKLOADS` +
  `build_workload_inputs(i)` (the unified `problem.md` contract) — has
  nothing to do on problems with only one canonical workload.
- `NUM_NICHES` (optional, default 3): representative workloads to track when
  `NICHE_WORKLOADS=true`.
- `LAZY_NICHE_SCAN` (optional, default `true`): at Finalize, run a
  scripted (non-agent) post-hoc scan of every candidate the run already
  produced — including rejected-but-correct regressions still on disk —
  against other real workloads, and mechanically synthesize a dispatcher
  if a safe one exists (see Variants). Mutually exclusive with
  `NICHE_WORKLOADS=true`. Requires `WORKLOADS` + `build_workload_inputs`;
  no-ops otherwise, so it's safe to leave on unconditionally, including on
  single-workload problems.
- `BESTOF3` (optional, default `false`): per round, generate 3 independent
  rewrite candidates from the same diagnosis instead of 1, verify+benchmark
  all 3, advance only on strict improvement (see `steps/08_bestof3.md`).
  **Opt-in, not a safe default** — real, replicated wins on 2 of 4 tested
  problems, no reliable benefit on the other 2; read
  `steps/08_bestof3.md`'s "When this helps, honestly" section (or
  `insights/BEST-OF-N RESAMPLING (IDEA F) - EXPLORATION.md`) before
  recommending it for a given problem. Costs ~1.7-2.2x a vanilla round's
  true compute, every round it runs.

## Workflow (optimize mode)

Create the run directory first:
```bash
RUN_DIR=$KERNEL_DIR/.optimize/run_$(date +%Y%m%d_%H%M%S) && mkdir -p $RUN_DIR
```

### Round 0: Baselines and Database
Read and follow `${CLAUDE_SKILL_DIR}/steps/01_baseline.md` — verify the
initial kernel, measure PyTorch eager, torch.compile, and initial-kernel
times, and register the initial kernel in the program database:

```bash
python "${CLAUDE_SKILL_DIR}/tools/program_db.py" init \
  --db $RUN_DIR/program_db.json --kernel-file $RUN_DIR/kernel_round_0.py \
  --problem-id $RUN_DIR/problem.py --time-ms $INITIAL_KERNEL_MS
```

Abort the run (`success: false`, error `"Initial kernel failed correctness
verification"`) if the initial kernel fails its test.

If `NICHE_WORKLOADS=true`, also do Step 1 of
`${CLAUDE_SKILL_DIR}/steps/06_dispatch_synthesis.md` here (auto-select
`NICHE_INDICES`, fixed for the rest of the run) before entering round 1.

### Rounds 1..MAX_ROUNDS
Each round, in order (this mirrors
`OptimizationManager.run_optimization` → `select_candidates` → worker →
`update_with_results`):

1. **Select the parent kernel(s)** from the program database:
   - greedy: `program_db.py best` — always continue from the global best,
     unless step 05's divergence rule already chose the continuation.
   - beam: `program_db.py top --k 2` — the two parents for this round.
   - `NICHE_WORKLOADS=true`: `program_db.py top-niche` (step 06, Step 3) —
     one parent per distinct niche champion, same per-parent fan-out as beam.
2. **Profile** — `${CLAUDE_SKILL_DIR}/steps/02_profile.md`
   (NCU + roofline + grid analysis on the parent kernel). In round 1 also
   profile the PyTorch eager reference once (`--target eager`) and report a
   one-time **Baseline Profiling** section comparing eager vs initial kernel.
3. **Stop check** — if `roofline.at_roofline` (efficiency ≥ 95%), stop with
   success: the kernel is at the hardware limit.
4. **Diagnose** — `${CLAUDE_SKILL_DIR}/steps/03_diagnose.md`
   (you classify the bottleneck and root causes, grounded in metrics).
5. **Rewrite** — `${CLAUDE_SKILL_DIR}/steps/04_rewrite.md`
   (consult `reference/` patterns; produce `kernel_candidate.py`).
   `BESTOF3=true`: replace with `steps/08_bestof3.md` Step 1 — three
   independent candidates from this same diagnosis instead of one.
6. **Verify + Accept/Reject + Reflect** —
   `${CLAUDE_SKILL_DIR}/steps/05_verify_accept.md`
   (correctness with ≤3 refinements, benchmark, program-database update with
   lineage, two-track best tracking, divergence revert, reflexion).
   `NICHE_WORKLOADS=true`: benchmark against every `NICHE_INDICES` workload
   and register with `--metrics-by-workload` instead of a single time
   (step 06, Step 2).
   `BESTOF3=true`: replace with `steps/08_bestof3.md` Step 2-3 — verify +
   benchmark all 3 candidates, pick the fastest that passed, and apply the
   **strict** improvement-only gate (not the 50%-divergence-tolerant rule
   above) before advancing.
7. **Report the round** (see Round Reporting below).

Additional stop conditions checked at the end of each round:
- **Goal reached**: when a matching `<gpu>/perf_test.py` exists, run it
  on each newly accepted best kernel; if it exits 0 (all latency targets
  met), stop with success — the stated performance goal is achieved.
  Report the measured vs target numbers.
- **Plateau** (greedy): `MAX_NO_IMPROVEMENT` consecutive rounds without a new
  best-runtime kernel. When a perf goal exists and is NOT yet met, prefer
  continuing to the full round budget over an early plateau stop, and say
  honestly in the final report that the goal was not reached.
- **Convergence**: the last 5 rounds' efficiency varies by < 0.1%.
- Round budget exhausted.

### Round Reporting

After each round print a compact results block (the manager's per-round
table):
- time in ms, speedup vs PyTorch eager and vs the initial kernel
- when a perf goal exists: measured vs `target_ms` per targeted workload
  (from `perf_test.py`'s JSON line) and the remaining gap %
- SOL: combined / compute / memory %
- key NCU metrics when available: DRAM throughput %, DRAM BW (GB/s), warp
  active %, grid X, block X, blocks/SM, L1 hit %, L2 hit %, memory
  coalescing %, long-scoreboard stalls %
- a compact code diff (± lines only, max 20) of parent → candidate:
  `diff -u parent.py candidate.py | grep -E '^[+-][^+-]' | head -20`
- for failed candidates: the one-line failure reason

### Finalize
```bash
cp <best-runtime kernel> $KERNEL_DIR/optimized_kernel.py
python "${CLAUDE_SKILL_DIR}/tools/program_db.py" top --db $RUN_DIR/program_db.json --k 5
```

`NICHE_WORKLOADS=true`: replace the copy above with
`${CLAUDE_SKILL_DIR}/steps/06_dispatch_synthesis.md` Step 4 — synthesize a
dispatcher between champions if `top-niche` found more than one distinct
program_id, otherwise copy the single champion as usual.

`LAZY_NICHE_SCAN=true`: replace the copy above with
`${CLAUDE_SKILL_DIR}/steps/07_lazy_niche_scan.md` — scan the existing
program database (including rejected rounds) for a rescue opportunity and
mechanically synthesize a dispatcher only if a safe one exists; otherwise
copy the single champion as usual.

Report the final result in the `run_optimization` contract, plus prose:

```json
{
  "success": true,
  "kernel_path": "$KERNEL_DIR/optimized_kernel.py",
  "best_time_ms": 0.0,
  "total_rounds": 0,
  "pytorch_baseline_ms": 0.0,
  "pytorch_compile_ms": 0.0,
  "initial_kernel_time_ms": 0.0,
  "top_kernels": [{"program_id": "...", "time_ms": 0.0, "generation": 0}]
}
```

- Speedups: best vs eager, vs torch.compile, vs initial kernel
- When a perf goal exists: final `perf_test.py` verdict (targets met or
  the honest remaining gap per workload)
- Per-round summary table and final roofline state (remaining headroom)
- Run directory path (kernels per round, NCU CSVs/JSONs, diagnoses,
  program_db.json, attempts.jsonl, reflexions)

## Variants

- **Beam (STRATEGY=beam)**: mirrors `BeamSearchStrategy` (top-N kernels × M
  bottleneck directions, defaults N=2, M=2). Per round: take the top-2
  parents from `program_db.py top --k 2`; for each parent, diagnose the top-2
  bottlenecks (step 03 with 2 objects) and generate one candidate per
  direction, sequentially. All verified candidates go into the database; the
  next round's parents are again the global top-2. No plateau early-stop —
  runs the full round budget. Costs ~4× per round vs greedy — use when the
  user asks for a thorough search.
- **Profile / diagnose modes**: run steps 01–02 (and 03 for diagnose) once on
  the given kernel and present the metrics, roofline verdict, grid
  assessment, and (diagnose) root causes + recommended fixes — no rewrites.
- **Niching (`NICHE_WORKLOADS=true`)**: implements "direction 3"
  (`insights/DIRECTION 3 PROPOSAL - PER-WORKLOAD NICHING.md`) — track a
  champion per representative workload (`program_db.py top-niche`) instead
  of one global best, then synthesize a dispatcher between the intact
  champions at Finalize (`steps/06_dispatch_synthesis.md`). The selection
  mechanism is correct and repeatedly verified to protect real regressions
  when they occur. **But a controlled, 3x-replicated test found it costs
  more in per-round fix quality than it recovers through protection, at a
  round budget matched to vanilla** — niching's extra per-round bookkeeping
  (tracking `NUM_NICHES` workloads, checking `top-niche`, watching for
  divergence) appears to come at the cost of shallower diagnosis/rewrite
  depth, not just the expected extra benchmarking calls (see the proposal
  doc's "Controlled comparison" section for the numbers). **Do not enable
  this by default or recommend it for routine use** — it's implemented and
  gated off (`NICHE_WORKLOADS=false` by default) as correct, working
  infrastructure, not as something proven beneficial to turn on. The one
  untested variable that could change this conclusion is round budget
  scaled to `NUM_NICHES` rather than matched to vanilla's; unresolved.
- **Lazy niche scan (`LAZY_NICHE_SCAN=true`)**: a redesign that avoids the
  problem above by construction — the agent never knows this exists.
  Runs the exact vanilla loop untouched; only at Finalize, a scripted,
  non-agent pass (`steps/07_lazy_niche_scan.md`) checks whether any
  already-produced candidate (including rounds rejected as regressions on
  the canonical workload, still on disk, still correct) is secretly the
  best choice for some other real workload, and mechanically synthesizes a
  dispatcher — but only when the WORST regret across `--repeats` (default
  3) independent measurement passes is below a safety threshold (1.10x);
  otherwise it reports the finding and ships the canonical champion alone
  rather than risk an unquantified regression. The repeat-and-take-worst
  design is load-bearing, not defensive boilerplate: a single-pass version
  of this gate, run live on a real shared GPU box, flipped between
  "refuse" and "ship" on 2 of 5 identical sequential runs from ordinary
  contention noise alone — a safety gate a single noisy sample can fool
  isn't a safety gate. Also fixed after independent code review: the split
  search only ever tried one label orientation, which could make it blind
  to the correct routing rule entirely (see `docs/
  KA_KERNEL_OPT_LAZY_NICHE_SCAN.md` §3.3 for the full list — 5 real bugs
  found and fixed across two review passes, including this one, an
  incomplete-benchmark-matrix gap, and a hardcoded-axis-extraction gap
  now auto-verified before any dispatcher ships). Tested in offline replay
  against 3 real controlled runs: 2/3 found nothing to rescue (cost: ~96
  cheap benchmark calls per run, no change); 1/3 found a real one — **re-verified
  after the orientation fix, achieves geomean ~1.176x with a perfect,
  0-misclassification, 1.000x-worst-case-regret split**, matching the
  idealized ceiling exactly (the pre-fix search only found a flawed rule
  with 1.478x worst-case regression, understating what the mechanism can
  actually do). Also exercised as a live, end-to-end feature (full
  Setup→Rounds→Finalize on a real 4th problem, not just against
  already-completed runs' saved program databases) — the pipeline and
  dispatcher-codegen path are correct; that live run is also what
  surfaced the noise-robustness gap the repeat-and-take-worst fix above
  addresses, and separately found `pick_target_kernel()`'s
  cost-based-not-launch-order fix (see `tools/profile_ncu.py`) — a
  general multi-kernel-problem correctness fix, not niching-specific.
  The one form of direction 3 with a positive, live-validated result
  behind it — on by default.
- **Best-of-3 resampling (`BESTOF3=true`)**: a different lever from
  niching entirely — not selection between existing candidates, but
  generating 3 independent implementations of the *same* diagnosed fix
  each round instead of 1, verifying and benchmarking all 3, and keeping
  the best (`steps/08_bestof3.md`). Only the rewrite step is resampled;
  the diagnosis is one LLM call per round, same as vanilla. Tested on 4
  real problems, with real replicates on each (not single-run comparisons)
  after an initial mixed result turned out to be partly a harness bug (a
  missing divergence-reversion gate): 2 of 4 problems (GDN, SOL-187) show
  a real, replicated win (1.14x-1.56x); the other 2
  (`fp8_group_gemm`, MoE L2/008) show no reliable separation from vanilla
  once genuinely replicated, and MoE L2/008's replicate leaned the other
  way. The pattern isn't random: it helps when a round's diagnosis is
  likely right but its *implementation* is uncertain (multiple genuinely
  different ways to build the fix, real risk of a fragile or
  non-compiling "obvious" choice), and it doesn't help — and can waste
  rounds — when the problem needs a long *sequence* of distinct diagnoses,
  since resampling only multiplies attempts within one diagnosis, it
  can't discover the next one. See `steps/08_bestof3.md`'s "When this
  helps, honestly" for the full reasoning and a cheap, checkable-in-advance
  heuristic (naive kernel's launch/operation diversity). **Opt-in only —
  do not enable by default or recommend without checking that heuristic
  first.**

## Ground Rules
- NCU requires exclusive GPU access: never profile and benchmark at the same
  time.
- Never optimize past correctness: a candidate that fails its test is never
  accepted, no matter how fast.
- Report failures honestly — a round that regresses is recorded as a
  regression and informs the next reflexion; do not hide it.
- **No blind guessing**: before any diagnosis or fix that is not directly
  backed by profile evidence, read the "Checklist" sections in
  `${CLAUDE_SKILL_DIR}/reference/insights/` (learned methodology from past
  campaigns) and satisfy them — name the dominant stall with the metrics
  that prove it, locate the known-good fix before writing code, change
  exactly one thing per round, and confirm the targeted metric moved (not
  just wall-time).
