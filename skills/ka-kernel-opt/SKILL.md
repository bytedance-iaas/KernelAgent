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
- **cuda_skill** — general CUDA optimization/debugging/profiling practice
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
  targets and exits 0 iff every target is met, 2 when run on a different
  GPU (emitted by ka-kernel-parser when the problem's workloads carry
  `latency` entries; its last-but-one stdout line is a JSON report with
  `measured_ms` / `baseline_ms` / `target_ms` per workload). Pick the
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

### Rounds 1..MAX_ROUNDS
Each round, in order (this mirrors
`OptimizationManager.run_optimization` → `select_candidates` → worker →
`update_with_results`):

1. **Select the parent kernel** from the program database:
   - greedy: `program_db.py best` — always continue from the global best,
     unless step 05's divergence rule already chose the continuation.
   - beam: `program_db.py top --k 2` — the two parents for this round.
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
6. **Verify + Accept/Reject + Reflect** —
   `${CLAUDE_SKILL_DIR}/steps/05_verify_accept.md`
   (correctness with ≤3 refinements, benchmark, program-database update with
   lineage, two-track best tracking, divergence revert, reflexion).
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
