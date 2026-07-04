# Step: Profile with NCU and Run Roofline Analysis

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-opt skill directory; the
> tool scripts live at `${CLAUDE_SKILL_DIR}/tools/`.

## Purpose
Collect hardware counters for the current kernel and classify how far it is
from the machine's roofline. Replaces `KernelProfiler.profile_kernel` +
`RooflineAnalyzer.analyze` + `_compute_grid_analysis`.

## Inputs
- `KERNEL_PATH`: current kernel file (e.g. `$RUN_DIR/kernel.py`)
- `PROBLEM_PATH`: `$RUN_DIR/problem.py`
- `ROUND`: current round number (for artifact naming)

## Workflow

### Step 1: NCU Profile

```bash
python "${CLAUDE_SKILL_DIR}/tools/profile_ncu.py" \
  --kernel $KERNEL_PATH --problem $PROBLEM_PATH \
  --workdir $RUN_DIR --csv ncu_round_$ROUND.csv \
  --out $RUN_DIR/ncu_round_$ROUND.json
```

Notes:
- NCU needs exclusive GPU access — never run it concurrently with a benchmark.
- On `ERR_NVGPUCTRPERM` permission errors, retry with `--sudo` (or ask the
  user to set `KERNELAGENT_NCU_USE_SUDO=1` / enable counter permissions).
- If NCU fails twice, skip profiling for this round: fall back to a
  diagnosis from the kernel code + grid math alone, and note the missing
  metrics in the report.

The output JSON is keyed by kernel name; `target_kernel` names the first
non-PyTorch (`at::*`) kernel — that is the kernel under optimization.

**Baseline profiling (round 1 only):** also profile the PyTorch eager
reference once for comparison in the report and diagnosis (e.g. how close
the kernel's DRAM throughput is to what eager achieves):

```bash
python "${CLAUDE_SKILL_DIR}/tools/profile_ncu.py" \
  --target eager --problem $PROBLEM_PATH \
  --workdir $RUN_DIR --csv ncu_eager.csv --out $RUN_DIR/ncu_eager.json
```

### Deep profiling (escalation path)

The quick metric set above feeds the roofline and the round loop. Escalate to
a full Nsight Compute deep dive — following the **ncu-report-skill**
methodology (`.claude/skills/ncu-report-skill/SKILL.md`) — when:
- the quick metrics are inconclusive (bottleneck `unknown`, or two rounds of
  fixes moved the metrics but not the runtime),
- you need per-line stall attribution to locate the hot instruction, or
- the target is B200/sm_100, where several metric names differ from older
  GPUs (see that skill's `reference/08-b200-metric-names.md`) and its
  diagnosis playbook is the better guide.

Capture the full report in the same pass as the quick metrics:

```bash
python "${CLAUDE_SKILL_DIR}/tools/profile_ncu.py" \
  --kernel $KERNEL_PATH --problem $PROBLEM_PATH --workdir $RUN_DIR \
  --csv ncu_round_$ROUND.csv --out $RUN_DIR/ncu_round_$ROUND.json \
  --save-report profile_round_$ROUND
```

then analyze `profile_round_$ROUND.ncu-rep` with the ncu-report-skill's
`ncu_report` Python API patterns, six analysis dimensions, and diagnosis
playbook. If that skill is not installed, fall back to
`ncu --import <rep> --page details` and the quick metrics.

### Step 2: Roofline + Grid Analysis

```bash
python "${CLAUDE_SKILL_DIR}/tools/roofline.py" \
  --metrics $RUN_DIR/ncu_round_$ROUND.json \
  --gpu-specs $RUN_DIR/gpu_specs.json \
  --kernel $KERNEL_PATH --kernel-language $KERNEL_LANGUAGE \
  > $RUN_DIR/roofline_round_$ROUND.json
```

Read the output:
- `roofline.bottleneck`: `memory` | `compute` | `underutilized` | `unknown`
  (higher SOL = the saturated resource; both < 60% = underutilized)
- `roofline.efficiency_pct` and `at_roofline` (≥ 95% = essentially optimal)
- `grid_analysis.assessment` / `recommendation` — occupancy and wave-count
  heuristics; a `CRITICAL`/`WARNING` here often dominates everything else
- `kernel_config` — the kernel's current tile sizes / num_warps / num_stages

## Output
- `$RUN_DIR/ncu_round_$ROUND.json`, `$RUN_DIR/roofline_round_$ROUND.json`
- If `roofline.at_roofline` is true, the kernel is at ≥95% of peak — signal
  the main loop to stop with success.
