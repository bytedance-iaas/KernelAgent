# Step: Diagnose the Bottleneck

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-opt skill directory.

## Purpose
You are the GPU performance expert: analyze the NCU metrics, roofline result,
grid analysis, the static PTX/SASS analysis, and the kernel code, then
produce a structured diagnosis. Replaces the LLM call in
`BottleneckAnalyzer.analyze` (the `BOTTLENECK_PROMPT` from
`kernel_perf_agent/kernel_opt/diagnose_prompt/judger_prompt.py`).

## Inputs
- `$RUN_DIR/ncu_round_$ROUND.json` (target kernel metrics, dynamic)
- `$RUN_DIR/ptx_round_$ROUND.json` (step 03: compiler-side static analysis —
  may be partial/absent; then diagnose on NCU data alone and say so)
- `$RUN_DIR/roofline_round_$ROUND.json` (roofline + grid analysis + config)
- `$RUN_DIR/gpu_specs.json`
- The current kernel code

## Before Hypothesizing: Read the Learned Checklists

`${CLAUDE_SKILL_DIR}/reference/insights/` holds hard-won methodology from
previous optimization campaigns. **Before proposing any bottleneck
hypothesis, read the "Checklist" section of every document in that folder**
(locate them with `grep -n -i "checklist" reference/insights/*.md`) and hold
your diagnosis to their evidence bar — e.g. the dominant stall reason must be
*named, with the 2–3 metric values that prove it*, not inferred from
intuition. If a checklist item you can act on now is unmet (missing ceiling
experiment, no per-line stall attribution), do that first rather than
guessing around it. If the folder is missing (skill used outside the
KernelAgent repo), proceed with the metric-grounding rules below.

## How to Diagnose

Classify the bottleneck into exactly one category:
- **memory** — memory bandwidth is the limiting factor
- **compute** — compute throughput is the limiting factor
- **underutilized** — neither saturated (both SOL < 60%): stalls / occupancy /
  launch-shape issues

The roofline tool already gives a mechanical classification (higher SOL wins);
your job is to confirm or overrule it with evidence and find the *root cause*.
Read the metrics in these groups (the labels below map to the NCU keys in the
metrics JSON):

**SM & Compute Utilization** — `sm__cycles_active.avg`, warp active %
(`sm__warps_active...`), instructions executed, tensor-core utilization and
pipeline activity.

**Memory Bandwidth & Cache** — DRAM throughput % of peak, DRAM bandwidth
(bytes/sec), DRAM bytes read/write, L1/L2 hit rates and throughput.

**Memory Access Patterns** — memory coalescing
(`smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct`; < 50% means
scattered access), branch uniformity.

**Launch Shape & Occupancy** — grid/block dims, blocks per SM,
`launch__occupancy_limit_{blocks,registers,shared_mem}` (which resource caps
occupancy), registers per thread, shared memory per block. Cross-check with
`grid_analysis` — a CRITICAL/WARNING assessment there (too few blocks, or
tens of sequential waves) usually IS the root cause.

**Stall Metrics** — short/long scoreboard stalls (long scoreboard = waiting on
global memory), barrier stalls, branch-resolving stalls.

**Static PTX/SASS cross-check** — read `ptx_round_$ROUND.json` next to the
NCU numbers; a diagnosis is stronger when a dynamic symptom has a static
cause (and suspicious when the two disagree):
- `flags[]` are pre-computed hypotheses (spills, launch-bounds register
  budget ceiling, narrow/unvectorized access, NaN-propagating min/max
  expansion, conversion-heavy mix, shuffle/barrier density) — each with the
  evidence numbers. Confirm or refute each HIGH-severity flag with an NCU
  metric before using it as a root cause.
- Cross-checks that matter:
  * spills flagged + long-scoreboard stalls high → spills ARE the memory
    dependency; fix registers before touching cache/coalescing.
  * `launch__registers_per_thread` (NCU, runtime) vs `ptx[].ptxas.registers`
    (offline): a mismatch means the runtime JIT used different options
    (e.g. launch-width register budget) — trust the runtime number.
  * `narrow_global_access` flagged + coalescing < 50% → vectorize loads;
    narrow but well-coalesced access is a weaker finding.
  * neither SOL saturated + no stall dominates + SASS histogram dominated
    by FSETP/FSEL, conversions, or shuffles → instruction/issue-bound:
    the fix is fewer instructions, not more overlap.

Typical signatures:
- High DRAM throughput (> 80%) + low compute SOL → **memory**-bound; look at
  coalescing, cache hit rates, redundant loads, datatype width.
- High compute SOL + tensor cores inactive on matmul-like work → **compute**;
  use tensor cores (fp16/bf16 `tl.dot`), increase arithmetic intensity.
- Both < 60% + high long-scoreboard stalls → **underutilized**; latency-bound:
  raise occupancy, overlap memory with compute (more warps, `num_stages`),
  fix launch shape.
- Both < 60% + `grid_analysis` CRITICAL → too few blocks; fix the grid first,
  nothing else matters until SMs have work.

## Output

Write the diagnosis to `$RUN_DIR/diagnosis_round_$ROUND.json` in this exact
schema (one object per bottleneck; produce 1 by default, 2 when running a
beam variant that explores two directions):

```json
[
  {
    "category": "memory" | "compute" | "underutilized",
    "summary": "One-line summary",
    "reasoning": "Explanation citing the metrics",
    "root_causes": [
      {
        "cause": "Description",
        "evidence": [{"metric": "name", "value": 0.0, "interpretation": "meaning"}],
        "fixes": [{"fix": "Actionable instruction", "rationale": "Why"}]
      }
    ]
  }
]
```

Requirements:
- Order bottlenecks by importance (most critical first).
- Each bottleneck: 2 root causes, each with 1 concrete fix (actionable at the
  code level — name the tile size, the load pattern, the API to use).
- Ground every claim in a metric value; no generic advice. Cite PTX/SASS
  evidence with the metric that corroborates it (e.g. `"evidence":
  [{"metric": "sass.fsetp_fsel_pairs", "value": 24, ...}, {"metric":
  "smsp__warp_issue_stalled_...", ...}]`).
- If `BOTTLENECK_OVERRIDE` was provided by the user, use that category and
  diagnose root causes within it.
