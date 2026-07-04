# Step: Diagnose the Bottleneck

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-opt skill directory.

## Purpose
You are the GPU performance expert: analyze the NCU metrics, roofline result,
grid analysis, and kernel code, then produce a structured diagnosis. Replaces
the LLM call in `BottleneckAnalyzer.analyze` (the `BOTTLENECK_PROMPT` from
`kernel_perf_agent/kernel_opt/diagnose_prompt/judger_prompt.py`).

## Inputs
- `$RUN_DIR/ncu_round_$ROUND.json` (target kernel metrics)
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
- Ground every claim in a metric value; no generic advice.
- If `BOTTLENECK_OVERRIDE` was provided by the user, use that category and
  diagnose root causes within it.
