# Step: Rewrite the Kernel

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-opt skill directory. Real
> production kernel repos are linked per platform under
> `${CLAUDE_SKILL_DIR}/reference/<platform>/` (e.g. `reference/cuda/`).

## Purpose
Generate an optimized kernel variant that applies the diagnosed fix. Replaces
the LLM rewrite call driven by the `kernel_optimization.j2` template.

## Inputs
- The current kernel code and its diagnosis (`diagnosis_round_$ROUND.json`)
- Roofline + grid analysis, GPU specs, baselines
- Attempt history and reflexions from previous rounds (if any)
- `ERROR_FEEDBACK` from a failed previous attempt (if any)

## Step 1: Consult the Optimization Knowledge Sources

This replaces the embedding-based RAG retrieval. Pull from three sources,
matched to the diagnosis — consult only what the bottleneck calls for:

**1. KernelWiki skill** (`.claude/skills/KernelWiki/`) — authoritative for
Hopper (SM90/H100) and Blackwell (SM100/B200) techniques: TMA, tcgen05/TMEM,
warp specialization, persistent kernels, cluster launch, 2-SM cooperative,
grouped GEMM / MoE patterns — with concrete PR references from
CUTLASS/SGLang/vLLM/FlashInfer. Read its `SKILL.md` and follow its lookup
flow for the technique the diagnosis points at. Check
`gpu_specs.architecture` first: Hopper/Blackwell techniques don't apply to
Ampere/Ada.

**2. cuda** (`.claude/skills/cuda/`) — general CUDA kernel
optimization and development practice (memory coalescing, occupancy tuning,
shared-memory patterns, PTX-level inspection). Use for platform-generic
fixes and whenever the diagnosis is about classic access-pattern or
occupancy problems.

**3. Real production kernels** (`${CLAUDE_SKILL_DIR}/reference/cuda/`) —
linked kernel repos with battle-tested implementations:
- `flash-attention/` — attention kernels (tiling, softmax, warp specialization)
- `flashinfer/` — inference kernels (paged attention, sampling, GEMM)
- `sgl-DeepGEMM/` — FP8/grouped GEMM (TMA, persistent, SM90/SM100)

Search these for the concrete pattern you are about to apply and read the
real implementation before writing your version. **Prefer codegraph when the
workspace is indexed** (a `.codegraph/` directory exists at the repo root) —
it returns the verbatim symbol source plus callers/call paths, which beats
text matching in repos this large:

- MCP tool `codegraph_explore` (if exposed in the session), or equivalently
  the CLI: `codegraph explore "swizzle pattern in DeepGEMM GEMM kernels"`

Both accept a natural-language question or symbol/file names. If codegraph is
not available, fall back to grep (e.g.
`grep -rn "swizzle" reference/cuda/sgl-DeepGEMM --include=*.cuh -l`).

**4. Learned insights** (`${CLAUDE_SKILL_DIR}/reference/insights/`) —
methodology distilled from previous optimization campaigns. Re-check the
"Checklist" sections before writing code; in particular:
- *the known-good fix must be located first* (a KernelWiki page, or a
  reference-source file:line from the repos above) — if you cannot point at
  where this fix is proven to work, you are guessing;
- *exactly ONE change per round*, so the re-profile can attribute the effect.

If a source is not installed (skill missing from `.claude/skills/`, or the
`reference/cuda` submodules not initialized — `git submodule update --init`),
note it and proceed with the remaining sources and your own knowledge.

## Step 2: Plan the Change

State (briefly, before writing code):
1. Which fix from the diagnosis you are applying.
2. The expected effect on the bottleneck metric (e.g. "coalescing 32% → >80%").
3. What config values change (`BLOCK_*`, `num_warps`, `num_stages`, grid).

Honor the history:
- **Never repeat** a change listed in a previous attempt that failed
  verification or regressed performance (see `avoid_patterns` in reflexions).
- **Prefer** directions listed in `try_patterns`.
- If `ERROR_FEEDBACK` is set, fix that specific failure first.

## Step 3: Write the Optimized Kernel

Requirements (all languages):
1. Apply the recommended fix for the diagnosed bottleneck category.
2. Complete, valid Python file — a full replacement, not a diff.
3. The public entry point stays `kernel_function` with the SAME signature,
   input/output shapes, and dtypes (the existing test must still pass).
4. Keep the wrapper free of PyTorch compute primitives — all math stays in
   the kernel DSL.
5. Maintain numerical correctness (the test's tolerances, typically
   rtol/atol 1e-3 to 1e-4).
6. No testing code, benchmarks, or explanatory prose in the file.

Performance target: beat `CURRENT_BEST_MS` by ≥10% if possible; at minimum
produce a measurable improvement or a deliberate experiment from the
diagnosis.

Save to `$RUN_DIR/kernel_candidate.py` (do NOT overwrite `kernel.py` until it
verifies — step 05 promotes it).

## Output
- `$RUN_DIR/kernel_candidate.py`
- A one-line record of {category, fix applied, config changes} for the
  attempt log.
