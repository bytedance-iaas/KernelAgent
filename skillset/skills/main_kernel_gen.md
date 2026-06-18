# Skill: Main Kernel Generation Orchestrator

## Purpose
Top-level orchestrator for end-to-end kernel generation from a PyTorch problem file.
Replaces `Fuser.auto_agent.AutoKernelRouter.solve()` and `Fuser.pipeline.run_pipeline()`.

## When to Use
When you need to generate a verified kernel from a KernelBench-style problem file
or any PyTorch module definition.

## Inputs
- `PROBLEM_PATH`: Absolute path to the problem `.py` file
- `TARGET_PLATFORM`: `cuda` (default) or `xpu`
- `KERNEL_BACKEND`: `triton` (default), `tilelang`, or `cutedsl`
- `MAX_ITERS`: Maximum refinement rounds per stage (default: 5)
- `VERIFY`: Whether to verify the final kernel (default: true)

## Workflow

### Step 1: Analyze and Route the Problem

Run the analysis tool:
```bash
python skillset/tools/analyze_problem.py --problem $PROBLEM_PATH
```

Parse the JSON output. The `route_recommendation` field will be one of:
- `kernelagent` — Simple problem, use the direct path
- `fuser` — Complex problem, use the full pipeline

Additionally, apply your own judgment by reading the problem file. Consider:
- **Route to `kernelagent`** if: simple/linear op chain (pointwise, pooling, small convs),
  no attention/conv_transpose, no complex normalization chains, no control flow, or a
  self-contained SDPA block (Q,K,V + softmax + matmul).
- **Route to `fuser`** if: conv_transpose2d present, group_norm with conv/pool chains,
  explicit control flow in forward, long chains (≥4 steps), or multi-branch/residual merges.
- **Route to `kernel_then_fuser`** if likely trivial but uncertain.
- **Route to `fuser_then_kernel`** if likely complex but some chance direct works.

### Step 2: Execute the Chosen Path

#### Path A: Direct KernelAgent (`kernelagent` or `kernel_then_fuser`)

Execute skill `06_direct_kernel.md` with the problem file and `KERNEL_BACKEND`.

If it fails and fallback is allowed, execute the full Fuser pipeline (Path B).

#### Path B: Full Fuser Pipeline (`fuser` or `fuser_then_kernel`)

Execute these skills in sequence:

1. **`01_fuse_model.md`** — Rewrite the PyTorch model into fusable subgraph modules.
   Input: problem file. Output: fused code file, run directory.

2. **`02_extract_subgraphs.md`** — Analyze the fused code and emit subgraphs JSON.
   Input: fused code + problem file. Output: `subgraphs.json`.

3. **For each subgraph** in `subgraphs.json`:
   - **`03_generate_kernel.md`** — Generate a kernel (pass `KERNEL_BACKEND`).
   - **`04_refine_kernel.md`** — Test and refine until PASS (pass `KERNEL_BACKEND`).

4. **`05_compose_kernels.md`** — Stitch all subgraph kernels into the final program.
   Input: problem file, subgraphs.json, kernel files. Output: `composed_kernel.py`.

If the full pipeline fails and fallback is allowed, try Path A.

### Step 3: Create Run Directory and Save Artifacts

Run:
```bash
python skillset/tools/manage_artifacts.py create-run-dir --base .fuse
```

Save all artifacts to the run directory using:
```bash
python skillset/tools/manage_artifacts.py write-artifact \
  --run-dir $RUN_DIR --name composed_kernel.py --content-file $KERNEL_PATH
```

### Step 4: Report Results

Report to the user:
- Route chosen and reason
- Whether generation succeeded
- Path to the generated kernel
- Number of refinement rounds
- Run directory path
