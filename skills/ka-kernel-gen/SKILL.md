---
name: ka-kernel-gen
description: "KernelAgent kernel generation. Generate a verified GPU kernel from a PyTorch problem file or text description (auto-routes between direct and Fuser pipeline), analyze a problem for routing only, or iteratively refine a failing kernel. Supports triton, tilelang, and cutedsl backends."
allowed-tools:
  - Bash
---

# Skill: KernelAgent Kernel Generation

The user's input arguments are: `$ARGUMENTS`

> **Paths:** `${CLAUDE_SKILL_DIR}` is the directory containing this SKILL.md
> (resolve it with `echo "${CLAUDE_SKILL_DIR}"` in bash if needed). The shared
> tool scripts live at `${CLAUDE_SKILL_DIR}/tools/` and the pipeline
> step instructions in `${CLAUDE_SKILL_DIR}/steps/`.

## Purpose
Single entry point for end-to-end kernel generation from a PyTorch problem.
Replaces `Fuser.auto_agent.AutoKernelRouter.solve()`, `Fuser.pipeline.run_pipeline()`,
and `triton_kernel_agent.agent`.

## Mode Dispatch — interpret the arguments first

| Arguments look like | Mode |
|---|---|
| a problem `.py` path or a plain-text problem description | **generate** — full flow: analyze, route, generate, verify |
| `analyze <problem.py>`, or the user only wants complexity/routing analysis | **analyze** — routing decision only |
| `refine <workdir>`, or paths to an existing `kernel.py` + `test_kernel.py`, or the user wants a failing kernel fixed | **refine** — refinement loop only |

- **analyze** → read and follow `${CLAUDE_SKILL_DIR}/steps/00_route_problem.md`.
  Report the routing decision JSON and stop.
- **refine** → read and follow `${CLAUDE_SKILL_DIR}/steps/04_refine_kernel.md`.
  Report the refinement result and stop.
- **generate** → run the full workflow below.

## Inputs (generate mode)
- `PROBLEM_PATH` or `PROBLEM_DESCRIPTION`: Path to the problem `.py` file OR text description
- `TARGET_PLATFORM`: `cuda` (default) or `xpu`
- `KERNEL_LANGUAGE`: `triton` (default), `tilelang`, or `cutedsl`
- `MAX_ITERS`: Maximum refinement rounds per stage (default: 5; 10 for the direct path)
- `VERIFY`: Whether to verify the final kernel (default: true)
- `TEST_CODE` (optional): User-provided test code

## Workflow (generate mode)

### Step 1: Analyze and Route the Problem

Read and follow `${CLAUDE_SKILL_DIR}/steps/00_route_problem.md`. It runs the
static analysis tool and combines it with your own reading of the problem file
to pick one of four strategies:

- `kernelagent` → Path A (direct)
- `fuser` → Path B (full pipeline)
- `kernel_then_fuser` → Path A, fall back to Path B on failure
- `fuser_then_kernel` → Path B, fall back to Path A on failure

If the input is a plain-text problem description (no file), skip the analysis
and go straight to Path A.

### Step 2a — Path A: Direct Kernel Generation (bypass Fuser)

Best for simple/linear operations. Replaces `AutoKernelRouter._solve_with_kernelagent()`.

1. **Prepare the problem description.** If `PROBLEM_PATH` is provided, read the
   file content as the problem description; otherwise use `PROBLEM_DESCRIPTION`.

2. **Create a working directory:**
   ```bash
   python "${CLAUDE_SKILL_DIR}/tools/manage_artifacts.py" create-run-dir --base .fuse
   ```
   Use the returned `run_dir` and create a session subdirectory:
   ```
   $RUN_DIR/direct_kernel/
     kernel.py        # Generated kernel
     test_kernel.py   # Generated test
   ```

3. **Generate test code and kernel** following
   `${CLAUDE_SKILL_DIR}/steps/03_generate_kernel.md`, treating the whole
   problem as a single subgraph (use the problem description directly; skip
   the `build_reference.py` step). Save to `$WORKDIR/test_kernel.py` and
   `$WORKDIR/kernel.py`. If `TEST_CODE` is provided by the user, use it as an
   additional test alongside the generated one.

4. **Verify and refine** following `${CLAUDE_SKILL_DIR}/steps/04_refine_kernel.md`
   with `MAX_ROUNDS` = 10 (or the user's `MAX_ITERS`).

5. **Report** (see Step 4 below), with `"route": "kernelagent"` and
   `"session_dir"` set to `$RUN_DIR/direct_kernel`.

### Step 2b — Path B: Full Fuser Pipeline

Execute these steps in sequence (read each file for the detailed instructions):

1. **`${CLAUDE_SKILL_DIR}/steps/01_fuse_model.md`** — Rewrite the PyTorch model
   into fusable subgraph modules.
   Input: problem file. Output: fused code file, run directory.

2. **`${CLAUDE_SKILL_DIR}/steps/02_extract_subgraphs.md`** — Analyze the fused
   code and emit subgraphs JSON.
   Input: fused code + problem file. Output: `subgraphs.json`.

3. **For each subgraph** in `subgraphs.json`:
   - **`${CLAUDE_SKILL_DIR}/steps/03_generate_kernel.md`** — Generate a kernel
     (pass `KERNEL_LANGUAGE`).
   - **`${CLAUDE_SKILL_DIR}/steps/04_refine_kernel.md`** — Test and refine until
     PASS (pass `KERNEL_LANGUAGE`).

4. **`${CLAUDE_SKILL_DIR}/steps/05_compose_kernels.md`** — Stitch all subgraph
   kernels into the final program.
   Input: problem file, subgraphs.json, kernel files. Output: `composed_kernel.py`.

### Step 3: Fallback

If the chosen path fails and the strategy allows fallback (`kernel_then_fuser`,
`fuser_then_kernel`), try the other path before giving up.

### Step 4: Save Artifacts and Report

Save all artifacts to the run directory:
```bash
python "${CLAUDE_SKILL_DIR}/tools/manage_artifacts.py" write-artifact \
  --run-dir $RUN_DIR --name composed_kernel.py --content-file $KERNEL_PATH
```

Report to the user:
- Route chosen and reason
- Whether generation succeeded
- Path to the generated kernel
- Number of refinement rounds
- Run directory path

On success, the structured result is:
```json
{
  "route": "kernelagent|fuser",
  "success": true,
  "kernel_path": "/abs/path/to/kernel.py",
  "session_dir": "/abs/path/to/run_dir/...",
  "rounds": <number of rounds>
}
```

On failure:
```json
{
  "route": "kernelagent|fuser",
  "success": false,
  "message": "Failed to generate working kernel after N rounds",
  "session_dir": "/abs/path/to/run_dir/..."
}
```
