# Step: Fuse Model into Subgraph Modules

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-gen skill directory; the
> shared tool scripts live at `${CLAUDE_SKILL_DIR}/tools/`.

## Purpose
Rewrite a PyTorch model into fusable subgraph modules with explicit input/output shapes.
Replaces `Fuser.orchestrator.Orchestrator` + `Fuser.worker.Worker` + `Fuser.prompting`.

## When to Use
As the first step of the full Fuser pipeline path, when the router decides the problem
is too complex for direct KernelAgent.

## Inputs
- `PROBLEM_PATH`: Absolute path to the problem `.py` file
- `MAX_ITERS`: Maximum refinement rounds (default: 5)
- `TARGET_PLATFORM`: `cuda` or `xpu` (default: `cuda`)

## Workflow

### Step 1: Read the Problem File

Read the entire problem file content.

### Step 2: Generate Fused Code

Rewrite the model following these hard requirements:

**SYSTEM PROMPT:** Return a single runnable Python file only.

**FUSION GUIDANCE:**
- Rewrite the provided model into fusable subgraph modules with explicit input/output shapes.
- Each fused subgraph must be represented by its own `nn.Module` class with a clearly documented `forward`.
- Do not leave raw `nn.*` ops inline in the top-level Model.
- Detect scaled dot-product attention patterns and aggressively fuse the entire block (QKV linears, splits/reshapes, scaled QK^T, causal masking, ReLU or gating, applying V, and head merge) into a single attention subgraph whenever feasible.
- Only decompose attention into smaller subgraphs when you are certain fusion is impossible.

**HARD REQUIREMENTS:**
- Return ONE runnable Python file, fenced as a single ```python block.
- Include a function `run_tests()` that validates numerical equivalence to the original using helpers in the problem file. On success, `run_tests()` must print `PASS` and `exit(0)`.
- If you cannot implement `run_tests()`, then at minimum print the exact sentinel `ALL_TESTS_PASSED` and `exit(0)` when tests succeed.
- No network or file I/O outside the current directory. Avoid extra dependencies.
- Deterministic: set seeds where relevant.

**ITERATION CONTRACT:**
- On each attempt, re-emit the entire single-file solution.
- When ERROR_CONTEXT is provided, carefully analyze and fix issues, then re-emit the whole file.

### Step 3: Write and Verify

Write the generated code to a temporary file and run:
```bash
python "${CLAUDE_SKILL_DIR}/tools/run_candidate.py" --code-path $FUSED_CODE_PATH --timeout 120
```

### Step 4: Iterate on Failures

If verification fails, read the error output from the run result and refine the code.
Include the error context in your next attempt:

```
ERROR_CONTEXT:
<stderr from previous run>
```

Repeat up to `MAX_ITERS` times.

### Step 5: Save Artifacts

Once the fused code passes, save it using:
```bash
python "${CLAUDE_SKILL_DIR}/tools/manage_artifacts.py" write-artifact \
  --run-dir $RUN_DIR --name orchestrator/code.py --content-file $FUSED_CODE_PATH
```

## Output
- Path to the verified fused code file
- Run directory path
