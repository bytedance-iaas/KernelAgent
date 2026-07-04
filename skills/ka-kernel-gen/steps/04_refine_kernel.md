# Step: Iterative Kernel Refinement

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-gen skill directory; the
> shared tool scripts live at `${CLAUDE_SKILL_DIR}/tools/`.

## Purpose
Test a kernel implementation against its test suite and iteratively refine
it until it passes. Replaces `VerificationWorker.run()` from `triton_kernel_agent/worker.py`.

## When to Use
After kernel generation (step 03, on either the direct path or the Fuser
pipeline) has produced initial kernel and test files, to verify correctness
and iteratively fix any issues — or standalone in refine mode, on an existing
`kernel.py` + `test_kernel.py` pair.

## Inputs
- `KERNEL_PATH`: Path to `kernel.py`
- `TEST_PATH`: Path to `test_kernel.py`
- `PROBLEM_DESCRIPTION`: The original problem description (for refinement context)
- `KERNEL_LANGUAGE`: `triton` (default), `tilelang`, or `cutedsl`
- `MAX_ROUNDS`: Maximum refinement iterations (default: 10)
- `WORKDIR`: Working directory containing kernel.py and test_kernel.py

## Workflow

### Step 1: Initial Verification

First, scan the kernel code for **disallowed patterns** (apply to all languages):
- `import torch.nn` or `from torch import nn`
- `torch.nn.functional` or `F.*` calls
- `torch.conv*`, `torch.relu`, `torch.sigmoid`, `torch.tanh`, `torch.softmax`
- `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.einsum` (or tensor methods)
- `torch.ops.aten.*`
- `class ... (nn.Module)` or `.forward()` calls
- `import inspect`, `sys._getframe`, `globals()`, `locals()`

Also check for **language cross-contamination** — the kernel must only use its own language:
- If `KERNEL_LANGUAGE=tilelang`: must NOT import `triton` or `cutlass`
- If `KERNEL_LANGUAGE=cutedsl`: must NOT import `triton` or `tilelang`
- If `KERNEL_LANGUAGE=triton`: must NOT import `tilelang` or `cutlass`

If any pattern is found, treat it as a failure with the message:
`"Disallowed usage detected: <pattern description>"`

### Step 2: Run the Test

```bash
python "${CLAUDE_SKILL_DIR}/tools/run_candidate.py" --code-path $TEST_PATH --timeout 60
```

Parse the JSON output. If `"passed": true`, the kernel is verified — stop.

### Step 3: Analyze Failure and Refine

If the test fails, read the `stderr_tail` and `stdout_tail` from the result.

**Build refinement context:**
- Include the language guidelines (from
  `python "${CLAUDE_SKILL_DIR}/tools/render_template.py" --template language_guidelines --vars '{"kernel_language": "$KERNEL_LANGUAGE"}'`)
- Include the test code
- Include the current kernel code
- Include the error output
- Include history from previous attempts (if any)

**Generate a refined kernel following these rules:**

1. Analyze the error message and fix the specific issue
2. The implementation must be a complete, valid Python file
3. The main function must be named `kernel_function`
4. Follow all Triton programming guidelines
5. Learn from previous attempts if any
6. Focus on fixing the specific error while maintaining correctness
7. **Fusion priority:** Preserve or extend operation fusion; never intentionally unfuse unless documented
8. Keep the wrapper free of PyTorch compute primitives

**Common pitfalls by language:**
- **triton**: `tl.broadcast(0.0, ...)` → just use `0.0` directly; missing masks on boundary loads/stores; wrong grid dimension calculations; type mismatches between kernel parameters
- **tilelang**: forgetting `lru_cache` on the builder; wrong `T.Kernel` block/grid shapes; incorrect `T.prim_func` buffer declarations
- **cutedsl**: forgetting `.mark_layout_dynamic()` on CuTe tensors; wrong `grid`/`block` dimensions; `from_dlpack` called on non-contiguous tensors

Save the refined kernel to `$KERNEL_PATH` (overwriting the previous version).

### Step 4: Iterate

Repeat Steps 1-3, tracking a history of previous attempts (kernel code + error output).
Include up to 8 previous attempts as context for the LLM refinement.

Stop when:
- Test passes → return success
- `MAX_ROUNDS` reached → return failure with last error

## Output
```json
{
  "success": true|false,
  "kernel_path": "/path/to/kernel.py",
  "rounds": <number of rounds taken>,
  "final_error": "<error message if failed>"
}
```
