# Skill: Direct Kernel Generation (Bypass Fuser)

## Purpose
Generate a kernel directly from a problem description or KernelBench file
without going through the Fuser pipeline (extract → dispatch → compose).
Replaces `AutoKernelRouter._solve_with_kernelagent()`.

## When to Use
- When the router recommends `kernelagent` (simple/direct problems)
- When a user provides a plain language problem description
- As a fallback if the Fuser pipeline fails

## Inputs
- `PROBLEM_PATH` or `PROBLEM_DESCRIPTION`: Path to problem file OR text description
- `TARGET_PLATFORM`: `cuda` or `xpu` (default: `cuda`)
- `KERNEL_LANGUAGE`: `triton` (default), `tilelang`, or `cutedsl`
- `MAX_ROUNDS`: Maximum refinement rounds (default: 10)
- `TEST_CODE` (optional): User-provided test code

## Workflow

### Step 1: Prepare Problem Description

If `PROBLEM_PATH` is provided, read the file content as the problem description.
If `PROBLEM_DESCRIPTION` is provided directly, use that.

### Step 2: Create Working Directory

```bash
python skillset/tools/manage_artifacts.py create-run-dir --base .fuse
```

Use the returned `run_dir` as the working directory. Create a subdirectory for this
kernel session:
```
$RUN_DIR/direct_kernel/
  kernel.py        # Generated kernel
  test_kernel.py   # Generated test
```

### Step 3: Generate Test Code

Following the test generation guidelines from skill `03_generate_kernel.md`:

1. Create a test that imports from `kernel` module
2. Uses EXACT shapes and dtypes from the problem
3. Compares against PyTorch reference
4. Prints PASS/FAIL, exits 0/1

If `TEST_CODE` is provided by the user, use it as an additional test alongside
the generated one.

Save to `$WORKDIR/test_kernel.py`.

### Step 4: Generate Initial Kernel

Following the kernel generation guidelines from skill `03_generate_kernel.md` with `KERNEL_LANGUAGE`:

1. Read the language guidelines (via `render_template.py --template language_guidelines --vars '{"kernel_language": "$KERNEL_LANGUAGE"}'`)
2. Generate a complete kernel with the language-appropriate structure and a `kernel_function` wrapper
3. All computation must use the chosen language's primitives (no PyTorch math ops)
4. Attempt to fuse all operations into a single kernel when feasible

Save to `$WORKDIR/kernel.py`.

### Step 5: Verify and Refine

Execute skill `04_refine_kernel.md` with:
- `KERNEL_PATH`: `$WORKDIR/kernel.py`
- `TEST_PATH`: `$WORKDIR/test_kernel.py`
- `PROBLEM_DESCRIPTION`: The problem description
- `KERNEL_LANGUAGE`: `$KERNEL_LANGUAGE`
- `MAX_ROUNDS`: `$MAX_ROUNDS`
- `WORKDIR`: `$WORKDIR`

### Step 6: Report Result

If the kernel passes verification:
```json
{
  "route": "kernelagent",
  "success": true,
  "kernel_path": "/abs/path/to/kernel.py",
  "session_dir": "/abs/path/to/run_dir/direct_kernel",
  "rounds": <number of rounds>
}
```

If all rounds are exhausted without success:
```json
{
  "route": "kernelagent",
  "success": false,
  "message": "Failed to generate working kernel after N rounds",
  "session_dir": "/abs/path/to/run_dir/direct_kernel"
}
```

## Example Usage

For a simple problem like ReLU with `KERNEL_LANGUAGE=triton`:
```
Problem: Implement ReLU over a contiguous 1D tensor of length 1024
```

This should produce a kernel like:
```python
import triton
import triton.language as tl
import torch

@triton.jit
def _relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)

def kernel_function(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _relu_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    return output
```

For `KERNEL_LANGUAGE=tilelang` or `KERNEL_LANGUAGE=cutedsl`, the structure will differ —
refer to the language guidelines rendered in Step 4 for the exact skeleton to follow.
