# Skill: Compose End-to-End Kernel

## Purpose
Stitch verified subgraph kernels into a single end-to-end program that
replaces the original PyTorch model's forward pass.
Replaces `Fuser.compose_end_to_end.compose()`.

## When to Use
After all subgraph kernels have been generated and verified (steps 03-04),
as the final step of the Fuser pipeline.

## Inputs
- `PROBLEM_PATH`: Path to the original problem file
- `SUBGRAPHS_JSON`: Path to `subgraphs.json`
- `KERNEL_FILES`: List of verified kernel file paths (one per subgraph)
- `OUT_DIR`: Output directory for composed artifacts
- `TARGET_PLATFORM`: `cuda` or `xpu` (default: `cuda`)
- `KERNEL_LANGUAGE`: `triton` (default), `tilelang`, or `cutedsl`
- `MAX_ITERS`: Maximum composition refinement rounds (default: 5)
- `VERIFY`: Whether to verify the composed kernel (default: true)

## Workflow

### Step 1: Read All Inputs

Read:
- The original problem file (full content)
- The subgraphs JSON (list of subgraph specs)
- Each verified kernel file (full content)

### Step 2: Generate Composed Implementation

Generate ONE complete Python file that:

**HARD REQUIREMENTS:**
- Exposes a top-level wrapper named `kernel_function(...)`
- `kernel_function` accepts the same primary input tensor(s) as the original model plus any required weights/biases
- Orchestrates language kernel(s) to produce the final output tensor
- **NO PyTorch math path:** `kernel_function` MUST compute outputs using the chosen language's primitives only
- DO NOT use `torch.nn`, `torch.nn.functional`, or PyTorch math ops for producing the final result
- Using PyTorch for reference comparisons is allowed only inside the self-test
- Allocate all tensors on device=`{TARGET_PLATFORM}` and keep them there
- CPU is acceptable only for metadata and scalars

**LANGUAGE-SPECIFIC REQUIREMENTS:**
- **triton**: at least one `@triton.jit` kernel; imports `triton`, `triton.language as tl`; use `tl.load`/`tl.store` with masks; do NOT call `tl.broadcast` on scalars
- **tilelang**: at least one `@tilelang.jit` builder + `@T.prim_func`; imports `tilelang`, `tilelang.language as T`; no Triton or CuTe imports
- **cutedsl**: at least one `@cute.kernel` function; imports `cutlass`, `cutlass.cute as cute`, `from cutlass.cute.runtime import from_dlpack`; no Triton or TileLang imports

**SELF-TEST:**
- Include a `test_kernel()` or `run_tests()` function
- Compare language result to a PyTorch reference from the original problem code
- Use `get_init_inputs()` and `get_inputs()` if present in the problem
- Use `torch.allclose` with rtol≤1e-3, atol≤1e-3 for fp32; up to 2e-2 for fp16/bf16
- Print `PASS` on success and exit with code 0

**IMPLEMENTATION TIPS:**
- May inline, adapt, or reuse the given subgraph kernels
- Prefer fusing into as few kernel launches as possible
- Ensure intermediate tensor shapes match between subgraphs
- Hoist constant weights to avoid reloading per block
- Favor coalesced memory access; tile by blocks; compute grid from shape

**ALLOWED IMPORTS:**
- `torch`, language-specific packages (`triton`/`tilelang`/`cutlass`), and stdlib only
- No I/O beyond the current directory

### Step 3: Verify (if VERIFY is true)

Save the composed file and run:
```bash
python skillset/tools/run_candidate.py --code-path $COMPOSED_PATH --timeout 120
```

### Step 4: Refine on Failure

If verification fails, read the error output and generate a corrected version.

**Refinement guidance:**
- Fix language compilation/runtime errors using the language-specific pitfalls from `04_refine_kernel.md`
- Keep `kernel_function(...)` name unchanged
- Retain the self-test
- Do NOT reintroduce PyTorch math in `kernel_function`
- Return the complete corrected file (not diffs)

**Triton-specific auto-patches to apply before running:**
- Replace `tl.broadcast(0.0, ...)` → `0.0`
- Replace `tl.broadcast(1.0, ...)` → `1.0`

Repeat up to `MAX_ITERS` times.

### Step 5: Save Final Artifacts

Save to `$OUT_DIR/composed_kernel.py` and create a summary:
```bash
python skillset/tools/manage_artifacts.py write-artifact \
  --run-dir $RUN_DIR --name compose_out/composed_kernel.py --content-file $COMPOSED_PATH
```

## Output
```json
{
  "success": true|false,
  "composed_path": "/abs/path/to/composed_kernel.py",
  "rounds": <number of rounds>,
  "verify_passed": true|false
}
```
