# Step: Generate Kernel for a Subgraph

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-gen skill directory; the
> shared tool scripts live at `${CLAUDE_SKILL_DIR}/tools/`.

## Purpose
Generate a kernel implementation and test code for a single subgraph.
Replaces `TritonKernelAgent._generate_kernel_seeds()` and `._generate_test()`.

## When to Use
For each subgraph extracted in step 02, or when generating a kernel directly
from a problem description (the direct path of `ka-kernel-gen`).

## Inputs
- `SUBGRAPH_JSON`: Path to subgraphs.json + index, OR inline problem description
- `SUBGRAPH_DIR`: Directory to save kernel and test files
- `TARGET_PLATFORM`: `cuda` or `xpu` (default: `cuda`)
- `KERNEL_LANGUAGE`: `triton` (default), `tilelang`, or `cutedsl`

## Workflow

### Step 1: Get Problem Description

If working from a subgraph JSON item, run:
```bash
python "${CLAUDE_SKILL_DIR}/tools/build_reference.py" \
  --subgraph-file $SUBGRAPHS_JSON --index $INDEX --platform $TARGET_PLATFORM
```

This outputs JSON with `problem_description` and `reference_code`.

If working from a raw problem description (direct path), use that directly.

### Step 2: Get Language Guidelines

```bash
python "${CLAUDE_SKILL_DIR}/tools/render_template.py" \
  --template language_guidelines \
  --vars '{"kernel_language": "$KERNEL_LANGUAGE"}'
```

Read the output — these are the programming rules and code structure you must follow for the chosen language.

### Step 3: Generate Test Code

Following the test generation template structure, generate a test file that:

1. Imports `kernel_function` from `kernel` module: `from kernel import kernel_function`
2. Creates test data on the target device using EXACT shapes from the problem
3. Calls `kernel_function` as a normal Python function (no grid configuration)
4. Compares results against a PyTorch reference with tolerances (rtol=1e-3, atol=1e-3)
5. Prints detailed debug info on numerical mismatch
6. Returns True/False, exits with code 0/1

**Critical test rules:**
- Use the exact shapes and dtypes from the problem description
- Use non-zero random tensors (torch.randn/torch.rand) for inputs
- Do NOT pass precomputed reference outputs to kernel_function
- Do NOT use globals; keep all tensors local
- Device checks: use `result.device == input.device`, never compare to literal 'cuda'

Save to `$SUBGRAPH_DIR/test_kernel.py`.

### Step 4: Generate Kernel

Generate a complete Python file implementing the kernel. The exact structure depends on `KERNEL_LANGUAGE` — follow the guidelines read in Step 2 for the required imports, decorator patterns, and skeleton.

**UNIVERSAL REQUIREMENTS (all languages):**
- Expose a top-level `kernel_function(...)` that accepts PyTorch tensors/scalars and returns output tensor(s)
- `kernel_function` handles tensor allocation and kernel launch setup; all numerical work lives in language primitives
- DO NOT call PyTorch math ops (torch.add, torch.matmul, torch.mm, torch.bmm, etc.) for computation
- DO NOT use `torch.nn`, `torch.nn.functional`, aliases like `F.*`, or `torch.ops.aten.*`
- DO NOT subclass `nn.Module` or call `.forward()`
- DO NOT import `inspect`, use `sys._getframe`, `globals()`, or `locals()`

**LANGUAGE-SPECIFIC STRUCTURE:**
- **triton**: `@triton.jit` kernel + `kernel_function` wrapper; use `tl.load`/`tl.store`, `tl.program_id`, `tl.arange`, power-of-two BLOCK_SIZE with masking
- **tilelang**: `@tilelang.jit` builder returning a `@T.prim_func`; `kernel_function` compiles/launches via TileLang; no Triton or CuTe imports
- **cutedsl**: `@cute.kernel` function(s); `kernel_function` converts tensors via `from_dlpack`, sets grid/block, and launches; no Triton or TileLang imports

**FUSION PRIORITY:**
- Prefer a single fused kernel covering the entire operator pipeline
- Only fall back to unfused if fusion is provably impossible (add a comment explaining why)

Save to `$SUBGRAPH_DIR/kernel.py`.

## Output
- Path to `kernel.py`
- Path to `test_kernel.py`
