# Step: Analyze the Source Kernel

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-parser skill directory.

## Purpose
Extract everything the KernelBench problem file must encode from the source
kernel: the math it implements, the input/output tensor specs, the workload
shapes, and any constructor-level parameters.

## What to Extract (all source types)

Produce an analysis record before generating anything:

1. **Semantics** — the exact math as a composition of PyTorch-expressible
   ops (matmul, conv, norm, softmax, elementwise chain, reduction, ...).
   Include epilogues (bias add, activation, scaling) — fused kernels must be
   decomposed into their logical op sequence.
2. **Input specs** — order, shapes (symbolic M/K/N... and concrete workload
   values), dtypes, layouts (row/col-major, contiguity requirements).
3. **Output specs** — shape(s), dtype(s).
4. **Module parameters vs runtime inputs** — which tensors are weights/bias
   (belong in `Model.__init__` / `get_init_inputs`) vs per-call inputs
   (belong in `get_inputs`). Hyperparameters (kernel_size, stride, eps,
   groups...) go to `get_init_inputs`.
5. **Concrete workload shapes** — the shapes the kernel is actually tuned
   for. Priority order: an accompanying test/benchmark harness > launch-site
   callers > asserts in the wrapper > tile-size-implied minimums. If nothing
   pins them, choose realistic sizes and say so in the report.
6. **The kernel's existing unit tests** — find them; they are the accuracy
   contract the parsed Model must satisfy (step 03 ports them). Look for
   `test_*.py` / `*_test.py` next to the kernel, pytest suites referencing
   the kernel symbol, and `if __name__ == "__main__"` self-tests inside the
   kernel file. Record for each test: input construction (shapes, dtypes,
   seeds, distributions), the comparison target (a reference impl? golden
   values? properties?), and the EXACT tolerances (`rtol`/`atol`/equality).

**Searching large sources:** if the workspace is codegraph-indexed
(`.codegraph/` at the repo root), prefer `codegraph_explore` (MCP) or the
CLI `codegraph explore "<symbol or question>"` to find the kernel's callers,
tests, and related definitions; otherwise use grep.

## Per-Source-Type Guidance

### Triton
- Entry point: the Python wrapper (often `kernel_function` or a launcher
  calling `kernel[grid](...)`); the `@triton.jit` kernel holds the math.
- Read `tl.load`/`tl.store` indexing to recover logical shapes and layouts;
  `tl.dot` → matmul, `tl.sum/max` → reductions, elementwise `tl.*` chains.
- Shapes/dtypes: wrapper asserts, `torch.empty(...)` allocations,
  `@triton.autotune` key args, test files next to the kernel.
- `tl.constexpr` tile sizes are implementation detail — do NOT leak them
  into the problem file.

### PyTorch
- An `nn.Module` or function: semantics are already PyTorch — the job is
  mostly extraction of init args vs forward args and workload shapes.
- Custom autograd Functions / `torch.ops.*` calls: map to their documented
  math.

### CUTLASS / CUDA C++
- Identify the operation from template instantiations (Gemm, GemmGrouped,
  Conv2dFprop, ...), the element types (`cutlass::half_t`,
  `float_e4m3_t`, ...), layouts (RowMajor/ColumnMajor), and the epilogue
  (LinearCombination = alpha*AB + beta*C, activations, per-channel scale).
- Problem sizes come from the host-side launcher, test, or profiler config.
- FP8/quantized kernels: the PyTorch reference expresses the *dequantized
  math* (e.g. `(scale_a * A.float()) @ (scale_b * B.float())`), with a
  comment stating the original dtype; note that tolerance must be loose.
- The source is not runnable from Python without bindings — record that the
  equivalence check (step 03) will be skipped or needs an ad-hoc harness.

## Output
A short analysis summary (in your reply, and saved next to the output file
as `<name>.analysis.md` if the user wants artifacts): semantics, input/output
specs, init vs forward split, chosen workload shapes and their provenance,
and the target KernelBench level (1 = single op, 2 = fused op sequence,
3 = full architecture, 4 = pretrained HF model).
