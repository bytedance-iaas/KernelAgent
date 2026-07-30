# Step: Validate the Problem File

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-parser skill directory;
> the tool scripts live at `${CLAUDE_SKILL_DIR}/tools/`.

## Step 1: Format + Execution Check

```bash
python "${CLAUDE_SKILL_DIR}/tools/validate_problem.py" --problem $OUTPUT_PATH
```

Checks the contract (Model / get_inputs / get_init_inputs, nn.Module, pure
PyTorch, forward runs, returns tensors) and reports input/output specs.
On failure, fix the problem file and re-run — up to 3 attempts. Compare the
reported `input_specs`/`output_specs` against the step-01 analysis; a shape
or dtype mismatch here means the problem file, not the validator, is wrong.

Note: this executes one full forward pass on the workload shapes (CPU if no
GPU). For very large workloads (level 3/4) expect it to be slow but run it
anyway — a problem file that cannot execute is not a problem file.

## Step 2: Numerical Equivalence vs the Source Kernel

Skip only when the source cannot be called from Python (CUTLASS C++ without
bindings) or no GPU is available for a GPU-only DSL — and say so explicitly
in the report.

For a source kernel exposing a plain callable (the `kernel_function`
convention, or any function taking the forward tensors):

```bash
python "${CLAUDE_SKILL_DIR}/tools/check_equivalence.py" \
  --problem $OUTPUT_PATH --kernel $KERNEL_PATH --entry kernel_function
  # add --with-model-params if the kernel takes weights/bias as trailing args
```

Tolerances default by dtype (fp32 1e-4, fp16 1e-2, bf16 2e-2); loosen
explicitly (`--rtol/--atol`) for FP8-style dequantized references and note
it in the report.

If the source entry is not a plain function (a class, a bound method, a
launcher needing extra config), write a small ad-hoc harness in the run
directory that calls both sides on the same seeded inputs and prints
max-abs/rel error, then run it via
`python "${CLAUDE_SKILL_DIR}/tools/run_candidate.py" --code-path harness.py`.

**On mismatch**: the reference (problem file) is wrong until proven
otherwise — re-check operand order, layouts (transposed weights are the #1
culprit), accumulation dtype, and epilogue order. Do not silently widen
tolerances to make it pass.

## Step 3: Port the Source Kernel's Unit Tests (accuracy contract)

The random-input equivalence check above is necessary but not sufficient —
the kernel's own unit tests carry the edge cases and the authoritative
tolerances. Duplicate them against the Model (tests were located in
step 01; if none exist, say so in the report and stop after Step 2).

Write `test_<problem_name>.py` next to the problem file, porting **every
test case** with these rules:

- **Preserve exactly**: input construction (shapes, dtypes, seeds,
  distributions, special values like zeros/NaN/inf/masks), parametrized
  case lists, and the original tolerances. If the original asserts exact
  equality (integer outputs, bitwise checks), keep exact equality.
- **Replace the subject**: where the original test calls the kernel, the
  ported test calls `Model` (imported from the problem file — instantiate
  per test case with the case's hyperparameters, falling back to
  `get_init_inputs()`).
- **Keep the original comparison target**:
  - Original compares kernel vs a reference implementation → port as
    `Model` vs that same reference, same tolerance.
  - Original checks golden values or properties (shape/dtype/finiteness/
    monotonicity) → port those assertions directly against `Model`.
- **When the kernel is Python-callable, add the direct check**: for each
  test case's inputs, also assert `Model(...) ≈ kernel(...)` with the
  original test's tolerance — this is the "same accuracy as the existing
  kernel" guarantee, on the kernel's own test vectors.
- The ported file must be standalone (no pytest requirement): run all
  cases, print per-case results, print `PASS` and exit 0 only if every
  case passes.

Run it:

```bash
python "${CLAUDE_SKILL_DIR}/tools/run_candidate.py" \
  --code-path test_<problem_name>.py --run-dir <problem dir> --timeout 300
```

All ported cases must pass (fix the problem file, not the tolerances). GPU-
only cases that cannot run in the current environment are marked SKIPPED
with the reason printed — never silently dropped.

Keep the ported test file: it doubles as the `test.py` required by the
`ka-kernel-opt` input contract, so downstream optimization runs verify
against the same accuracy bar as the original kernel.

## Output
Report:
- Validation verdict + input/output specs
- Equivalence verdict (max abs/rel error vs tolerance) or the explicit
  reason it was skipped
- Ported unit tests: source test file(s), number of cases ported /
  passed / skipped (with reasons), path to `test_<problem_name>.py`
- Final problem file path and its KernelBench level
