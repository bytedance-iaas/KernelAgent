---
name: ka-kernel-parser
description: "Parse an existing GPU kernel (Triton, PyTorch, or CUTLASS/CUDA) into KernelBench problem format: a pure-PyTorch nn.Module reference (Model) with get_init_inputs()/get_inputs(), validated and numerically checked against the source kernel. Also converts SOL-ExecBench problems (definition.json + reference.py + workload.jsonl) or unified problem.md files into KernelBench format deterministically. Use to onboard existing kernels into the ka-kernel-gen / ka-kernel-opt pipelines."
allowed-tools:
  - Bash
---

# Skill: KernelAgent Kernel → KernelBench Problem Parser

The user's input arguments are: `$ARGUMENTS`

> **Paths:** `${CLAUDE_SKILL_DIR}` is the directory containing this SKILL.md
> (resolve it with `echo "${CLAUDE_SKILL_DIR}"` in bash if needed). Tool
> scripts live at `${CLAUDE_SKILL_DIR}/tools/`, step instructions in
> `${CLAUDE_SKILL_DIR}/steps/`.

## Purpose
Convert an existing kernel implementation into the KernelBench problem
format: a **pure-PyTorch** `Model(nn.Module)` whose `forward` is the
reference implementation of the kernel's math, plus `get_init_inputs()`
(constructor args) and `get_inputs()` (forward args with concrete workload
shapes). Canonical format examples live in
`examples/KernelBench/KernelBench/level{1..4}/` in the KernelAgent repo.

The output is the input format of the sibling skills: feed it to
`ka-kernel-gen` to regenerate the kernel in another DSL, or use it as
`problem.py` in a `ka-kernel-opt` run.

## Mode Dispatch — inspect `KERNEL_PATH` first

| `KERNEL_PATH` looks like | Mode |
|---|---|
| a SOL-ExecBench problem dir (`definition.json` + `workload.jsonl`) or a unified `problem.md` | **solbench convert** — deterministic; read and follow `${CLAUDE_SKILL_DIR}/steps/04_solbench_convert.md`, skipping steps 01–02 |
| anything else (Triton/PyTorch/CUDA source, repo+symbol) | **parse** — full workflow below (steps 01–03) |

## Inputs
- `KERNEL_PATH`: the source kernel — a Triton `.py` (with `@triton.jit` +
  wrapper), a PyTorch module/function file, or a CUTLASS/CUDA `.cu`/`.cuh`
  source. May also be a repo path + symbol name for kernels inside a large
  codebase — or a SOL-ExecBench problem directory / unified `problem.md`
  (see Mode Dispatch above).
- `OUTPUT_PATH` (optional): where to write the problem file (default: next
  to the source, named per the KernelBench convention). If it ends in
  `.md`, emit the unified single-file problem format instead
  (`docs/PROBLEM_MD_FORMAT.md`: axes/inputs/outputs tables, the PyTorch
  reference as a `run(...)` code block, workloads as a jsonl block) and
  validate it with `python <repo>/scripts/problem_md.py check <out.md>`
  followed by `materialize` + running the emitted `test.py`.
- `SHAPES` (optional): user-pinned workload shapes/dtypes; otherwise
  extracted from tests/callers/asserts in step 01.
- `LEVEL` (optional): target KernelBench level hint (1–4).

## Workflow

Execute the steps in order (read each file for detailed instructions):

1. **Analyze** — `${CLAUDE_SKILL_DIR}/steps/01_analyze_kernel.md`
   Identify the source type (triton / pytorch / cutlass) and extract:
   semantics as PyTorch-expressible ops, input/output specs, the
   init-vs-forward split, concrete workload shapes with provenance, and the
   kernel's **existing unit tests** (its accuracy contract).
   Prefer codegraph over grep for searching large repos when indexed.

2. **Generate** — `${CLAUDE_SKILL_DIR}/steps/02_generate_problem.md`
   Write the problem file following the exact format contract (pure PyTorch
   Model, module-level shape constants, CPU tensors in `get_inputs`, level
   classification, naming convention).

3. **Validate** — `${CLAUDE_SKILL_DIR}/steps/03_validate.md`
   Three gates: contract + execution check (`validate_problem.py`, ≤3 fix
   attempts); numerical equivalence against the source kernel
   (`check_equivalence.py`) whenever the source is Python-callable; and
   **ported unit tests** — the kernel's own test cases duplicated against
   the Model with identical inputs, seeds, and tolerances, saved as
   `test_<problem_name>.py`. The parse is not done until all gates pass or
   the report states exactly why one was skipped. The Model does not need
   the kernel's performance — it must match the kernel's accuracy.

## Batch Mode
For multiple kernels (a directory or list), run the three steps per kernel
independently and finish with a summary table: source → output file, level,
validation verdict, equivalence verdict. For a SOL-ExecBench benchmark
tree (e.g. `examples/SOL-ExecBench/data/benchmark/L1`), loop the solbench
convert step per problem directory instead (see the batch section of
`steps/04_solbench_convert.md`).

## Ground Rules
- The Model is a *reference*, not a port: express the logical math in
  idiomatic PyTorch; never copy scheduling details (tile sizes, swizzling,
  pipelining) into the problem file. Performance does not matter —
  accuracy parity with the source kernel does.
- Never leave a generated problem file unvalidated; never widen tolerances
  to force a pass — the tolerances come from the kernel's own unit tests,
  and a mismatch means the reference is wrong.
- Weights and hyperparameters go through `__init__`/`get_init_inputs()`;
  only per-call tensors go through `get_inputs()`.
- Ship the ported unit tests alongside the problem file — they are the
  `test.py` for downstream `ka-kernel-opt` runs.
