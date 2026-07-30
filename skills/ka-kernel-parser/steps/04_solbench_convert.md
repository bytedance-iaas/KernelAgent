# Step: Convert a SOL-ExecBench Problem to KernelBench Format

> Deterministic fast path — no LLM analysis of the kernel is needed. The
> tool is `${CLAUDE_SKILL_DIR}/tools/problem_md.py` (symlink to
> `scripts/problem_md.py` at the repo root); the format spec is
> `docs/PROBLEM_MD_FORMAT.md`.

## When this step applies

`KERNEL_PATH` is:
- a **SOL-ExecBench problem directory** — contains `definition.json`,
  `reference.py` (or a `reference` field in the json), and
  `workload.jsonl` (e.g. `examples/SOL-ExecBench/data/benchmark/L1/...`), or
- an existing **unified `problem.md`** (already-converted single-file
  problem).

Steps 01/02 (analyze + generate) are skipped: the reference is already a
pure-PyTorch `run(...)` and the workloads are already pinned.

## Procedure

1. **Convert** (skip if the input is already a `problem.md`):

   ```bash
   python "${CLAUDE_SKILL_DIR}/tools/problem_md.py" from-solbench \
       <problem_dir> -o <OUTPUT_DIR>/problem.md
   ```

   This validates the md round-trip. It carries over `name`, `hf_id`,
   `custom_inputs_entrypoint`, axes/inputs/outputs, the full
   `reference.py` source (including any custom `get_inputs` factory),
   and every workload line verbatim.

2. **Materialize** the KernelBench pair:

   ```bash
   python "${CLAUDE_SKILL_DIR}/tools/problem_md.py" materialize \
       <OUTPUT_DIR>/problem.md -o <OUTPUT_DIR>
   ```

   Emits `problem.py` (KernelBench contract: `Model`, `get_inputs()`,
   `get_init_inputs()`, plus `WORKLOADS` / `build_workload_inputs(i)` /
   `workload_tolerance(i)` / `gpu_key()` / `workload_latency(i)`) and
   `test.py` (checks `kernel.kernel_function` over every workload with
   SOL-ExecBench's matched-ratio correctness semantics). When any
   workload carries a `latency` object (per-GPU `{baseline, target?,
   sol?, min_score?}` in ms — `target` is a hard gate, `sol` gates via
   the SOL-Score >= `min_score` (default 0.5 = match baseline), see
   `docs/PROBLEM_MD_FORMAT.md`), it also emits one pinned
   `<gpu>/perf_test.py` per GPU spec — the performance goal gates
   consumed by `ka-kernel-opt`.

3. **Validate** — the parser's usual three gates collapse to two here
   (the reference IS the source, so no cross-implementation equivalence
   run is needed):

   a. Contract/execution gate: `Model` + inputs must build and run —

      ```bash
      cd <OUTPUT_DIR> && python -c "
      import problem
      m = problem.Model().cuda()
      out = m(*problem.build_workload_inputs(0))
      print('ok', tuple(out.shape) if hasattr(out, 'shape') else type(out))"
      ```

   b. Harness self-check: the reference used AS the kernel must pass
      every workload —

      ```bash
      cd <OUTPUT_DIR> \
        && printf 'from problem import run as kernel_function\n' > kernel.py \
        && python test.py; rc=$?; rm kernel.py; (exit $rc)
      ```

      `PASS` (exit 0) is required. A failure means the conversion — not
      the problem — is broken; fix the md, re-materialize, retry (≤3
      attempts), else report the failure honestly.

4. **Report**: source dir → emitted paths, #inputs/#outputs/#workloads,
   canonical workload axes, self-check verdict.

## Supported features and known limitations

Supported (all verified by a full-corpus sweep, 214/235 problems pass
the self-check on H200): `custom_inputs_entrypoint` (factory rides
inside the Reference block, called `fn(axes_and_scalars, device)`),
`expr` axes, multi-output references, 0-d tensor shapes (`[]`),
`safetensors` workload inputs (blob root recorded in front matter at
conversion, override with env `SOLBENCH_BLOB_ROOT`; requires the
`safetensors` package), and SOL-ExecBench's name-based input heuristics
(norm weights/biases, causal + binary masks, rope cos/sin, positive
stats, SSM decay, softmax outputs, fan-in-scaled weights).

Expected skips — classify them, do not call them converter bugs:
- **nvfp4 problems** (`Quant/*nvfp4*`, 14 in the corpus): reference uses
  Blackwell-only `torch._scaled_mm` fp4 paths → `CUBLAS_STATUS_NOT_
  SUPPORTED` on sm_90. SOL-ExecBench marks these `@requires_sm100`.
- **Oversized workloads** (2 in the corpus): sized for 192 GB B200; OOM
  on smaller GPUs.
- **Reference-nan edge cases** (~3): some workloads (paged prefill with
  empty KV ranges, fully-masked MoE groups) produce nan from the
  reference itself under randomized inputs; the correctness gate
  rejects any nan by design. Report as SKIP(reference-nan).

## Batch mode

For a benchmark tree (`.../data/benchmark/L1` etc.), loop the procedure
per problem directory and finish with a summary table:
`source → problem.md → validate verdict (PASS / FAIL / SKIP+reason)`.
