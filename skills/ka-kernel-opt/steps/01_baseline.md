# Step: Verify and Establish Baselines

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-opt skill directory; the
> tool scripts live at `${CLAUDE_SKILL_DIR}/tools/`.

## Purpose
Before optimizing, confirm the initial kernel is correct and measure the
reference points every later round is judged against. Replaces
`OptimizationManager`'s initial verification + baseline benchmarks.

## Inputs
- `KERNEL_DIR`: directory containing `input.py` (initial kernel), `problem.py`, `test.py`
- `RUN_DIR`: artifacts directory for this optimization run
- `WARMUP` / `REPEAT`: benchmark iterations (defaults 25 / 100)

## Workflow

### Step 1: Set Up the Working Copy

```bash
mkdir -p $RUN_DIR
cp $KERNEL_DIR/input.py $RUN_DIR/kernel.py
cp $KERNEL_DIR/problem.py $KERNEL_DIR/test.py $RUN_DIR/
cp $RUN_DIR/kernel.py $RUN_DIR/kernel_round_0.py
```

All later work happens on `$RUN_DIR/kernel.py`; `test.py` imports
`from kernel import kernel_function` and `from problem import ...`, so tests
must run with `$RUN_DIR` as the working directory.

### Step 2: Verify Initial Correctness

```bash
python "${CLAUDE_SKILL_DIR}/tools/run_candidate.py" \
  --code-path $RUN_DIR/test.py --run-dir $RUN_DIR --timeout 120
```

If `EXTRA_TESTS` were provided, copy them into `$RUN_DIR` and run each the
same way — the primary test plus every extra must pass.

If the initial kernel fails any test, **stop the whole run** and report
`"Initial kernel failed correctness verification"` — never optimize a broken
kernel.

### Step 3: Benchmark the Three Baselines

Run each in its own invocation (a crash cannot take the session down):

```bash
python "${CLAUDE_SKILL_DIR}/tools/benchmark.py" \
  --mode eager --problem $RUN_DIR/problem.py --warmup $WARMUP --repeat $REPEAT

python "${CLAUDE_SKILL_DIR}/tools/benchmark.py" \
  --mode compile --problem $RUN_DIR/problem.py --warmup $WARMUP --repeat $REPEAT

python "${CLAUDE_SKILL_DIR}/tools/benchmark.py" \
  --mode kernel --kernel $RUN_DIR/kernel.py --problem $RUN_DIR/problem.py \
  --warmup $WARMUP --repeat $REPEAT
```

Record from each JSON output:
- `PYTORCH_BASELINE_MS` (eager)
- `PYTORCH_COMPILE_MS` (compile; may fail — treat as unavailable, not fatal)
- `INITIAL_KERNEL_MS` — this is the starting `BEST_TIME_MS`

Save all three results to `$RUN_DIR/baselines.json` and report a summary table
with speedups (`eager / t`).

### Step 4: Resolve GPU Specs

```bash
python "${CLAUDE_SKILL_DIR}/tools/gpu_specs.py" --detect
# or, if the user named the GPU:
python "${CLAUDE_SKILL_DIR}/tools/gpu_specs.py" --name "$GPU_NAME"
```

Save the specs to `$RUN_DIR/gpu_specs.json`. If the GPU is unknown to the
database, tell the user which GPUs are supported and continue without grid
analysis (roofline still works — it only needs NCU SOL percentages).

## Output
- `BEST_TIME_MS` = `INITIAL_KERNEL_MS`, `BEST_KERNEL` = `kernel_round_0.py`
- `$RUN_DIR/baselines.json`, `$RUN_DIR/gpu_specs.json`
