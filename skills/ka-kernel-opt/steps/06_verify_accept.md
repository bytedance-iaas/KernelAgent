# Step: Verify, Benchmark, Accept/Reject, Reflect

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-opt skill directory; the
> tool scripts live at `${CLAUDE_SKILL_DIR}/tools/`.

## Purpose
Close the loop on one optimization round. Replaces
`verify_with_refinement` + `benchmark_kernel` + `_update_kernels` +
`_generate_reflexion`.

## Step 1: Verify Correctness (with up to 3 refinements)

Promote the candidate and run the test from the run directory:

```bash
cp $RUN_DIR/kernel_candidate.py $RUN_DIR/kernel.py
python "${CLAUDE_SKILL_DIR}/tools/run_candidate.py" \
  --code-path $RUN_DIR/test.py --run-dir $RUN_DIR --timeout 120
```

If `EXTRA_TESTS` exist, run each of them too — all must pass.

- On failure, read `stderr_tail`/`stdout_tail`, fix the specific error in the
  candidate, and retry — at most **3 refinement attempts**. Refinements must
  preserve the optimization being tested (don't quietly revert to the old
  kernel to make the test pass).
- If still failing after 3 attempts: restore the previous kernel
  (`cp $RUN_DIR/kernel_round_$((ROUND-1)).py $RUN_DIR/kernel.py`), record the
  attempt as `passed_verification: false` with the error message, set
  `ERROR_FEEDBACK` for the next round, write the reflexion (Step 4), and end
  the round.

## Step 2: Benchmark and Re-Profile

```bash
cp $RUN_DIR/kernel.py $RUN_DIR/kernel_round_$ROUND.py
python "${CLAUDE_SKILL_DIR}/tools/benchmark.py" \
  --mode kernel --kernel $RUN_DIR/kernel.py --problem $RUN_DIR/problem.py \
  --warmup $WARMUP --repeat $REPEAT
```

Record `NEW_TIME_MS`. Then re-profile for SOL (step 02 with the new round
number) to get `NEW_SOL` (`roofline.efficiency_pct`) — profiling after
benchmarking, never concurrently.

Register the verified candidate in the program database with its lineage
(`PARENT_ID` is the program_id of the kernel this round started from —
`initial` for round 1):

```bash
python "${CLAUDE_SKILL_DIR}/tools/program_db.py" add \
  --db $RUN_DIR/program_db.json \
  --kernel-file $RUN_DIR/kernel_round_$ROUND.py \
  --time-ms $NEW_TIME_MS --sol-pct $NEW_SOL \
  --round $ROUND --parent-id $PARENT_ID \
  --category "$BOTTLENECK_CATEGORY" --fix "$FIX_ONE_LINER"
```

(Only verified candidates enter the database — failed ones are recorded in
`attempts.jsonl` only, matching `GreedyStrategy.update_with_results`.)

## Step 3: Accept / Reject (two-track)

Maintain two bests across the run:
- **best-runtime**: accept iff `NEW_TIME_MS < BEST_TIME_MS`
- **best-SOL**: accept iff `NEW_SOL > BEST_SOL_PCT`

Then choose the kernel the NEXT round starts from:
- If `(NEW_TIME_MS - BEST_TIME_MS) / BEST_TIME_MS * 100 > DIVERGENCE_THRESHOLD`
  (default 50%): the candidate diverged — next round restarts from the
  best-runtime kernel.
- Otherwise continue from the new kernel even if it is somewhat slower
  (exploration is allowed within the divergence budget).

Compute for the attempt record:
- `improvement_pct = (BEST_TIME_MS - NEW_TIME_MS) / BEST_TIME_MS * 100`
- `config_changes`: diff of `kernel_config` (from roofline.py) old → new,
  formatted `"64→128"`

Append the attempt to `$RUN_DIR/attempts.jsonl`:

```json
{"round": N, "bottleneck_category": "...", "root_cause": "...",
 "recommended_fix": "...", "config_changes": {"BLOCK_M": "64→128"},
 "time_before_ms": 0.0, "time_after_ms": 0.0, "improvement_pct": 0.0,
 "is_improvement": true, "compute_sol_pct": 0.0, "memory_sol_pct": 0.0,
 "combined_sol_pct": 0.0, "passed_verification": true, "error_message": ""}
```

## Step 4: Reflexion

Write an honest self-assessment to `$RUN_DIR/reflexion_round_$ROUND.json`.
For failed attempts this is mechanical (diagnosis untested, fix ineffective,
lesson = fix the error class); for completed attempts, reason about it:

```json
{"round": N,
 "was_diagnosis_correct": true, "was_fix_effective": false,
 "expected_outcome": "what you predicted in step 04",
 "actual_outcome": "what the numbers showed",
 "reasoning": "why the fix worked or didn't",
 "lessons": ["..."],
 "avoid_patterns": ["patterns that failed — do not retry"],
 "try_patterns": ["promising directions for the next round"]}
```

Judge `was_diagnosis_correct` by whether the targeted metric moved as
predicted (re-profile data), independently of whether wall-clock improved.

## Output
- Updated `BEST_TIME_MS` / `BEST_KERNEL` / `BEST_SOL_PCT` and next-round
  starting kernel
- `attempts.jsonl` and `reflexion_round_$ROUND.json` entries
