# Step: Static Analysis (PTX + SASS)

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-opt skill directory; the
> tool scripts live at `${CLAUDE_SKILL_DIR}/tools/`.

## Purpose
Look at the kernel the way the *compiler* left it, not just the way the
hardware counters saw it run. NCU tells you *what* is slow (stalls,
throughput); the static view tells you *what was actually compiled* —
register spills, launch-bounds register budgets, narrow (unvectorized)
loads, NaN-propagating min/max expansions, conversion-heavy inner loops,
convergent-op density. The diagnose step reads BOTH so no conclusion rests
on a single tool.

**This step is MANDATORY and runs every round.** It is never skipped: a
round must not reach the diagnose step without a static view of the
compiled kernel. If the tool degrades, fall back manually (see Failure
handling) — "proceed on NCU data alone" is not an option.

The step has two halves, both required:
- **PTX analysis** — the compiler-frontend view (launch directives,
  declared locals, access widths, `ptxas --verbose` resource usage).
- **SASS analysis** — the final machine-code view (cubin resource usage,
  opcode histogram, spill ops, load/store width mix).

## Inputs
- `KERNEL_PATH`, `PROBLEM_PATH`, `ROUND` (same as step 02)
- `KERNEL_LANGUAGE`: `triton` | `tilelang` | `cutedsl`

## Workflow

```bash
python "${CLAUDE_SKILL_DIR}/tools/analyze_ptx.py" \
  --kernel $KERNEL_PATH --problem $PROBLEM_PATH \
  --workdir $RUN_DIR --kernel-language $KERNEL_LANGUAGE \
  --dump-dir $RUN_DIR/static_dump_round_$ROUND \
  --out $RUN_DIR/static_round_$ROUND.json
```

What the tool does (one kernel run + offline analysis, no NCU — safe to run
right after profiling):
1. Runs the kernel once with the DSL's dump hooks (`TRITON_CACHE_DIR`,
   `CUTE_DSL_KEEP_PTX/CUBIN` + `CUTE_DSL_DUMP_DIR`, `TILELANG_CACHE_DIR`)
   to collect the JIT-generated `.ptx`/`.cubin` artifacts.
2. **PTX analysis pass**: launch directives (`.reqntid`/`.maxntid`/
   `.maxnreg` → the implied per-thread register budget `65536/threads`),
   `.local` declarations, global access widths; `ptxas --verbose` for
   registers / spill bytes / barriers / performance warnings.
3. **SASS analysis pass** (`cuobjdump`): per-function resource usage
   (REG/STACK/LOCAL), opcode histogram, spill ops (LDL/STL), global &
   shared load/store width mix, FSETP+FSEL pair count, SHFL/BAR density.
4. Emits `flags`: deduplicated inefficiency findings, each with evidence
   and a code-level hint, ordered by severity.

## Reading the output (`static_round_$ROUND.json`)

- `flags[]` — the headline. Every flag carries `evidence` (the numbers) and
  a `hint` (the standard fix). Treat these as *hypotheses to cross-check
  against the NCU metrics in step 04*, not as verdicts:
  - `register_spills` / `local_memory_ops` — confirm with NCU local-memory
    traffic and long-scoreboard stalls before acting.
  - `reg_budget_ceiling` — wide CTAs (>8 warps) cap ptxas at
    `65536/threads` registers; confirm with `launch__registers_per_thread`
    from NCU (the *runtime* allocation can differ from offline ptxas!).
  - `narrow_global_access` — confirm with NCU coalescing
    (`smsp__sass_average_data_bytes_per_sector...`): narrow but coalesced
    loads may still be fine.
  - `nan_propagating_minmax` / `conversion_heavy` / `shuffle_heavy` —
    instruction-mix costs that NCU shows only indirectly (issue-bound
    kernels with no saturated pipe).
- `ptx[].ptxas` vs `sass[].resource_usage` — if the offline ptxas register
  count differs from the cubin's (or from NCU's
  `launch__registers_per_thread`), the runtime JIT compiled with different
  options than the offline check: trust the cubin/NCU numbers.
- `sass[].sass_top_opcodes` — sanity-check the kernel's character: a
  "memory-bound" kernel whose SASS is dominated by conversions or
  FSETP/FSEL is really instruction-bound.

## Failure handling — degrade, never skip

The tool always exits 0 and records what it could not collect (missing
`ptxas`/`cuobjdump`, no artifacts dumped — flag `no_artifacts`). That is
NOT permission to skip the step. When part of the output is missing,
collect it manually before moving to step 04:

1. **No artifacts dumped** — re-run the kernel once yourself with the DSL's
   dump environment set (`TRITON_CACHE_DIR=<dir>` for Triton;
   `CUTE_DSL_KEEP_PTX=1 CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=<dir>` for
   CuTe DSL; `TILELANG_CACHE_DIR=<dir>` for TileLang) and locate the
   `.ptx`/`.cubin` files in that directory.
2. **PTX pass missing** — run `ptxas --verbose -arch=<sm> <file>.ptx -o
   /dev/null` on the dumped PTX yourself and read the register/spill/
   barrier report; grep the PTX for `.local`, launch directives, and
   access widths.
3. **SASS pass missing** — run `cuobjdump -sass <file>.cubin` (or
   `nvdisasm`) on the dumped cubin and derive the opcode histogram, LDL/STL
   spill count, and load/store width mix (a `grep -oE '^\s+\S+' | sort |
   uniq -c` over the SASS mnemonics is enough).
4. Only if the binaries genuinely cannot be produced on the machine
   (e.g. no CUDA toolkit binaries at all) do you proceed — and then the
   round report MUST state exactly which half (PTX analysis, SASS analysis,
   or both) is missing, why, and that the diagnosis is correspondingly
   weaker. This is the degraded exception, not a skip: the step still ran,
   its result is recorded, and the gap is declared.

For deeper instruction-level digging (per-line SASS ↔ source correlation,
`nvdisasm` control-flow graphs), follow the **cuda** skill's
`references/debugging-tools.md` (cuobjdump/nvdisasm sections).

## Output
- `$RUN_DIR/static_round_$ROUND.json` (+ raw artifacts under
  `$RUN_DIR/static_dump_round_$ROUND/`), including any manually collected
  results from the fallback path
