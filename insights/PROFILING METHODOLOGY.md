
# Beating flashinfer without blind guessing — a profiling-first methodology

A retrospective on how the fp8 groupwise grouped-GEMM (`tc_persist_gemm.py`) went from
**650 TF/s (3× under flashinfer) to 1140–2160 TF/s (beating it)** — and, more importantly,
*why the winning changes were found by measurement and the losing ones by guessing.*

The single lesson: **every hour spent guessing cost more than the ten seconds a profile
would have taken.** This document is the loop we *should* run from the start.

---

## 0. The goal, stated as a measurable

> Beat `flashinfer.gemm.group_gemm_fp8_nt_groupwise` on GB200 for fp8 (1,128,128)
> groupwise scaling, measured CUDA-graph-fair (host overhead removed), at a fixed set of
> representative shapes, while staying correct (rel-L2 < 5e-3 vs a torch reference).

Two shapes were the yardstick, always benchmarked *together* (small and large behave
differently): `m=512 n=512 k=4096 g=8` and `m=4096 n=8192 k=8192 g=8`. Never optimize
against a single shape or a single number.

---

## 1. The anti-pattern that burned the most time

Blind change → re-run → "still slow" → another blind change. In this session that produced
a string of **confident, plausible, wrong** hypotheses:

| Guess (no data behind it) | Predicted | Actual |
|---|---|---|
| "Release the tmem bank earlier so the MMA isn't stalled" | faster | **worse** (78→64 TF) |
| "Double-buffer the a_scale smem stage" | faster | no change |
| "Preload the whole a_scale block once per tile" | faster | 2365 → 1424 (helped) but not the real fix |
| "Fix the sSA layout stride to kill bank conflicts" | faster | **zero change** |
| "Vectorize / decouple the gather from the FMA" | faster | zero change |

Each guess *sounded* right. Every one was refuted by the next run. The reason they were
wrong is that none of them were derived from **what the hardware was actually doing** — they
were derived from a mental model of what it *might* be doing. The mental model was wrong in a
way one profile would have exposed immediately.

**Rule:** if you cannot name the two or three metric values that justify a change, you are
guessing. Stop and profile.

---

## 2. The loop that actually works

```
        ┌─────────────────────────────────────────────────────────┐
        │ 1. MEASURE a fair baseline (CUDA-graph, real shapes)      │
        │ 2. PROFILE the slow kernel with ncu (--set full + source) │
        │ 3. DIAGNOSE: match the signal to ONE bottleneck           │
        │ 4. LOOK UP the known fix (KernelWiki / reference source)   │
        │ 5. CHANGE ONE THING                                        │
        │ 6. RE-MEASURE + RE-PROFILE to confirm the metric moved     │
        └─────────────────────────────────────────────────────────┘
```

The skills map cleanly onto the steps. Use each for the question it answers, not as a
generic "make it faster" oracle.

---

## 3. Which skill answers which question

### `ncu-report-skill` — *"Why is THIS kernel slow?"* (the decisive one)

This is the step that broke the problem open, and it should have been step one, not step
twenty. What it told us in a single run that no amount of staring could:

```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum   6,626,752     ← smoking gun
launch__registers_per_thread                               56            ← not a spill problem
WarpStateStats: 77.8% of stalls = "waiting for a scoreboard dependency … to shared memory"
```

That is the entire diagnosis: **the epilogue is bottlenecked on shared-memory bank
conflicts**, not the MMA, not registers, not DRAM, not the store. Every prior guess that
touched the store, the pipeline depth, or the barriers was aimed at the wrong subsystem.

How to run it here (there was no `ncu` on `PATH`; it lives in the CUDA toolkit):

```bash
# minimal single-launch harness (no CUDA graph, no torch cast kernels in the way)
/usr/local/cuda/bin/ncu -k "regex:tc_pers" --launch-count 1 \
  --section LaunchStats --section Occupancy --section WarpStateStats \
  --metrics launch__registers_per_thread,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  .venv/bin/python /tmp/prof_harness.py
```

Gotchas this skill explicitly warns about, all of which bit us:
- **Target the kernel by name** (`-k regex:...`). Without it ncu profiles torch's fp8-cast
  `vectorized_elementwise_kernel` and you "profile" the wrong thing.
- **sm_100 metric names differ.** Many stock names return `None`; use the B200 names from
  `reference/08-b200-metric-names.md` (the `l1tex__data_bank_conflicts_*` names above are the
  working ones).
- **Build a tiny standalone harness** that launches the kernel once, warmup outside the
  profiled launch. Profiling through the CUDA-graph benchmark or the pytest path is noise.
- **Read the rule-engine `Est. Speedup` lines first** — they often point straight at it.

### Layout introspection — *"What data does each thread actually touch?"*

ncu said "bank conflicts on smem." It did **not** say *why* the access pattern conflicts.
The `cuda_skill` philosophy — *"printf is your strongest tool"* — applies to CuTe DSL too,
except the print happens at **trace time** and reveals the layout algebra:

```python
print("DBG tTR_rAcc.layout =", tTR_rAcc.layout)
print("DBG tTR_cC coord   =", tTR_cC_full[(None,None,None,None,None,0,0,0)].layout)
# → tTR_rAcc.layout = ((32,1),1,1):((1,0),0,0)
# → tTR_cC coord     = ((32,1),1,1,1,4):((1@1,0),0,0,0,32@1)
```

Decoded: a thread's 32 elements **and** all 4 subtiles vary **only along N** (`1@1`, `32@1`
are N-axis strides; the M-axis stride is 0). **Each thread owns exactly one M-row.** Therefore
the per-row `a_scale` is *one* value per thread per K-block — the code was doing 128 redundant,
conflicting smem loads for a value that never changed. This is the fact that turned "bank
conflicts" into a concrete fix. You cannot get it from ncu; you get it by printing the layout.

**Rule:** ncu tells you *which subsystem* is the bottleneck; the layout print tells you
*why the access pattern hits it*. You usually need both.

### `KernelWiki` — *"How did the experts solve this exact problem?"*

Use it for the **known-good pattern**, so you don't reinvent (or misinvent) it. It carries the
Blackwell-specific techniques as cross-referenced, source-backed pages:

```bash
python3 scripts/query.py "overlapping accumulator tmem double buffer"   # → pr-cutlass-2995
python3 scripts/query.py "tma store epilogue smem" --type kernel
python3 scripts/query.py --symptom tail-effect --compact
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
```

Where it paid off:
- The **overlapping-accumulator** (2 TMEM banks, `num_acc_stage=2`) came from
  `pr-cutlass-2995`, including the warning that the PR *itself* shipped release-timing bugs —
  which is exactly the class of bug that later produced "unspecified launch failure" and cost
  a debugging cycle. Reading the wiki first would have pre-warned us.
- Respect the confidence ladder (`verified` > `source-reported` > `inferred`) and always quote
  perf claims with all six fields (gpu/dtype/shape/metric/value/source). A wiki number is a
  *hypothesis to reproduce on your hardware*, not a target — our own base measured 3718 TF/s
  where an early assumption of "~2250 peak" would have made us stop three optimizations short.

### `cuda_skill` — *"Debug the crash / what does this instruction guarantee?"*

Two distinct uses:
1. **Correctness debugging** when a change segfaults: `compute-sanitizer`, trace-time prints,
   minimize-the-diff. The TMA-store epilogue's launch failures (mbarrier overflow from
   releasing on all 128 threads; missing `fence_view_async_tmem_load`) are exactly this — the
   fix was "release with `cute.arch.elect_one()`," found by staring at the diff against the
   working SIMT path, which the skill explicitly endorses as a legitimate technique.
2. **Instruction-level ground truth** via the local PTX/CUDA docs (`references/ptx-docs/`).
   When you need to know what `tcgen05.ld`, TMA `cp.async.bulk`, or a proxy fence actually
   orders, grep the ISA rather than guess the semantics.

### flash-attention CuTe DSL source (`reference/flash-attention/.../cute`) — *"What does a real warp-specialized persistent kernel look like?"*

This is the **architecture reference**, not a copy-paste target. FA4's structure is the
template the whole win is built on:
- **Warp specialization**: dedicated MMA warp, TMA/load warp, epilogue warps running
  concurrently through pipelines + named barriers. Our kernel's warp roles (epilogue 0–3,
  MMA 4, TMA 5) mirror this.
- **Persistent tile scheduling**: fixed grid = #SMs looping over work-tiles, avoiding tail
  effects — the reason the persistent base was 3718 TF/s vs the non-persistent 1257.
- **Pipeline handshakes**: producer/consumer `acc_pipeline`, `PipelineTmaStore`, and the
  `elect_one` / `fence` / `arrive_and_wait` idioms that make async TMEM↔smem↔gmem safe.

Read it to answer "how are the warps and pipelines supposed to fit together," then adapt the
concrete library helpers (`utils.gemm.sm100.epilogue_*`, `cpasync.tma_partition`) to your
epilogue. Reading the reference is *not* guessing — it's importing a verified structure.

---

## 4. The worked example, start to finish

| Step | Tool | What it produced |
|---|---|---|
| Fair baseline | CUDA-graph bench, both shapes | 650 TF/s, honest 3× gap (an earlier "win" was a measurement artifact — retracted) |
| Ceiling probe | one-line experiment (`sa = 1.0`) | TMA store alone → **2365 TF/s**: proves the store, not the math, was the wall |
| Known-good store pattern | KernelWiki + `dense_gemm_persistent.py` | r2s → `PipelineTmaStore` → TMA epilogue, wired from library helpers |
| The real bottleneck | **ncu** | 6.6M smem bank conflicts, 78% smem-scoreboard stalls |
| Root cause of the pattern | **trace-time layout print** | each thread owns ONE M-row ⇒ 128× redundant conflicting loads |
| The fix | one change | hoist `sa` to a single coalesced gmem load per K-block, fold with `sb` |
| Confirm | re-measure + re-profile | 1140 / 2153 TF/s, conflicts gone, rel-L2 1.66e-3 |

Note the shape of it: **two measurements bracket the search** (baseline + ceiling probe), the
profile names the subsystem, the layout print names the cause, and the fix is *one line*.
Everything in between that was a guess got reverted.

---

## 5. Checklist to run before touching a kernel

- [ ] Fair baseline recorded (CUDA-graph, ≥2 representative shapes, vs the real competitor).
- [ ] A *ceiling* experiment done (disable the suspect work) so you know the prize is worth it.
- [ ] ncu `--set full` + `--section WarpStateStats` on the actual kernel (by name, sm_100
      metric names, standalone harness).
- [ ] The dominant stall reason named, with the 2–3 metric values that prove it.
- [ ] For layout/access-pattern bottlenecks: the thread→data mapping printed and decoded.
- [ ] The known-good fix located (KernelWiki page id or reference-source file:line) before
      writing new code.
- [ ] Exactly ONE change made, then re-profiled to confirm the metric moved (not just wall-time).
- [ ] Any perf number reported with the shape + measurement method, and any regression/skip
      stated plainly.

---

## 6. TL;DR

The win was not cleverness — it was **one ncu run plus one layout print**. The three weeks of
plausible edits before that produced nothing because they were aimed by intuition at the wrong
subsystem. Profile first; the hardware will tell you where to look, and the reference sources
will tell you what to write once you know.
