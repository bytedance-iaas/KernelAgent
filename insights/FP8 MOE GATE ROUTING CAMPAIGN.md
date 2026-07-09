# fp8_moe_gate_routing: every option tried, what won, what lost, and why

A complete retrospective of taking DeepSeek-R1's noaux_tc FP8 MoE gate (SOL-ExecBench
`Quant/011`, 16 workloads M=256..8192, N=256, K=7168, B200-graded) from a
**3.74 ms** correctness-first CuTe DSL kernel to **0.034–0.085 ms**
(SOL-Scores 0.843–0.878 locally, ~0.74 on the SOL-ExecBench website), including
the dead ends — several of which ended in *proofs*, not plateaus.

Companion to `PROFILING METHODOLOGY.md`; this campaign followed that loop
(profile → name the stall → one change → re-measure) and its failures happened
exactly where the loop was shortcut.

Evidence trails: `problems/fp8_moe_gate_routing/.optimize/run_20260708_143346/notes_round9.md`
(rounds 1–15, program DB) and `problems/fp8_moe_gate_routing/.optimize/bf16a/PROGRESS.md`
(fusion campaigns, delegated agent). Environment: GB200, nvidia-cutlass-dsl 4.4.1,
torch 2.9.0+cu130 (matched to the grading venv).

---

## 0. The problem and its gates

`run(hidden_states bf16 [M,7168], weight bf16 [256,7168], bias [256], scale_x
[M,56], scale_w [56,2], rsf)` → fp8-quantize x (1×128 scales) and wᵀ (128×128
scales) *with the exact torch round trip*, fp8 GEMM → bf16 logits, sigmoid +
bias, sum-of-top-2 per group of 32, top-4 of 8 groups, top-8 experts **in
torch's undocumented topk order**, normalize → `(topk_idx int64, topk_weight f32)`.

- Correctness gate: matched-ratio ≥ 0.99 elementwise (int64 indices ⇒ near-exact),
  atol 5e-3 / rtol 2e-2 on weights. Tolerances are the problem's own — never widened.
- Perf gate: SOL-Score `S = 1/(1+(t−sol)/(baseline−sol))` ≥ 0.5 per workload
  (`sol` = SOLAR analytic roofline; S=1 at SOL, 0.5 at baseline).

**Discovery that unblocked correctness at all:** `torch.topk(..., sorted=False)`
on CUDA returns *indices with value strictly greater than the k-th value in
ascending index order, then the k-th-value ties ascending*. Reproducing that
layout (plus lowest-index tie-breaks everywhere) is what makes int64 index
outputs comparable at 0.99 matched ratio.

---

## 1. The architecture that won (three stages + CUDA graphs)

```
quant_x (448-thr/row, 128-bit atoms, bit-exact e4m3 round trip)   ~6–35 us
quant_w + scale-transpose (side stream, hidden)                    ~4–6 us
tcgen05 persistent groupwise GEMM (tc_grouped, G=1;               ~16–21 us
    shape-aware (64,128)/(128,128) tiler)
routing epilogue (warp-per-token, redux.sync argmax,               ~8–22 us
    registers-only hot loops, torch-topk output order)
— all captured in one CUDA graph per shape; caller I/O bound in via
  graph-exec pointer patching (see §5)
```

## 2. Score ledger (workload 0 = M4352 unless noted)

| step | change | time | note |
|---|---|---|---|
| gen | correctness-first scalar CUDA-core kernel | 3.74 ms | 16/16 correct, matched=1.0 |
| r1 | quant kernels + tcgen05 GEMM + routing kernel | 0.415 ms | tensor pipe 0%→used |
| r2 | CUDA graph per shape (copy-in) | 0.161 ms | host launch overhead was ~60% |
| r3 | warp-per-token routing (shuffle argmax) | 0.141 ms | routing 108→44 us @8192 |
| r4 | zero-copy pointer-keyed graphs | 0.106 ms | kills 43–60 us copy-in |
| r5 | 128-bit copy atoms in quant | 0.099 ms | |
| r6 | register-resident routing hot loops | 0.084 ms | routing → 26.6 us @8192 |
| r7 | side-stream quant_w; rsf fill cache | 0.072 ms | first PERF PASS scores .59–.68 |
| r9/10 | fast sigmoid (exp2+rcp) + redux.sync argmax | 0.092→ | @8192; −4 us total |
| r11 | vector math in quant | −2 us | |
| — | stack → 4.4.1 + torch 2.9 (grading env) | 0.060 ms | everything ~20% faster free |
| r13 | exact-fit 448-thr quant blocks | −2 us mid-M | |
| t64 | (64,128) tiler where one wave fits (spin-off of fusion campaign) | −0.9..−4.2 us on 12/16 workloads | scores → 0.843–0.878 |

## 3. Everything that was tried and REJECTED (with the refuting measurement)

| option | expectation | measurement | verdict |
|---|---|---|---|
| transposed smem `[tok][j][lane]` routing layout (r8) | kill 8-way bank conflicts | wall unchanged (26.6 us) | conflicts weren't the bottleneck; NCU showed MIO-throttle+latency |
| routing v3: 2 tokens/warp for ILP (r14) | overlap latency stalls | 8.2→10.3 @256, 22.5→24.6 @8192 | regression; register state + fewer CTAs beat the ILP win |
| non-persistent `DenseGemmKernel` | lower fixed cost than persistent | 61–63 us vs 27–30 | 2× worse |
| 2-CTA (256,128) tiler | more MMA throughput | identical timing | grid too small to matter |
| cluster (1,2) A-multicast | halve A reads (N=256 = 2 tiles) | identical timing | not A-BW-bound; fixed-cost floor |
| (64,128) tile64, first 2 attempts (r12/15) | halve fixed cost | numerics broken, then correct-but-slow (memo +8 us, bitmask +25 us/elem overhead ×56 k-tiles) | per-element epilogue overhead multiplies by k_tile_cnt |
| tile64 at M≥5888 | more tiles | 43–53 us vs 30.7 | wave quantization (184+ tiles = 1.24+ waves) |
| `_foreach_copy_` staging | fuse 5 copies | 40.6 vs 40.5 us | 60 MB copy is BW-bound |
| quant QV=32 / unroll / 2-row blocks | beat 5.1 TB/s | 35–43 us, none better | at the wall: torch's own clone = 6.0 TB/s ceiling |
| div→mul in quant (probe only) | div-bound hypothesis | −4 us of 35 | div is NOT the limiter; also not bit-exact, never shippable |

## 4. The fusion campaigns — ended in proofs, not plateaus

Goal: SOL-Score ≥ 0.9 needs t ≤ sol+(baseline−sol)/9 = 23.2/41.6/70.0 us
@256/4352/8192; the 3-stage floors sum above that, so quantization had to move
inside the GEMM. Delegated to a background agent (fresh context, staged gates).

**bf16-A producer fusion (128-tile):** BUILT and bit-exact — producer warps
load bf16 A, quantize in registers (Markstein rcp+2-Newton division proven
bit-identical to `div.rn.f32` over ~200M elements), write swizzled smem, MMA
unchanged. **Failed on perf (78–100 us vs 22–56 sequential)** with two
quantified walls: (1) the 255-reg groupwise epilogue forbids extra warps —
uniform ptxas budget 192thr=255r → 512thr=64r with 233K local-memory ops, and
**4.4.1 has no register-cap plumbing at all** (compiles LLVM IR direct to SASS;
`--maxrregcount` and `setmaxnreg` verifiably ignored); (2) streaming bf16 A
needs ~108 KB in flight per CTA vs ~32 KB attainable — 8192 tiny quant CTAs
beat any 128-CTA persistent kernel on memory parallelism.

**2-CTAs/SM:** disproved in one probe — 255r×192thr×2 = 98K > 64K register
file (NCU: not co-resident), and decisively: the grid has only 68/128 CTAs at
M=4352/8192, *fewer than 148 SMs* — a second CTA per SM adds parallelism only
above M≈9500.

**tile64 fusion:** the (64,128) epilogue (~130 regs) finally fit 8 producer
warps and everything stayed bit-exact — and still failed on a **closed-form
issue-bandwidth bound**: the budget at M=4352 allows 6.25 instructions/element,
exactly the theoretical minimum of the bit-exact quant math with zero loads and
zero stalls. In-GEMM quantization is *impossible* on this stack, not merely
unprofitable. (Its spin-off — the mode-static 2-row a_scale epilogue with
integer-exact row selects — is what shipped as the t64 tiler, −2.1 us wherever
one wave fits. The third attempt at that epilogue succeeded because it moved
the row-selection out of the per-element k-loop entirely; the fp-blend form
`a+w·(b−a)` was a bit-exactness trap.)

## 5. The serving/harness layer (where the website score lives)

The grading harness allocates **fresh input tensors every call**; local
perf_test reuses them. Three iterations:

1. Pointer-keyed zero-copy graphs with pinned refs → **~1.5 ms/call on the
   website** (pinning blocks allocator recycling → every call re-captures).
2. Promotion-after-repeat + copy-in staging fallback → capture spikes moved
   into warmup, medians 0.05–0.11 ms; but only reaches zero-copy speed if the
   harness recycles pointer sets.
3. **Graph-exec pointer patching (shipped):** torch 2.9
   `CUDAGraph(keep_graph=True)` → `raw_cuda_graph_exec()`; pointer slots found
   *empirically* (capture twice with different buffers, byte-diff the 5 kernel
   nodes' param blobs → 7 slots, incl. one at +512 inside a torch
   TensorIterator blob); `cuGraphExecKernelNodeSetParams` costs 10.6 us and
   only fires when pointers change. True zero-copy for ANY pointer behavior,
   zero recaptures, bit-identical, with a copy-in fallback on any failure.

Also load-bearing: staging buffers must be benign-initialized (capture warmup
runs the pipeline before real data exists; garbage scales → NaN scores → OOB
smem indices) and selection kernels NaN-hardened.

Remaining local-vs-website gap (~0.81 sim vs ~0.74 site): ~17 us/call of
harness-side sync+Python wall, plus B200-vs-GB200 clocks. Not reachable from
inside a submission.

## 6. Porting to the grading stack (4.4.1 + torch 2.9) — the drift list

Every one of these produced a confusing failure first: fp8 refuses DLPack
(export uint8 views + set `element_type`); `const_expr(<kernel param>)`
rejected (specialize kernels instead of mode flags); `cute.math` lost
clamp/min/max/rcp (use `cutlass.min/max`); smem-struct members lost `.ptr`;
**`from __future__ import annotations` silently breaks Constexpr detection in
@cute.jit signatures** (PEP-563 strings); compiled callables take dynamic args
only in BOTH versions. Full list with fixes:
`~/.claude/.../memory/cutedsl-dsl-pitfalls.md`.

## 7. Transferable lessons

1. **Match the grading environment early.** Half a day went to API drift and
   the fp8-DLPack surprise that a day-one `pip install` of the grader's deps
   would have surfaced; the stack switch also made everything 20% faster.
2. **Benchmark the caller's pattern, not just the kernel's.** Three separate
   website regressions (capture-per-call, capture-in-window, copy-in tax) were
   invisible to a pointer-stable local harness. The fresh-tensor simulation
   became a standing gate.
3. **Kill hypotheses with cheap probes before surgery.** The best rounds each
   started with a <30-line probe (fp8 round-trip bit-exactness, div-vs-mul,
   copy-BW ceiling, tile64 timing-with-broken-numerics, 2-CTA occupancy). The
   worst round (r8) skipped the probe and optimized a non-bottleneck.
4. **Fixed costs and wave quantization dominate thin-N GEMMs.** N=256 means
   the grid is CTA-starved at every graded M; almost every "throughput" idea
   (2-CTA, multicast, more tiles) died on grid geometry, not FLOPs.
5. **Accumulate rejections as facts.** The three fusion verdicts are proofs
   (register file arithmetic, CTA-count arithmetic, instruction-issue lower
   bound) — nobody needs to re-run them; that's the difference between a
   closed design space and a folder of abandoned branches.
6. **Bit-exactness is achievable and worth it** even through hardware fast
   paths: e4m3 cvt.rn.satfinite == torch, Markstein division == div.rn,
   integer-exact selects instead of fp blends. It converts "tolerance
   arguments" into `torch.equal`.
7. **Delegate deep surgery to a fresh context with a written spec and staged
   gates.** The fusion campaigns ran as a background agent off two documents
   (brief + pitfalls); every hand-off survived because state lived in
   PROGRESS.md, not in anyone's head.

## 8. Final state

- `problems/fp8_moe_gate_routing/b200/`: `kernel.py` == `submission.py`,
  `submission_final.py` (single-file, no local imports, `run` entrypoint),
  `tc_grouped.py` (+t64 dispatch), `tc_persist_t64.py`, pristine
  `tc_persist_gemm.py`/`tc_groupwise_gemm.py`.
- Gates: test.py 16/16 matched=1.0000 · perf gate PERF PASS, scores
  0.843–0.878 (mean 0.857) · fresh-tensor sim 0.810 with mean==median ·
  churn-safe, NaN-hardened, fallback-covered.
- Known-stale: `b200/submission.tar.gz` (predates all of the above).
