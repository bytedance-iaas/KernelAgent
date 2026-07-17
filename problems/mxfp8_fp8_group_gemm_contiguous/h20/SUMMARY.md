# ka-kernel-opt run summary — mxfp8_fp8_group_gemm_contiguous (H20)

Goal shape: large_m (G=4, m/g=2048, N=K=1024, M=8192). Metric: h20/perf_tests.py.
DeepGEMM baseline at goal: 184.4 us. User stretch target: 66 us.

## Rounds (goal-shape wall time, event-timed per call)
| round | change | us | verdict |
|---|---|---|---|
| 0 | initial (2-deep chunk pipeline, int-domain scales) | 314.0 | baseline |
| 1 | full-depth 4-accum pipeline (single drain) | 356.5 | reject (no overlap, 240 regs) |
| 2 | fp32 scales precomputed at staging; promotion = LDS+FMUL/FFMA | 212.7 | ACCEPT |
| 3 | host launch-state cache (wrappers/compiled reuse; empty-vs-zeros) | 151.5 | ACCEPT |
| 4 | cross-tile promote-behind | 156.3 | reject |
| 5 | + per-warpgroup scale regions/named barriers | 153.1 | reject |
| 6 | per-warpgroup barriers on round-3 structure | 147.6 | ACCEPT |
| 7 | hoist A-scales via 2x LDS.128 (unroll=2 rejected) | 145.9 | ACCEPT |
| 8 | CUDA-graph hot path (+clone) | 119.3 | ACCEPT |
| 9 | permuted B-scale LDS.128 | 135.8 | reject (short-sb 10->35%, l1tex 88%) |
| 10 | drop clone; return internal buffer (documented reuse contract) | 110.0 | ACCEPT (final) |
| 11 | B-scale pair prefetch | 110.1 | reject (flat) |
| 12 | staging LDG latency carry | 109.9 | reject (noise, complexity) |
| 13 | ring-4 scales + barrier every 2nd tile | 112.3 | reject |
| 14 | STAGES=2 / STAGES=4 sweep | 118.1 / 138.5 | reject (3 optimal) |
| 15 | BM=256, 4 warpgroups, 1 CTA/SM | 112.2 | reject; plateau stop |

## Final (h20/perf_tests.py)
| case | deepgemm us | cutedsl us | ratio | diff |
|---|---|---|---|---|
| canonical 256x48x128 | 18.0 | 12.3 | 0.68x | 0.0 |
| small_m 512x1024x1024 | 30.5 | 17.5 | 0.57x | 0.0 |
| mid_m 2048x1024x1024 | 69.4 | 38.2 | 0.55x | 0.0 |
| large_m 8192x1024x1024 | 184.4 | 110.4 | 0.60x | 0.0 |

Beats DeepGEMM on every workload (1.47-1.82x). Output remains bit-identical
to DeepGEMM / diff ~1e-7 vs the fp32 reference.

## Why 66 us was not reached (floor analysis, NCU-grounded)
- FP8 tensor-pipe floor at 296 TF: 17.18 TFLOP -> 58 us at 100% WGMMA
  utilization (clocks verified at 1980 MHz, not power-throttled).
- Issue floor: 34.6M instructions / (312 SMSP x 1.98G) = 56 us at IPC 1;
  the exact MXFP8 promotion (1 FMUL + 1 FFMA per element per 32-K chunk =
  the irreducible fp32 scale application) is ~64% of that stream.
- 66 us wall would require BOTH floors at ~100% occupancy simultaneously
  plus zero host cost; no exact-semantics implementation can do it. Final
  110.4 us = 1.9x the max(floor); remaining gap is correlated
  barrier/scoreboard stalls (measured 12%/10%/8%).

## Phase 2 (rounds 16-20, target 80 us at large_m)
| round | change | us | verdict |
|---|---|---|---|
| 16 | per-wg-half A staging split | 110.8 | reject (flat) |
| 17 | per-chunk smem view + LDS [base+imm] (ALU 31->21%) | 107.5 | ACCEPT |
| 18 | async mbarrier scale pipeline replacing bar.sync | 114.0 | reject (mbar op cost > parked-warp stall) |
| 19 | merge TMA read/release states + hoist sb base | 106.9 | ACCEPT (final) |
| 20 | interior epilogue fast path + coord hoist | 111.0 | reject |

Final large_m: 107.0 us vs DeepGEMM 184.1 (0.58x, 1.72x faster), 160.6 TF.
80 us NOT reached. Convergence evidence: issue-active pinned 58-65% across
all structural variants; stall profile flat across 5 categories (~0.5-1.0
warps each); reaching 80 us needs issue-active ~79% at the measured 30.4M
instructions -- no attempted lever (barrier elision, LDS widening/prefetch,
staging splits, pipeline depth, tile shape, warpgroup count) moved it more
than ~2pp without costing more elsewhere. Practical asymptote of this
exact-semantics architecture on H20 is ~100-105 us.

## Phase 3 (rounds 21-24, target 80 us, SASS-driven via mandatory static analysis)
Static analysis (manual fallback; artifacts in static_dump_round_21/):
mainloop = 466 SASS instructions (FMUL 128 + FFMA 128 = 55% irreducible
promotion math, LDS 34, ~176 misc), zero spills (no LDL/STL), regs 122.
Dynamic inst count matches statically (466 x 8 tiles = 3712/warp = 30.4M).

| round | change | us | verdict |
|---|---|---|---|
| 21 | issue chunk-0 WGMMA before the scale barrier | 106.2 | ACCEPT (final) |
| 22 | pre-issue chunks 0+1 (2 in flight over barrier) | 106.7 | reject |
| 23 | BM=64 N-split wgs, 72 regs, 3 CTAs/SM, 24 warps | 117.7 | reject (issue-active 58.6->66.6% but inst +32%) |
| 24 | r23 + dedup staging + CTA barrier | 124.4 | reject (re-coupled warpgroups) |

Final large_m: 106.5 us vs DeepGEMM 183.9 (0.58x, 1.73x faster), 161.4 TF.
80 us NOT reached — now proven from three independent directions:
(1) instruction-issue floor: 30.4M inst = 97.5K cycles/SMSP ~ 49-56 us pure
    issue, and the SASS shows the stream is already 55% irreducible math;
(2) FP8 tensor-pipe floor 58 us at 100% WGMMA utilization;
(3) the occupancy<->instruction tradeoff measured directly in r23: +50%
    warps bought +8pp issue-active but cost +32% instructions.
Practical asymptote ~104-106 us for exact-MXFP8 semantics on H20.

## Phase 4 (rounds 25-33, "can op fusion reach 80us?")
Fusion analysis: the only fusion that breaks the exact-semantics floor is
fusing the 4 per-32K promotions into 1 via gran-32->128 scale requantization
(rescale each 128-block to its max chunk exponent; the 4 chunk WGMMAs then
hardware-accumulate like plain FP8). Torch simulation on all 4 workloads:
calc_diff ~0.0, matched 1.0000, max_abs <= 0.125 (atol 0.5-2.0) -> viable
under the problem's tolerance gates (NOT bit-exact in general; loss bounded
by per-block exponent spread; final kernel diff 2.4e-7 vs reference).

| round | change | us | verdict |
|---|---|---|---|
| 25 | requant pre-kernel + hw-accum GEMM (graph of 3 launches) | 114.1 | works, slower (IDIV + full drain) |
| 26 | constexpr-K unrolled loop + cross-tile ping-pong batches | 102.6 | ACCEPT |
| 27 | STAGES=4 (early) | 103.0 | reject |
| 28 | chunk-per-thread requant (128-bit frags, vector cvt) | 94.5 | ACCEPT |
| 29 | scale-carry prefetch in GEMM staging | 92.4 | ACCEPT |
| 30 | parallel requant A||B graph branches | 92.2 | ACCEPT (marginal) |
| 31 | STAGES=4 + ring-8 scale slots (fixed slot-alias race) | 90.9 | ACCEPT |
| 32 | transposed B scales from requant; GEMM reads scales from
|    | global (no scale smem / staging / barrier at all) | 90.2 | ACCEPT (final) |
| 33 | BM=256 1-CTA | 101.3 | reject (1-CTA correlated stalls, 3rd confirmation) |

Final (h20/perf_tests.py): large_m 89.9us vs DeepGEMM 182.3 (0.49x, 2.03x
faster), 191 TF; all four workloads faster than DeepGEMM; all accuracy
gates pass. GEMM alone 80.7us at 81% SM-SOL; requants ~12us at the memory
floor (24MB r/w); the 80us wall target misses by ~10us: requant floor (~7
effective) + GEMM's remaining 19% SOL gap (scale-LDG latency pinned by the
128-register/2-CTA cliff) + graph host (~2.5).
Deployed: h20/kernel.py = fused variant; h20/kernel_exact.py = bit-exact
round-21 kernel (106.5us) for callers that need gran-32 exactness.

## Phase 5 (round 34, "what about enable cudagraph?")
CUDA graphs were already enabled (since round 8; parallel branches round 30).
Measured decomposition at round 32: GPU pipeline 83.3us + host replay-enqueue
6.8us (visible in the sync-per-call perf metric). Round 34 added an
object-identity single-entry fast path (skips tuple/data_ptr/weakref work
before graph.replay()): host 6.8 -> 4.6us.

Final: large_m 87.8us vs DeepGEMM 183.7 (0.48x, 2.09x faster), 195.7 TF.
Remaining gap to 80us (~8us): graph launch itself ~4us (torch binding floor)
+ GPU 83.3 (requant ~7 at memory floor + GEMM 80.7 at 81% SM-SOL; every
remaining GPU lever measured at <= 3us). All accuracy gates pass
(diff 2.4e-7).
