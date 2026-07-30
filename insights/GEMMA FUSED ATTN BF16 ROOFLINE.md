# gemma_fused_attn: when the honest answer is "the target is below the roofline"

A retrospective on a fused **Gemma decoder layer** (Pi0.5 VLM / prefix side, gemma_2b:
B=2, S=968, H=2048, MLP intermediate 16384, GQA 8q/1kv, head_dim 256) taken from a
correctness-first Triton kernel to a **near-roofline** one — and, more importantly, on
the decision *not* to spend 30 optimization rounds chasing a physically impossible number.

The single lesson: **establish the compute roofline and a measured floor before committing
to a perf target.** A ten-second cuBLAS decomposition turned "optimize until 3× for at most
30 rounds" from an open-ended grind into a one-line proof of infeasibility — and reframed
the work around what was actually achievable.

Companion to `PROFILING METHODOLOGY.md`; this followed the same loop (measure → name the
limit → one change → re-measure), and the one place it could have gone wrong was accepting
the target at face value.

Environment: NVIDIA **H20** (bf16 tensor peak measured 138 TF/s, spec ~148), transformers
4.53.2 (openpi patched, for the reference layer). Artifacts:
`problems/gemma_fused_attn/h20/{kernel.py, test_kernel.py, perf_tests.py}`,
reference `problems/gemma_fused_attn/ref_tests.py`.

---

## 0. The goal, stated as a measurable

> Make the Triton kernel **3× faster** than the reference `GemmaDecoderLayer`
> (`ref_tests.build_reference_layer`, transformers eager) on the gemma_2b workload,
> in bf16, while staying correct (rel < 2e-2 — the reference harness's own `report()`
> threshold). Plus: ship a `perf_tests.py`, and make the kernel CUDA-graph capturable.

Reference latency: **4.52 ms**. So "3×" means **≤ 1.51 ms**.

---

## 1. The feasibility check that saved 30 rounds

Before touching the kernel, one decomposition — every GEMM run in **cuBLAS** (optimal,
zero norm/rope/residual/launch overhead):

| GEMM (cuBLAS) | ms |
|---|---|
| qkv proj | 0.20 |
| o proj | 0.16 |
| gate proj | 1.00 |
| up proj | 1.00 |
| down proj | 1.19 |
| attention (SDPA) | 0.24 |
| **Σ GEMMs** | **3.55** |
| **+ attention** | **3.79** |

The three MLP GEMMs alone are **3.19 ms** of irreducible bf16 tensor-core work. This is the
floor **fusion cannot cross** — fusion removes memory traffic and launches, never matmul
FLOPs.

The arithmetic that ended the debate:
- Layer is **441.7 GFLOP** (389 of it the MLP).
- H20 bf16 peak (measured, 8192³ matmul) = **138 TF/s**.
- Compute floor = 441.7 / 138 = **3.19 ms** → best possible **1.42×** vs the 4.52 ms reference.
- The 3× target (1.51 ms) would require **~230% of hardware peak**. Impossible in bf16.

And the starting kernel was already at **3.53 ms** — *already at the cuBLAS GEMM sum*. There
was almost nothing left to win in bf16.

> **Decision:** surface the roofline with numbers and ask, rather than grind. The user chose
> to stay bf16 ("squeeze the last few %") once shown that 3× requires FP8. Grinding 30 rounds
> at a sub-roofline target would have produced 30 confident-but-doomed diffs.

The only lever that reaches 3× is **fewer FLOP-equivalents** → FP8 tensor cores (H20 FP8 ≈ 2×
bf16 → ~1.6–2.0 ms realistic incl. quant overhead, i.e. ~2.4–2.8×, possibly still short of an
exact 3×), at the cost of the rel < 2e-2 gate. Deferred by choice.

---

## 2. What the profile said about the *non*-GEMM time

3.53 ms − 3.19 ms (GEMMs) ≈ 0.34 ms of headroom, and it decomposed as:
- **attention ~0.24 ms** — already at SDPA level; swept block sizes 32/64/128 and 64×64 was
  the best that fit shared memory (128×* → out-of-resource). No headroom.
- **gate·up elementwise mul** — a separate kernel reading gate+up, writing prod (~190 MB of
  HBM traffic + a launch).
- **launches / norms / rope** — small.

So the *only* actionable bf16 win was the mul.

---

## 3. The one change that helped, and the one that (correctly) didn't

**Helped — fuse `up * gelu_tanh(gate)` into the up-projection epilogue.** The GEMM already
has `up` in registers; load `gate`, multiply, store `prod`. Removes a full [1936,16384]
buffer round-trip and a launch.
- 3.53 ms → **3.47 ms** (1.28× → 1.30×). Correctness held (rel 7.1e-3).

**Did ~nothing — CUDA graphs.** Capture/replay works and is **bit-identical** to eager
(rel 0.00), but replay (3.456) ≈ eager (3.467). Expected: the layer is compute-bound, so the
long GEMMs already hide the ~9 launches. **CUDA graphs help launch-bound kernels, not
compute-bound ones** — the value here is the *capability* the user asked for, not latency.
This is worth stating because it's a common wrong expectation ("add a graph, get a speedup").

Not attempted (measured to be pointless): autotuning the big GEMMs. gate/up already run at
**135.6 TF/s (98% of peak)**, down at 133 TF/s (96%) — the naive 64×64×64 tiles are already
at/above cuBLAS for these shapes (my down-proj beats cuBLAS 0.97 vs 1.19 ms). There is no MFU
left to autotune into.

---

## 4. Final numbers (perf_tests.py, H20)

| impl | ms | speedup | TFLOP/s | MFU |
|---|---|---|---|---|
| reference GemmaDecoderLayer (eager) | 4.52 | 1.00× | 97.7 | 70.6% |
| triton kernel (eager) | 3.46 | 1.305× | 127.5 | 92.1% |
| triton kernel (cudagraph replay) | 3.46 | 1.307× | 127.8 | 92.3% |

- H20 bf16 roofline (measured): **138.5 TF/s** (spec ~148).
- bf16 end-to-end floor ≈ **3.44 ms** (GEMMs 3.20 at 96–98% MFU + attention 0.24, sequential)
  → **~1.31×**, which the kernel sits at.

The kernel went from **71% → 92% MFU** relative to the reference. The remaining 8% is the
irreducible sequential attention + norms/rope that can't overlap the GEMMs. `perf_tests.py`
measures the roofline live (so it self-adjusts on other GPUs) and prints per-impl TFLOP/s +
MFU, making "we are at the ceiling" a number, not a claim.

---

## 5. Weaving the kernel into the reference harness (fwd + bwd)

`ref_tests.main()` checks **forward, grad_input, and every parameter gradient** — but the
Triton kernel is forward-only. `build_fused_layer` wraps it in a `torch.autograd.Function`:
- **forward** runs the fast fused Triton kernel (the value returned is the kernel's, rel 7.3e-3);
- **backward** recomputes gradients through the equivalent validated PyTorch math
  (`_torch_layer`, reusing `problem.py`'s `_rms_norm`/`_rotate_half`).

Two details that make the check *actually* check:
- Parameter names mirror `GemmaDecoderLayer` (`self_attn.q_proj.weight`, `mlp.gate_proj.weight`,
  `input_layernorm.weight`, …) so the harness's `if n in ref_named` per-param grad comparison
  **engages for all 9** instead of silently skipping.
- Result: **ALL PASS** — forward 7.3e-3, grad_input 7.9e-3, param grads 2.2e-3–6.9e-3, all
  under the 2e-2 bar.

Honest limit: the backward is a PyTorch recompute, not a Triton backward. Correct for an
inference kernel + a grad *check*; a training-grade layer would need flash-attention-backward
+ GEMM-transpose kernels (a much larger effort).

---

## 6. Lessons

1. **Roofline before rounds.** A perf target expressed as a multiplier ("3×") is meaningless
   until compared to the hardware floor. cuBLAS-decompose the workload and divide FLOPs by
   measured peak *first*. If the target is below the floor, say so with numbers.
2. **Fusion ≠ speed on a compute-bound kernel.** It removes memory traffic and launches. When
   MFU is already 90%+, there is nothing for it to remove. Know which regime you're in before
   promising a fusion win.
3. **CUDA graphs ≠ speed on a compute-bound kernel** either — same reason. Add them for the
   capability (capturable inference), and *measure* rather than assume a latency gain.
4. **MFU is the honesty metric.** "1.3×" sounds unfinished; "92% of bf16 peak" says the work
   is done and the rest requires changing precision, not code.
5. **Make the checker actually check.** Mirroring reference parameter names was the difference
   between validating 2 quantities and validating 11.
