"""Performance comparison for mxfp8_fp8_group_gemm_contiguous on H20 (SM90).

Benchmarks, on identical seeded inputs, per workload:
  * DeepGEMM's plain-FP8 grouped kernel
    `m_grouped_fp8_gemm_nt_contiguous` (recipe (1,128)/(1,128), fp32
    scales) -- the upper-bound kernel DeepGEMM's own perf test compares
    MXFP8 against. NOTE: it consumes a DIFFERENT quantization (fp8 gran-128
    with non-power-of-2 scales, rebuilt here from the dequantized MXFP8
    operands), so its accuracy diff vs the MXFP8 reference is informational
    only and never gated;
  * the ORIGINAL CUTLASS/CUDA MXFP8 kernel
    `deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous` (JIT-compiled DeepGEMM),
  * the generated CuTeDSL kernel (`kernel.py` in this folder), and
  * the pure-PyTorch reference `run()` from problem.py (the parsed
    KernelBench Model, dequantize-then-FP32-matmul).

All are timed with CUDA events (median of many iters). For each workload it
prints latency (us), achieved TFLOP/s (2*M*N*K), speed ratios vs the DeepGEMM
kernel, and each implementation's numerical diff (calc_diff) vs the reference
so the timing is anchored to a correctness check.

Standalone: `python h20/perf_tests.py`. Requires an H20 (sm_90) GPU with the
DeepGEMM extension built; on any other GPU it prints the reason and exits 2
(pinned, never vacuous). Exit 0 once the comparison table is produced and
every implementation's diff vs the reference is below DIFF_TOL.
"""

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # problem.py in the parent problem dir
sys.path.insert(0, str(HERE))          # kernel.py (CuTeDSL) next to this file
import problem  # noqa: E402

GPU_KEY = "h20"
DIFF_TOL = 0.03  # DeepGEMM's own accuracy gate for this kernel family

# (label, groups, m_per_group, n, k) — the contiguous shapes DeepGEMM's own perf
# tests sweep (tests/test_sm90_mxfp8_fp8.py: groups=4, n=k=1024, small/large M),
# plus the canonical accuracy shape.
PERF_CASES = [
    ("canonical",   2,  128,   48,  128),
    ("small_m",     4,  128, 1024, 1024),
    ("mid_m",       4,  512, 1024, 1024),
    ("large_m",     4, 2048, 1024, 1024),
]


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def bench_us(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)  # ms -> us
    times.sort()
    return times[len(times) // 2]


def tflops(m, n, k, us):
    return 2.0 * m * n * k / (us * 1e-6) / 1e12


def _dequant(x_fp8, sf_u8, gran_k=32):
    sf = (sf_u8.to(torch.int32) << 23).view(torch.float32)
    gi = torch.arange(x_fp8.size(-1), device=x_fp8.device) // gran_k
    return x_fp8.float() * sf[..., gi]


def build_fp8_case(a_data, a_scale, b_data, b_scale):
    """Rebuild plain-FP8 (gran-128, fp32-scale) inputs from the dequantized
    MXFP8 operands -- the same construction DeepGEMM's perf test uses."""
    from deep_gemm.utils.math import per_token_cast_to_fp8
    G, N, K = b_data.shape
    a_fp8 = per_token_cast_to_fp8(_dequant(a_data, a_scale),
                                  use_ue8m0=False, gran_k=128)
    b_fp8_data = torch.empty((G, N, K), device="cuda", dtype=torch.float8_e4m3fn)
    b_fp8_sf = torch.empty((G, N, K // 128), device="cuda", dtype=torch.float32)
    for g in range(G):
        b_fp8_data[g], b_fp8_sf[g] = per_token_cast_to_fp8(
            _dequant(b_data[g], b_scale[g]), use_ue8m0=False, gran_k=128)
    return a_fp8, (b_fp8_data, b_fp8_sf)


def build_case(groups, m_per_group, n, k, seed):
    # Use the problem's own custom input builder for exact parity with problem.py.
    torch.manual_seed(seed)
    inp = problem._CUSTOM_INPUTS_FN(
        {"groups": groups, "m_per_group": m_per_group, "n": n, "k": k},
        torch.device("cuda"),
    )
    return (inp["a_data"], inp["a_scale"], inp["b_data"], inp["b_scale"],
            inp["grouped_layout"])


def main() -> int:
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "<no cuda>"
    if GPU_KEY.lower() not in name.lower():
        print(f"this perf test is pinned to {GPU_KEY!r} but the current GPU is "
              f"{name!r} - refusing to judge")
        return 2
    try:
        import deep_gemm
    except Exception as exc:
        print(f"deep_gemm import failed ({exc}); build reference/cuda/sgl-DeepGEMM first")
        return 2
    if not hasattr(deep_gemm, "m_grouped_mxfp8_fp8_gemm_nt_contiguous"):
        print("installed deep_gemm lacks m_grouped_mxfp8_fp8_gemm_nt_contiguous "
              "(rebuild reference/cuda/sgl-DeepGEMM)")
        return 2
    try:
        from kernel import kernel_function as cutedsl_kernel
    except Exception as exc:
        print(f"CuTeDSL kernel import failed ({exc}); expected kernel.py next to "
              "this file")
        return 2

    have_fp8 = hasattr(deep_gemm, "m_grouped_fp8_gemm_nt_contiguous")
    if not have_fp8:
        print("note: deep_gemm lacks m_grouped_fp8_gemm_nt_contiguous; "
              "fp8 columns will be n/a")

    print(f"GPU: {name}")
    header = ("case | M | N | K | fp8 us | mxfp8dg us | cutedsl us | ref us | "
              "fp8 TF | dg TF | cutedsl TF | cutedsl/dg | cutedsl/fp8 | "
              "diff(cutedsl,ref) | diff(dg,ref) | diff(fp8,ref)*")
    print(header)
    print(" | ".join("--" for _ in header.split(" | ")))

    all_ok = True
    results = []
    for i, (label, groups, m_per_group, n, k) in enumerate(PERF_CASES):
        m = groups * m_per_group
        a_data, a_scale, b_data, b_scale, gl = build_case(
            groups, m_per_group, n, k, seed=i)

        d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

        def run_deepgemm():
            deep_gemm.m_grouped_mxfp8_fp8_gemm_nt_contiguous(
                (a_data, a_scale), (b_data, b_scale), d, gl)

        def run_cutedsl():
            return cutedsl_kernel(a_data, a_scale, b_data, b_scale, gl)

        def run_ref():
            return problem.run(a_data, a_scale, b_data, b_scale, gl)

        # plain-FP8 upper-bound kernel (different quantization; may be
        # unsupported for some shapes -- degrade to n/a, never fail the run)
        fp8_us = fp8_diff = None
        run_fp8 = None
        if have_fp8:
            try:
                a_fp8, b_fp8 = build_fp8_case(a_data, a_scale, b_data, b_scale)
                d_fp8 = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

                def run_fp8():
                    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                        a_fp8, b_fp8, d_fp8, gl,
                        recipe_a=(1, 128), recipe_b=(1, 128))

                run_fp8()
                torch.cuda.synchronize()
            except Exception as exc:
                print(f"# fp8 kernel unavailable for {label}: "
                      f"{type(exc).__name__}: {exc}")
                run_fp8 = None

        # correctness anchor: every implementation vs the PyTorch reference
        run_deepgemm()
        cutedsl_out = run_cutedsl()
        ref_out = run_ref()
        torch.cuda.synchronize()
        diff_cutedsl = calc_diff(cutedsl_out, ref_out)
        diff_dg = calc_diff(d, ref_out)
        if run_fp8 is not None:
            fp8_diff = calc_diff(d_fp8, ref_out)
        case_ok = diff_cutedsl < DIFF_TOL and diff_dg < DIFF_TOL
        all_ok &= case_ok

        if run_fp8 is not None:
            fp8_us = bench_us(run_fp8)
        dg_us = bench_us(run_deepgemm)
        ct_us = bench_us(run_cutedsl)
        rf_us = bench_us(run_ref)

        results.append(dict(
            case=label, M=m, N=n, K=k,
            fp8_us=fp8_us, deepgemm_us=dg_us, cutedsl_us=ct_us, ref_us=rf_us,
            diff_cutedsl_ref=diff_cutedsl, diff_deepgemm_ref=diff_dg,
            diff_fp8_ref_informational=fp8_diff,
            ok=case_ok))
        fp8_us_s = f"{fp8_us:.1f}" if fp8_us is not None else "n/a"
        fp8_tf_s = f"{tflops(m, n, k, fp8_us):.1f}" if fp8_us else "n/a"
        fp8_ratio_s = f"{ct_us / fp8_us:.2f}x" if fp8_us else "n/a"
        fp8_diff_s = f"{fp8_diff:.6f}" if fp8_diff is not None else "n/a"
        print(f"{label} | {m} | {n} | {k} | "
              f"{fp8_us_s} | {dg_us:.1f} | {ct_us:.1f} | {rf_us:.1f} | "
              f"{fp8_tf_s} | {tflops(m, n, k, dg_us):.1f} | "
              f"{tflops(m, n, k, ct_us):.1f} | "
              f"{ct_us / dg_us:.2f}x | {fp8_ratio_s} | "
              f"{diff_cutedsl:.6f} | {diff_dg:.6f} | {fp8_diff_s}"
              + ("" if case_ok else "  <-- DIFF FAIL"))

        del a_data, a_scale, b_data, b_scale, gl, d, cutedsl_out, ref_out
        torch.cuda.empty_cache()

    print("* fp8 columns use a different (gran-128, fp32-scale) quantization "
          "of the same source data; diff(fp8,ref) is informational, not gated")
    print(json.dumps(dict(gpu=GPU_KEY, all_ok=all_ok, results=results)))
    print("PERF DONE" + ("" if all_ok else " (WITH DIFF FAILURES)"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
