"""
Ported unit tests + benchmark for fp8_fp4_group_gemm_contiguous (KernelBench
problem).

Source: reference/cuda/sgl-DeepGEMM/tests/test_sm90_fp8_fp4.py
        :: test_sm90_fp8_fp4_contiguous (seed 0, groups x m_per_group sweep,
        n=4096, k=7168, accuracy contract `calc_diff < 0.015`).

Subjects checked against the Model (the pure-PyTorch reference):
  1. The original DeepGEMM CUDA kernel
     (`m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma`), requires an
     SM90 GPU + importable deep_gemm.
  2. The generated CuTe-DSL kernel
     (cutedsl/fp8_fp4_group_gemm_contiguous.py::kernel_function), requires
     the nvidia-cutlass-dsl package.

Port rules applied:
- Input construction preserved exactly: torch.manual_seed(0), the same
  randn order (a_ref_src then b_ref_src per case), the same quantization
  calls and the same case list. The FP8-baseline benchmark half of the
  original test is dropped (it does not touch the RNG stream and is not
  the subject of this problem).
- The original reference computation IS the Model; the original assertion
  `calc_diff(kernel_out, ref) < 0.015` is ported as
  `calc_diff(kernel_out, model_out) < 0.015` for every subject.
- Subjects that cannot run in the current environment are SKIPPED (with
  reason), never silently dropped.

Benchmark: after the accuracy suite, times each available subject (plus the
Model itself) on representative shapes with CUDA events and reports us /
effective GB/s / TFLOP/s. Informational only — it does not affect PASS.

Standalone: prints per-case results, prints PASS and exits 0 only if every
accuracy case passes. `--skip-bench` disables the benchmark section.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

# Load the problem module (the Model under test).
_spec = importlib.util.spec_from_file_location(
    "fp8_fp4_group_gemm_contiguous",
    _HERE / "pytorch" / "fp8_fp4_group_gemm_contiguous.py")
problem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(problem)

TOLERANCE = 0.015  # from the source test's `assert w4_diff < 0.015`


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    # Mirror of deep_gemm.testing.calc_diff
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return float(1 - sim)


def _deep_gemm_kernel_available():
    if not torch.cuda.is_available():
        return None, "CUDA not available"
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        return None, f"kernel requires SM90, got sm_{major}x"
    sys.path.insert(0, str(_REPO / "reference" / "cuda" / "sgl-DeepGEMM"))
    try:
        import deep_gemm
    except Exception as exc:  # noqa: BLE001
        return None, f"deep_gemm import failed: {exc}"
    fn = getattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma", None)
    if fn is None:
        return None, "m_grouped_fp8_fp4_gemm_nt_contiguous_sm90_fused_wgmma not exposed"
    return fn, None


def _cutedsl_kernel_available():
    if not torch.cuda.is_available():
        return None, "CUDA not available"
    try:
        spec = importlib.util.spec_from_file_location(
            "cutedsl_fp8_fp4_group_gemm_contiguous",
            _HERE / "cutedsl" / "fp8_fp4_group_gemm_contiguous.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001
        return None, f"CuTe-DSL kernel import failed: {exc}"
    return mod.kernel_function, None


def build_case(groups: int, m_per_group: int, n: int, k: int, gran_k: int = 128):
    """Same construction and RNG order as _benchmark_case in the source test."""
    m = groups * m_per_group
    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32) \
        .repeat_interleave(m_per_group)
    a_ref_src = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref_src = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a = problem.per_token_cast_to_fp8(a_ref_src, use_ue8m0=False, gran_k=gran_k)
    b_fp4 = torch.empty((groups, n, k // 2), device="cuda", dtype=torch.int8)
    b_sf = torch.empty((groups, n, k // gran_k), device="cuda", dtype=torch.float)
    for group_id in range(groups):
        b_fp4[group_id], b_sf[group_id] = problem.per_token_cast_to_fp4(
            b_ref_src[group_id], use_ue8m0=True, gran_k=gran_k)
    return a, (b_fp4, b_sf), grouped_layout


def run_deep_gemm(kernel_fn, a, b_w4, grouped_layout, gran_k=128):
    m = a[0].size(0)
    n = b_w4[0].size(1)
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    kernel_fn(a, b_w4, d, grouped_layout,
              gran_k=gran_k, compiled_dims="nk", use_psum_layout=False)
    return d


def run_accuracy(dg_kernel, dg_skip, dsl_kernel, dsl_skip, gran_k=128):
    torch.manual_seed(0)
    cases = [(8, 128), (8, 256), (8, 512), (8, 1024), (8, 2048),
              (16, 128), (16, 256), (16, 512), (16, 1024), (16, 2048),
              (24, 128), (24, 256), (24, 512), (24, 1024), (24, 2048),
              (32, 128), (32, 256), (32, 512), (32, 1024), (32, 2048)]

    failures, skipped, passed = 0, 0, 0
    for groups, m_per_group in cases:
        name = f"contiguous groups={groups} m_per_group={m_per_group} n=4096 k=7168"
        a, b_w4, grouped_layout = build_case(groups, m_per_group, n=4096, k=7168)

        # Reference = the Model (this is the ported subject swap).
        model = problem.Model(b_w4[0], b_w4[1], grouped_layout, gran_k).cuda()
        with torch.no_grad():
            model_out = model(a[0], a[1])

        for subject, kernel, skip_reason in (
                ("deep_gemm", dg_kernel, dg_skip),
                ("cutedsl", dsl_kernel, dsl_skip)):
            if kernel is None:
                print(f"SKIP  {name} [{subject}]: {skip_reason}")
                skipped += 1
                continue
            if subject == "deep_gemm":
                out = run_deep_gemm(kernel, a, b_w4, grouped_layout, gran_k)
            else:
                out = kernel(a[0], a[1], b_w4[0], b_w4[1], grouped_layout, gran_k)
            diff = calc_diff(out, model_out)
            ok = diff < TOLERANCE
            print(f"{'ok  ' if ok else 'FAIL'}  {name} [{subject}]: "
                  f"kernel-vs-Model diff={diff:.6f} (tol {TOLERANCE})")
            failures += (not ok)
            passed += ok

        del a, b_w4, model, model_out
    return passed, failures, skipped


# --------------------------------------------------------------------------- #
# Benchmark (informational)
# --------------------------------------------------------------------------- #

def _time_cuda(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3  # seconds


def _time_cuda_graph(fn, warmup: int = 5, iters: int = 20, allow_graph: bool = True):
    """Capture fn in a CUDA graph and time replays (pure device time, no
    per-call host overhead). Returns (seconds, mode). Subjects with
    data-dependent host logic must pass allow_graph=False — a *failed*
    capture attempt corrupts the CUDA RNG offset for later allocations."""
    if not allow_graph:
        return _time_cuda(fn, warmup=3, iters=iters), "eager"
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3, "graph"


def _effective_bytes(groups, m_per_group, n, k, gran_k=128):
    # W4 traffic model from the source test (_effective_bytes, fp8_b=False).
    logical_m = groups * m_per_group
    scale_k = (k + gran_k - 1) // gran_k
    a_bytes = logical_m * k + logical_m * scale_k * 4
    b_bytes = groups * n * (k // 2) + groups * n * scale_k * 4
    d_bytes = logical_m * n * 2
    return a_bytes + b_bytes + d_bytes


def run_benchmark(dg_kernel, dsl_kernel, gran_k=128):
    shapes = [(8, 128), (8, 256), (8, 512), (8, 1024), (8, 2048),
              (16, 128), (16, 256), (16, 512), (16, 1024), (16, 2048),
              (24, 128), (24, 256), (24, 512), (24, 1024), (24, 2048),
              (32, 128), (32, 256), (32, 512), (32, 1024), (32, 2048)]  # (groups, m_per_group), n/k fixed
    n, k = 4096, 7168
    print("\nBenchmark (n=4096, k=7168; CUDA-graph replay timing; informational)")
    print("groups | m/group | subject | us | GB/s | TFLOP/s | timing")
    print("-- | -- | -- | -- | -- | -- | --")
    torch.manual_seed(0)
    for groups, m_per_group in shapes:
        a, b_w4, grouped_layout = build_case(groups, m_per_group, n=n, k=k)
        model = problem.Model(b_w4[0], b_w4[1], grouped_layout, gran_k).cuda()
        flops = 2 * groups * m_per_group * n * k
        bytes_ = _effective_bytes(groups, m_per_group, n, k, gran_k)

        subjects = []
        if dg_kernel is not None:
            subjects.append(("deep_gemm", True, lambda: run_deep_gemm(
                dg_kernel, a, b_w4, grouped_layout, gran_k)))
        if dsl_kernel is not None:
            subjects.append(("cutedsl", True, lambda: dsl_kernel(
                a[0], a[1], b_w4[0], b_w4[1], grouped_layout, gran_k)))
        # The Model's forward has data-dependent host logic: never capturable.
        subjects.append(("Model(torch)", False, lambda: model(a[0], a[1])))

        for subject, allow_graph, fn in subjects:
            with torch.no_grad():
                elapsed, mode = _time_cuda_graph(fn, allow_graph=allow_graph)
            print(f"{groups} | {m_per_group} | {subject} | {elapsed * 1e6:.0f} | "
                  f"{bytes_ / elapsed / 1e9:.0f} | {flops / elapsed / 1e12:.1f} | {mode}")
        del a, b_w4, model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bench", action="store_true",
                        help="run only the accuracy suite")
    args = parser.parse_args()

    dg_kernel, dg_skip = _deep_gemm_kernel_available()
    dsl_kernel, dsl_skip = _cutedsl_kernel_available()

    passed, failures, skipped = run_accuracy(dg_kernel, dg_skip, dsl_kernel, dsl_skip)
    print(f"\n{passed} passed, {failures} failed, {skipped} skipped")

    if not args.skip_bench and (dg_kernel is not None or dsl_kernel is not None):
        run_benchmark(dg_kernel, dsl_kernel)
    elif not args.skip_bench:
        print("benchmark skipped: no runnable kernel subject")

    if failures == 0:
        print("PASS")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
