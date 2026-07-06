"""
Ported unit tests for fp8_fp4_group_gemm_masked (KernelBench problem).

Source: reference/cuda/sgl-DeepGEMM/tests/test_sm90_fp8_fp4.py
        :: test_sm90_fp8_fp4_masked            (seed 2, b_gran_k=32 sweep)
        :: test_sm90_fp8_fp4_masked_direct_fp32_scale (seed 3, b_gran_k=128)
        :: test_sm90_fp8_fp4_masked_skew_cases (seed 4) — the g24 skewed
           masked_m distributions; a representative subset is ported here
           (the full list is a performance benchmark; its accuracy metric is
           identical). Each subset case keeps the source's construction.

Port rules applied:
- Input construction preserved: same seeds, same randn order (a_ref_src then
  b_ref_src per case), same quantization calls, max_m=128 for the uniform
  sweeps and the source's max_m for skew cases. The FP8-baseline benchmark
  half of the original is dropped (no RNG impact, not the subject).
- Tolerance: the source's accuracy contract for the W4 kernel is
  `calc_diff < 0.015` (asserted in the contiguous test; present as the
  commented-out `assert w4_diff < 0.015` in _masked_benchmark_case).
- The original reference computation IS the Model; the original comparison
  `calc_diff(kernel_out[:valid_m], ref[:valid_m])` is ported as
  kernel-vs-Model on valid rows only (the kernel leaves invalid rows
  undefined; the Model zeroes them).
- Cases needing an SM90 GPU + deep_gemm are SKIPPED (with reason) when
  unavailable; never silently dropped.

Standalone: prints per-case results, prints PASS and exits 0 only if every
case passes.
"""

import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

_spec = importlib.util.spec_from_file_location(
    "fp8_fp4_group_gemm_masked",
    _HERE / "problem.py")
problem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(problem)

TOLERANCE = 0.015


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    # Mirror of deep_gemm.testing.calc_diff
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return float(1 - sim)


def _kernel_available():
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
    fn = getattr(deep_gemm, "m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma", None)
    if fn is None:
        return None, "m_grouped_fp8_fp4_gemm_nt_masked_sm90_fused_wgmma not exposed"
    return fn, None


def run_case(kernel_fn, groups: int, masked_m_values: list, expected_m: int,
             n: int, k: int, max_m: int, a_gran_k: int = 128,
             b_gran_k: int = 32, pass_hints: bool = False) -> float:
    masked_m = torch.tensor(masked_m_values, device="cuda", dtype=torch.int32)

    # Same construction and RNG order as the source test's case builders.
    a_ref_src = torch.randn((groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b_ref_src = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)

    a_data = torch.empty((groups, max_m, k), device="cuda", dtype=torch.float8_e4m3fn)
    a_sf = torch.empty((groups, max_m, k // a_gran_k), device="cuda", dtype=torch.float)
    b_fp4 = torch.empty((groups, n, k // 2), device="cuda", dtype=torch.int8)
    b_sf = torch.empty((groups, n, k // b_gran_k), device="cuda", dtype=torch.float)
    for group_id in range(groups):
        a_data[group_id], a_sf[group_id] = problem.per_token_cast_to_fp8(
            a_ref_src[group_id], use_ue8m0=False, gran_k=a_gran_k)
        b_fp4[group_id], b_sf[group_id] = problem.per_token_cast_to_fp4(
            b_ref_src[group_id], use_ue8m0=True, gran_k=b_gran_k)
    a = (a_data, a_sf)
    b_w4 = (b_fp4, b_sf)

    # Reference = the Model (ported subject swap).
    model = problem.Model(b_fp4, b_sf, a_gran_k, b_gran_k).cuda()
    with torch.no_grad():
        model_out = model(a_data, a_sf, masked_m)

    # Subject = the original CUDA kernel on identical inputs.
    d_w4 = torch.empty((groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    kwargs = dict(gran_k=a_gran_k, gran_k_a=a_gran_k, gran_k_b=b_gran_k)
    if pass_hints:
        kwargs["masked_m_max_hint"] = max(masked_m_values)
        kwargs["active_groups_hint"] = sum(1 for v in masked_m_values if v > 0)
    kernel_fn(a, b_w4, d_w4, masked_m, expected_m, **kwargs)

    # Compare valid rows only (kernel leaves rows >= masked_m[g] undefined).
    return max(
        calc_diff(d_w4[g, :valid_m], model_out[g, :valid_m]) if valid_m > 0 else 0.0
        for g, valid_m in enumerate(masked_m_values))


def main() -> int:
    kernel_fn, skip_reason = _kernel_available()
    results = []  # (name, diff or None, skipped)

    def run_suite(name_prefix, seed, case_list, **fixed):
        torch.manual_seed(seed)
        for case in case_list:
            name = f"{name_prefix} {case['name']}"
            if kernel_fn is None:
                print(f"SKIP  {name}: {skip_reason}")
                results.append((name, None, True))
                continue
            diff = run_case(kernel_fn, **{**fixed, **case["args"]})
            ok = diff < TOLERANCE
            print(f"{'ok  ' if ok else 'FAIL'}  {name}: kernel-vs-Model "
                  f"diff={diff:.6f} (tol {TOLERANCE})")
            results.append((name, diff, False))

    # --- test_sm90_fp8_fp4_masked (seed 2, b_gran_k=32, max_m=128) ---
    uniform_cases = []
    for groups, n, k in ((8, 4096, 7168), (8, 7168, 2048),
                         (16, 4096, 7168), (16, 7168, 2048),
                         (32, 4096, 7168), (32, 7168, 2048)):
        for m_per_group in (1, 4, 8, 16, 32):
            uniform_cases.append(dict(
                name=f"g{groups} m/g={m_per_group} n={n} k={k}",
                args=dict(groups=groups, masked_m_values=[m_per_group] * groups,
                          expected_m=m_per_group, n=n, k=k, max_m=128)))
    run_suite("masked[k32]", 2, uniform_cases, b_gran_k=32)

    # --- test_sm90_fp8_fp4_masked_direct_fp32_scale (seed 3, b_gran_k=128) ---
    fp32_cases = []
    for groups, n, k in ((8, 4096, 7168), (8, 7168, 2048),
                         (16, 4096, 7168), (16, 7168, 2048),
                         (32, 4096, 7168), (32, 7168, 2048)):
        for m_per_group in (1, 4, 8, 16, 32):
            fp32_cases.append(dict(
                name=f"g{groups} m/g={m_per_group} n={n} k={k}",
                args=dict(groups=groups, masked_m_values=[m_per_group] * groups,
                          expected_m=m_per_group, n=n, k=k, max_m=128)))
    run_suite("masked[k128]", 3, fp32_cases, b_gran_k=128)

    # --- test_sm90_fp8_fp4_masked_skew_cases (seed 4): representative subset
    # of the g24 distributions, preserving each case's construction. These
    # exercise the RS fast-path (BM32 skew), the small-M simple scheduler and
    # the hot/fan-out block selection with hints.
    def values_from_active(active_values, groups=24):
        return active_values + [0] * (groups - len(active_values))

    skew_cases = []
    for shape_name, n, k, max_m in (("g24_n6144_k7168", 6144, 7168, 4096),
                                    ("g24_n7168_k3072", 7168, 3072, 4096)):
        for dist_name, masked_m_values, expected_m in (
                ("uniform_17", [17] * 24, 17),
                ("one_hot_512", values_from_active([512]), 512),
                ("dense_tail_hot64", values_from_active([64, 32, 16, 8] + [4] * 20), 17)):
            skew_cases.append(dict(
                name=f"{shape_name}_{dist_name}",
                args=dict(groups=24, masked_m_values=masked_m_values,
                          expected_m=expected_m, n=n, k=k, max_m=max_m,
                          pass_hints=True)))
    run_suite("masked[skew]", 4, skew_cases, b_gran_k=32)

    failures = sum(1 for _, diff, skipped in results
                   if not skipped and diff >= TOLERANCE)
    skipped = sum(1 for _, _, s in results if s)
    passed = len(results) - failures - skipped
    print(f"\n{passed} passed, {failures} failed, {skipped} skipped")
    if failures == 0:
        print("PASS")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
