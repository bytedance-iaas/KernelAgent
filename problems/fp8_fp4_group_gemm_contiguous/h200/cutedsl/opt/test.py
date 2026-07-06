"""Correctness test for the CuTe-DSL W4A8 contiguous grouped GEMM candidate.

Imports `kernel_function` from kernel.py (workdir convention) and checks it
against the problem Model at the source-kernel accuracy contract
(calc_diff < 0.015) on a small case, an uneven-tile case, and the canonical
workload shape. Exits 0 on PASS.
"""

import sys

import torch

import problem
from kernel import kernel_function

TOLERANCE = 0.015


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def run_case(name, groups, m_per_group, n, k, gran_k=128):
    m = groups * m_per_group
    a_ref = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn(groups, n, k, device="cuda", dtype=torch.bfloat16)
    a_fp8, a_sf = problem.per_token_cast_to_fp8(a_ref, use_ue8m0=False, gran_k=gran_k)
    b_fp4 = torch.empty(groups, n, k // 2, device="cuda", dtype=torch.int8)
    b_sf = torch.empty(groups, n, k // gran_k, device="cuda", dtype=torch.float)
    for g in range(groups):
        b_fp4[g], b_sf[g] = problem.per_token_cast_to_fp4(
            b_ref[g], use_ue8m0=True, gran_k=gran_k)
    grouped_layout = torch.arange(groups, device="cuda", dtype=torch.int32) \
        .repeat_interleave(m_per_group)

    model = problem.Model(b_fp4, b_sf, grouped_layout, gran_k).cuda()
    with torch.no_grad():
        ref = model(a_fp8, a_sf)

    out = kernel_function(a_fp8, a_sf, b_fp4, b_sf, grouped_layout, gran_k)
    diff = calc_diff(out, ref)
    max_abs = (out.float() - ref.float()).abs().max().item()
    ok = diff < TOLERANCE
    print(f"{'ok  ' if ok else 'FAIL'} {name}: calc_diff={diff:.3e} max_abs={max_abs:.3e}")
    return ok


def main():
    torch.manual_seed(0)
    results = [
        run_case("small g2 m/g=8 n=128 k=256", 2, 8, 128, 256),
        run_case("uneven-tile g3 m/g=96 n=192 k=384", 3, 96, 192, 384),
        run_case("odd-rows g4 m/g=57 n=512 k=1024", 4, 57, 512, 1024),
        run_case("workload g8 m/g=128 n=4096 k=7168", 8, 128, 4096, 7168),
    ]
    if all(results):
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
