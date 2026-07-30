"""Auto-generated from problem.md by scripts/problem_md.py - do not edit.

Checks `kernel.kernel_function` against the reference over every workload
using SOL-ExecBench correctness semantics (core/bench/correctness.py):
an element matches when |out - ref| <= max_atol + max_rtol * |ref|; the
workload passes when matched_ratio >= required_matched_ratio (default
0.99), no nan/inf anywhere, output is not spuriously all-zero, and
max_abs_err <= max_error_cap when a cap is set. Exits 0 on PASS.
"""

import sys

import torch

import problem
from kernel import kernel_function


def check_one(out: torch.Tensor, ref: torch.Tensor, tol: dict):
    x, y = out.to(torch.float32), ref.to(torch.float32)
    if (~torch.isfinite(x)).any() or (~torch.isfinite(y)).any():
        return False, float("nan"), 0.0
    if y.abs().sum() > 0 and x.abs().sum() == 0:
        return False, float(y.abs().max()), 0.0
    abs_err = (x - y).abs()
    max_abs = float(abs_err.max())
    atol = float(tol.get("max_atol", 1e-2))
    rtol = float(tol.get("max_rtol", 1e-2))
    bound = atol + rtol * y.abs()
    matched = 1.0 - float((abs_err > bound).sum()) / abs_err.numel()
    ok = matched >= float(tol.get("required_matched_ratio", 0.99))
    cap = tol.get("max_error_cap")
    if cap is not None and max_abs > float(cap):
        ok = False
    return ok, max_abs, matched


def check(out, ref, tol: dict):
    """Handle single- and multi-output references uniformly."""
    outs = out if isinstance(out, (tuple, list)) else (out,)
    refs = ref if isinstance(ref, (tuple, list)) else (ref,)
    if len(outs) != len(refs):
        return False, float("nan"), 0.0
    ok, worst_err, worst_matched = True, 0.0, 1.0
    for o, r in zip(outs, refs):
        ok_i, err_i, matched_i = check_one(o, r, tol)
        ok &= ok_i
        worst_err = max(worst_err, err_i) if err_i == err_i else float("nan")
        worst_matched = min(worst_matched, matched_i)
    return ok, worst_err, worst_matched


def main() -> int:
    failures = 0
    for i, wl in enumerate(problem.WORKLOADS):
        args = problem.build_workload_inputs(i)
        model = problem.Model().cuda()
        with torch.no_grad():
            ref = model(*args)
            out = kernel_function(*args)
        ok, max_err, matched = check(out, ref, wl.get("tolerance", {}))
        axes = wl.get("axes", {})
        print(f"{'ok  ' if ok else 'FAIL'} workload[{i}] {axes}: "
              f"max_abs_err={max_err:.3e} matched={matched:.4f}")
        failures += (not ok)
        del args, model, ref, out
    print("PASS" if failures == 0 else f"{failures} FAILED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
