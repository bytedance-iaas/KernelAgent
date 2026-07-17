"""Test for the CuTeDSL mxfp8_fp8_group_gemm_contiguous kernel.

Runs every workload from problem.py: builds the deterministic quantized
inputs, computes the pure-PyTorch reference via problem.run (dequantize
UE8M0 scales then fp32 grouped matmul -> bf16), calls kernel_function on the
same tensors, and compares with the problem's workload tolerances
(|out - ref| <= atol + rtol*|ref| matched ratio >= 0.97, no NaN/Inf, not
all-zero) plus a cosine-style calc_diff < 0.03 gate (the DeepGEMM accuracy
contract). Exits 0 iff every workload passes.
"""

import sys
from pathlib import Path

import torch

# kernel.py sits next to this file; problem.py in the nearest ancestor dir
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_d = _HERE
for _ in range(4):
    if (_d / "problem.py").exists():
        break
    _d = _d.parent
sys.path.insert(0, str(_d))
import problem  # noqa: E402

from kernel import kernel_function  # noqa: E402

DIFF_TOL = 0.03  # DeepGEMM's own accuracy gate for this kernel


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def check_workload(i: int) -> bool:
    args = problem.build_workload_inputs(i, device="cuda")
    a_data, a_scale, b_data, b_scale, grouped_layout = args
    axes = problem.WORKLOADS[i]["axes"]
    atol, rtol = problem.workload_tolerance(i)
    tol = problem.WORKLOADS[i].get("tolerance", {})
    required_ratio = float(tol.get("required_matched_ratio", 0.99))

    ref = problem.run(a_data, a_scale, b_data, b_scale, grouped_layout)
    out = kernel_function(a_data, a_scale, b_data, b_scale, grouped_layout)
    torch.cuda.synchronize()

    ok = True
    msgs = []

    if out.shape != ref.shape or out.dtype != ref.dtype:
        msgs.append(f"shape/dtype mismatch: {tuple(out.shape)}/{out.dtype} "
                    f"vs {tuple(ref.shape)}/{ref.dtype}")
        ok = False
    if out.device != a_data.device:
        msgs.append(f"device mismatch: {out.device} vs {a_data.device}")
        ok = False

    if ok:
        out_f = out.float()
        ref_f = ref.float()
        if not torch.isfinite(out_f).all():
            msgs.append("non-finite values in output")
            ok = False
        if out_f.abs().max().item() == 0.0 and ref_f.abs().max().item() != 0.0:
            msgs.append("output is spuriously all-zero")
            ok = False

    if ok:
        err = (out_f - ref_f).abs()
        bound = atol + rtol * ref_f.abs()
        matched = (err <= bound).float().mean().item()
        diff = calc_diff(out, ref)
        max_abs = err.max().item()
        msgs.append(f"matched={matched:.4f} (need>={required_ratio}) "
                    f"calc_diff={diff:.6f} (need<{DIFF_TOL}) max_abs={max_abs:.4f}")
        if matched < required_ratio:
            # dump a few mismatches for debugging
            bad = (err > bound).nonzero()[:5]
            for idx in bad:
                r, c = int(idx[0]), int(idx[1])
                msgs.append(f"  mismatch [{r},{c}]: out={out_f[r, c].item():.6f} "
                            f"ref={ref_f[r, c].item():.6f}")
            ok = False
        if diff >= DIFF_TOL:
            ok = False

    print(f"{'ok  ' if ok else 'FAIL'} workload[{i}] {axes}: {'; '.join(msgs)}")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 1
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        print(f"kernel requires sm_90 (Hopper), got sm_{major}x")
        return 1

    all_ok = True
    for i in range(len(problem.WORKLOADS)):
        all_ok &= check_workload(i)

    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
