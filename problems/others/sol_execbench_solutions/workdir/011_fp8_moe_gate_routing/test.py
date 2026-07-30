# test.py for SOL-ExecBench kernel 187: 011_fp8_moe_gate_routing
import sys
import torch
from kernel import kernel_function
from problem import get_inputs, get_init_inputs, Model


def test_kernel():
    device = "cuda"
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
    model = Model(*get_init_inputs()).to(device)

    with torch.no_grad():
        ref_idx, ref_weight = model(*inputs)

    kernel_out = kernel_function(*inputs)
    if isinstance(kernel_out, (list, tuple)):
        ker_idx, ker_weight = kernel_out[0], kernel_out[1]
    else:
        print("FAIL: kernel_function must return (topk_idx, topk_weight)")
        return False

    # topk_idx: compare sorted expert sets per token (order may differ)
    ref_sorted = ref_idx.sort(dim=-1).values
    ker_sorted = ker_idx.sort(dim=-1).values
    if not torch.equal(ref_sorted, ker_sorted):
        mismatch = (ref_sorted != ker_sorted).float().mean().item()
        print(f"FAIL topk_idx: {mismatch:.2%} tokens have different expert sets")
        return False

    # topk_weight: gather in sorted index order for fair comparison
    ref_w_sorted = ref_weight.gather(1, ref_idx.argsort(dim=-1))
    ker_w_sorted = ker_weight.gather(1, ker_idx.argsort(dim=-1))
    if not torch.allclose(ref_w_sorted.float(), ker_w_sorted.float(), atol=1e-2, rtol=1e-2):
        max_err = (ref_w_sorted.float() - ker_w_sorted.float()).abs().max().item()
        print(f"FAIL topk_weight: max_err={max_err:.4f}")
        return False

    print("PASS")
    return True


if __name__ == "__main__":
    success = test_kernel()
    sys.exit(0 if success else 1)
