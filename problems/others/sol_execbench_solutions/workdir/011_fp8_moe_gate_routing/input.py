# Initial kernel for SOL-ExecBench kernel 187: 011_fp8_moe_gate_routing
# https://research.nvidia.com/benchmarks/sol-execbench/kernel/187
#
# FP8-quantized MoE gating with top-k expert selection (noaux_tc method).
# Constants: hidden_size=7168, n_routed_experts=256, num_experts_per_tok=8,
#            n_group=8, topk_group=4
# Inputs:  hidden_states[T,7168] bf16, weight[256,7168] bf16,
#          e_score_correction_bias[256] bf16,
#          scale_x[T,56] f32, scale_w[56,2] f32, routed_scaling_factor f32
# Outputs: topk_idx[T,8] int64, topk_weight[T,8] f32
# Target:  SOL score > 1.0 on NVIDIA B200

import torch
from enum import StrEnum


class ScalingType(StrEnum):
    TensorWise = "TensorWise"
    RowWise = "RowWise"
    BlockWise1x16 = "BlockWise1x16"
    BlockWise1x32 = "BlockWise1x32"
    BlockWise1x128 = "BlockWise1x128"
    BlockWise128x128 = "BlockWise128x128"

    @property
    def shape(self):
        return {
            ScalingType.TensorWise: (None, None),
            ScalingType.RowWise: (1, None),
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }[self]


class BlockwiseScaler:
    E4M3_MAX = 448.0

    def __init__(self, scaling_type):
        self.scaling_type = scaling_type
        scaling_map = {
            ScalingType.TensorWise: (None, None),
            ScalingType.RowWise: (1, None),
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }
        self.block_size_m, self.block_size_k = scaling_map[scaling_type]

    def apply_scaling(self, tensor, scales, inverse=False, clamp_to_fp8_range=False):
        old_shape = tensor.shape
        if self.scaling_type == ScalingType.RowWise:
            scales = scales.unsqueeze(1)
        elif self.scaling_type != ScalingType.TensorWise:
            M, K = tensor.shape
            new_shape = (M // self.block_size_m, self.block_size_m,
                         K // self.block_size_k, self.block_size_k)
            tensor = tensor.reshape(new_shape)
            scales = scales.unsqueeze(1).unsqueeze(3)
        if inverse:
            tensor_scaled = tensor * scales
        else:
            tensor_scaled = tensor / scales
            if clamp_to_fp8_range:
                tensor_scaled = torch.clamp(tensor_scaled, min=-self.E4M3_MAX, max=self.E4M3_MAX)
        return tensor_scaled.reshape(*old_shape)


class CuBLASRefBlockwiseGemm:
    def scaled_mm(self, mat_a, mat_b, scale_a, scale_recipe_a,
                  scale_b, scale_recipe_b, bias=None,
                  output_dtype=torch.bfloat16, use_fast_accum=True):
        scaler_a = BlockwiseScaler(scale_recipe_a)
        scaler_b = BlockwiseScaler(scale_recipe_b)
        a_f32 = scaler_a.apply_scaling(mat_a.to(torch.float32), scale_a, inverse=True)
        b_f32 = scaler_b.apply_scaling(mat_b.to(torch.float32), scale_b, inverse=True)
        y = a_f32 @ b_f32.T
        if bias is not None and bias.numel():
            y = y + bias
        return y.to(output_dtype)


@torch.no_grad()
def kernel_function(hidden_states, weight, e_score_correction_bias,
                    scale_x, scale_w, routed_scaling_factor):
    """FP8 MoE gate routing — standalone PyTorch baseline."""
    n_routed_experts = 256
    num_experts_per_tok = 8
    n_group = 8
    topk_group = 4

    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()

    bsz_seq_len = hidden_states.shape[0]

    x_scaled = activation_scaler.apply_scaling(
        hidden_states.to(torch.float32), scale_x, inverse=False, clamp_to_fp8_range=True)
    weight_t = weight.to(torch.float32).T
    w_scaled = weight_scaler.apply_scaling(
        weight_t, scale_w, inverse=False, clamp_to_fp8_range=True)

    qx = x_scaled.to(torch.float8_e4m3fn)
    qw = w_scaled.T.to(torch.float8_e4m3fn)
    scale_w_cublas = scale_w.T.contiguous()

    logits = gemm_ref.scaled_mm(
        mat_a=qx, mat_b=qw,
        scale_a=scale_x, scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_cublas, scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None, output_dtype=torch.bfloat16,
    )

    logits_f32 = logits.to(torch.float32)
    scores = 1.0 / (1.0 + torch.exp(-logits_f32))
    scores_for_choice = scores + e_score_correction_bias.to(torch.float32).unsqueeze(0)

    experts_per_group = n_routed_experts // n_group
    group_scores = scores_for_choice.view(bsz_seq_len, n_group, experts_per_group).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]

    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (group_mask.unsqueeze(-1)
                  .expand(bsz_seq_len, n_group, experts_per_group)
                  .reshape(bsz_seq_len, n_routed_experts))

    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_idx = torch.topk(tmp_scores, k=num_experts_per_tok, dim=-1, sorted=False)

    topk_weight = scores.gather(1, topk_idx)
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20) * routed_scaling_factor

    return topk_idx, topk_weight
