---
name: 011_fp8_moe_gate_routing
hf_id: deepseek-ai/DeepSeek-R1-0528
custom_inputs_entrypoint: get_inputs
---

# 011_fp8_moe_gate_routing

FP8-quantized MoE gating mechanism with top-k expert selection using noaux_tc method. Computes expert scores via FP8 linear projection, applies sigmoid activation, adds correction bias, performs group-based top-k selection across 256 experts with 8 groups and 4 top groups. Returns top-8 expert indices and normalized weights per token.

## Axes

| axis | type | value | description |
|------|------|-------|-------------|
| num_tokens | var | - | Number of tokens (batch_size * seq_len) |
| hidden_size | const | 7168 | Hidden dimension size |
| n_routed_experts | const | 256 | Total number of routed experts |
| num_experts_per_tok | const | 8 | Number of experts selected per token |
| n_group | const | 8 | Number of expert groups |
| topk_group | const | 4 | Number of top groups to select |
| hidden_blocks | expr | hidden_size // 128 | Number of 128-element blocks in hidden dimension |
| expert_blocks | expr | n_routed_experts // 128 | Number of 128-element blocks in expert dimension |

## Inputs

| name | shape | dtype | role | description |
|------|-------|-------|------|-------------|
| hidden_states | [num_tokens, hidden_size] | bfloat16 | input | Input hidden states |
| weight | [n_routed_experts, hidden_size] | bfloat16 | input | Gating projection weight matrix |
| e_score_correction_bias | [n_routed_experts] | bfloat16 | input | Score correction bias for noaux_tc routing |
| scale_x | [num_tokens, hidden_blocks] | float32 | input | Blockwise scales for input activation (BlockWise1x128) |
| scale_w | [hidden_blocks, expert_blocks] | float32 | input | Blockwise scales for weight (BlockWise128x128) |
| routed_scaling_factor | scalar | float32 | scalar | Scaling factor applied to final routing weights |

## Outputs

| name | shape | dtype | description |
|------|-------|-------|-------------|
| topk_idx | [num_tokens, num_experts_per_tok] | int64 | Selected expert indices per token |
| topk_weight | [num_tokens, num_experts_per_tok] | float32 | Normalized expert weights per token |

## Reference

```python
import torch
# --- inlined fp8_reference ---
import torch

from enum import StrEnum


class ScalingType(StrEnum):
    """
    Enum for different FP8 scaling strategies.

    Scaling types:
    - TensorWise: Global per-tensor scaling (no blocks)
    - RowWise: Per-row scaling (1 scale per row)
    - BlockWise1x16: 1x16 blocks (per-tensor in M, 16-sized blocks in K)
    - BlockWise1x32: 1x32 blocks (per-tensor in M, 32-sized blocks in K)
    - BlockWise1x128: 1x128 blocks (per-tensor in M, 128-sized blocks in K)
    - BlockWise128x128: 128x128 blocks (blockwise in both dimensions)
    """

    TensorWise = "TensorWise"
    RowWise = "RowWise"
    BlockWise1x16 = "BlockWise1x16"
    BlockWise1x32 = "BlockWise1x32"
    BlockWise1x128 = "BlockWise1x128"
    BlockWise128x128 = "BlockWise128x128"

    @property
    def shape(self) -> tuple[int, int]:
        return {
            ScalingType.TensorWise: (None, None),
            ScalingType.RowWise: (1, None),
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }[self]


class BlockwiseScaler:
    """
    Compute and apply scales for FP8 tensors.

    Supports various scaling strategies via ScalingType enum:
    - TensorWise: Global per-tensor scaling
    - RowWise: Per-row scaling
    - BlockWise1x16/32/128: Rectangular blocks
    - BlockWise128x128: Square blocks
    """

    E4M3_MAX = 448.0  # Maximum representable value in E4M3

    def __init__(self, scaling_type: ScalingType):
        """
        Initialize BlockwiseScaler with a specific scaling strategy.

        Args:
            scaling_type: ScalingType enum value
                Examples:
                - ScalingType.TensorWise -> global per-tensor scaling
                - ScalingType.RowWise -> per-row scaling (1 scale per row)
                - ScalingType.BlockWise1x128 -> 1x128 blocks
                - ScalingType.BlockWise128x128 -> 128x128 blocks
        """
        self.scaling_type = scaling_type
        self.shape = self.scaling_type.shape

        # Map enum to block dimensions (M, K)
        scaling_map = {
            ScalingType.TensorWise: (None, None),  # No blocking
            ScalingType.RowWise: (1, None),  # Per-row, full K dimension
            ScalingType.BlockWise1x16: (1, 16),
            ScalingType.BlockWise1x32: (1, 32),
            ScalingType.BlockWise1x128: (1, 128),
            ScalingType.BlockWise128x128: (128, 128),
        }

        self.block_size_m, self.block_size_k = scaling_map[scaling_type]

        # Keep for backward compatibility (use first dimension if available)
        self.block_size = self.block_size_m if self.block_size_m else None

    def compute_scales(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute scale factors based on the scaling type.

        This is a unified method that handles all scaling types:
        - TensorWise: Returns scalar tensor
        - RowWise: Returns (M,) tensor
        - BlockWise*: Returns (M//block_size_m, K//block_size_k) tensor

        Args:
            tensor: Input tensor (typically M, K for 2D)

        Returns:
            Scale tensor with shape depending on scaling type.
            These are inverse scales (amax / dtype_max) used for dequantization.
        """
        if self.scaling_type == ScalingType.TensorWise:
            # Global per-tensor scaling
            amax = torch.max(torch.abs(tensor)).clamp(min=1e-12)
            return amax / self.E4M3_MAX

        M, K = tensor.shape

        if self.scaling_type == ScalingType.RowWise:
            # Per-row scaling: (M, K) -> (M,)
            row_max = tensor.abs().amax(dim=1)
            scales = row_max / self.E4M3_MAX
            return torch.clamp(scales, min=1e-12)

        # BlockWise scaling
        assert M % self.block_size_m == 0, (
            f"M={M} must be a multiple of {self.block_size_m}"
        )
        assert K % self.block_size_k == 0, (
            f"K={K} must be a multiple of {self.block_size_k}"
        )

        # Reshape (M, K) -> (M//block_size_m, block_size_m, K//block_size_k, block_size_k)
        new_shape = (
            M // self.block_size_m,
            self.block_size_m,
            K // self.block_size_k,
            self.block_size_k,
        )
        tensor_blocked = tensor.reshape(new_shape)

        # Compute max over the block dimensions (dims 1 and 3)
        block_max = tensor_blocked.abs().amax(dim=3).amax(dim=1)

        # Compute inverse scales
        scales = block_max / self.E4M3_MAX
        return torch.clamp(scales, min=1e-12)

    def apply_scaling(
        self,
        tensor: torch.Tensor,
        scales: torch.Tensor,
        inverse: bool = False,
        clamp_to_fp8_range: bool = False,
    ) -> torch.Tensor:
        """
        Apply scaling to tensor based on the scaling type.

        This is a unified method that handles all scaling types:
        - TensorWise: Uses scalar scale
        - RowWise: Uses per-row scales (M,)
        - BlockWise*: Uses blockwise scales (M//block_size_m, K//block_size_k)

        Args:
            tensor: Input tensor (typically M, K for 2D)
            scales: Scale tensor with shape depending on scaling type
                   These are inverse scales (amax / dtype_max)
            inverse: If True, multiply by scales (dequantization)
                    If False, divide by scales (quantization)
            clamp_to_fp8_range: If True, clamp to FP8 range before returning

        Returns:
            Scaled tensor (same shape as input)
        """
        old_shape = tensor.shape
        if self.scaling_type == ScalingType.RowWise:
            # expand (M,) -> (M, 1)
            scales = scales.unsqueeze(1)
        elif self.scaling_type != ScalingType.TensorWise:
            # blockwise scaling
            M, K = tensor.shape
            new_shape = (
                M // self.block_size_m,
                self.block_size_m,
                K // self.block_size_k,
                self.block_size_k,
            )
            tensor = tensor.reshape(new_shape)
            scales = scales.unsqueeze(1).unsqueeze(3)

        if inverse:
            tensor_scaled = tensor * scales
        else:
            tensor_scaled = tensor / scales
            if clamp_to_fp8_range:
                tensor_scaled = torch.clamp(
                    tensor_scaled, min=-self.E4M3_MAX, max=self.E4M3_MAX
                )

        return tensor_scaled.reshape(*old_shape)


class CuBLASRefBlockwiseGemm:
    """
    Reference implementation of blockwise-scaled GEMM via dequantize-then-matmul.
    """

    def scaled_mm(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_recipe_a: ScalingType,
        scale_b: torch.Tensor,
        scale_recipe_b: ScalingType,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.bfloat16,
        use_fast_accum: bool = True,
    ) -> torch.Tensor:
        """
        Scaled matrix multiplication: dequantize A and B, then matmul in float32.

        Args:
            mat_a: Input matrix A (M, K) in float8_e4m3fn
            mat_b: Input matrix B (N, K) in float8_e4m3fn
            scale_a: Scaling factors for A
            scale_recipe_a: Scaling type for A
            scale_b: Scaling factors for B
            scale_recipe_b: Scaling type for B
            bias: Optional bias vector (N,)
            output_dtype: Output data type
            use_fast_accum: Unused (kept for API compatibility)

        Returns:
            Result matrix (M, N) with dtype=output_dtype
        """
        scaler_a = BlockwiseScaler(scale_recipe_a)
        scaler_b = BlockwiseScaler(scale_recipe_b)

        # Dequantize: FP8 values * inverse_scales -> float32
        a_f32 = scaler_a.apply_scaling(mat_a.to(torch.float32), scale_a, inverse=True)
        b_f32 = scaler_b.apply_scaling(mat_b.to(torch.float32), scale_b, inverse=True)

        # Single matmul in float32
        y = a_f32 @ b_f32.T

        if bias is not None and bias.numel():
            y = y + bias

        return y.to(output_dtype)

# --- end inlined fp8_reference ---



def get_inputs(
    axes_and_scalars: dict[str, ...], device: torch.device
) -> dict[str, torch.Tensor]:
    """Generate inputs with proper FP8 scales."""
    num_tokens = axes_and_scalars["num_tokens"]
    hidden_size = 7168
    n_routed_experts = 256
    
    # Generate random tensors
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(n_routed_experts, hidden_size, dtype=torch.bfloat16, device=device)
    e_score_correction_bias = torch.randn(n_routed_experts, dtype=torch.bfloat16, device=device) * 0.1
    
    # Compute FP8 scales
    hidden_states_fp32 = hidden_states.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    
    scale_x = activation_scaler.compute_scales(hidden_states_fp32)
    
    # Transpose weight to (K, N) for blockwise scaling
    weight_t = weight_fp32.T  # (7168, 256)
    scale_w = weight_scaler.compute_scales(weight_t)
    
    return {
        "hidden_states": hidden_states,
        "weight": weight,
        "e_score_correction_bias": e_score_correction_bias,
        "scale_x": scale_x,
        "scale_w": scale_w,
        "routed_scaling_factor": 2.5,
    }


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    routed_scaling_factor: float,
):
    """
    FP8-quantized MoE gating with top-k expert selection.
    
    Steps:
    1. FP8 GEMM for gating scores: hidden_states @ weight.T
    2. Sigmoid activation on scores
    3. Add score correction bias for noaux_tc method
    4. Group-based top-k selection (8 groups, select top 4 groups)
    5. Final top-k expert selection (8 experts per token)
    6. Score normalization and scaling
    """
    # Constants
    n_routed_experts = 256
    num_experts_per_tok = 8
    n_group = 8
    topk_group = 4
    
    bsz_seq_len = hidden_states.shape[0]
    
    # FP8 scaling infrastructure
    activation_scaler = BlockwiseScaler(ScalingType.BlockWise1x128)
    weight_scaler = BlockwiseScaler(ScalingType.BlockWise128x128)
    gemm_ref = CuBLASRefBlockwiseGemm()
    
    # Step 1: FP8 GEMM for gating scores
    hidden_states_fp32 = hidden_states.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    
    # Apply scaling before quantization
    x_scaled = activation_scaler.apply_scaling(
        hidden_states_fp32, scale_x, inverse=False, clamp_to_fp8_range=True
    )
    
    # Transpose weight to (K, N) for blockwise scaling
    weight_t = weight_fp32.T  # (7168, 256)
    w_scaled = weight_scaler.apply_scaling(
        weight_t, scale_w, inverse=False, clamp_to_fp8_range=True
    )
    
    # Quantize to FP8
    qx = x_scaled.to(torch.float8_e4m3fn)  # [bsz_seq_len, 7168]
    qw = w_scaled.T.to(torch.float8_e4m3fn)  # [256, 7168]
    
    # Transpose weight scales for CuBLAS format
    scale_w_cublas = scale_w.T.contiguous()  # [N//128, K//128]
    
    # FP8 GEMM: [bsz_seq_len, 7168] @ [256, 7168].T -> [bsz_seq_len, 256]
    logits = gemm_ref.scaled_mm(
        mat_a=qx,
        mat_b=qw,
        scale_a=scale_x,
        scale_recipe_a=ScalingType.BlockWise1x128,
        scale_b=scale_w_cublas,
        scale_recipe_b=ScalingType.BlockWise128x128,
        bias=None,
        output_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    
    # Step 2: Sigmoid activation
    scores = torch.sigmoid(logits.to(torch.float32))
    
    # Step 3: Add score correction bias for noaux_tc method
    scores_for_choice = scores + e_score_correction_bias.to(torch.float32).unsqueeze(0)
    
    # Step 4: Group-based top-k selection
    experts_per_group = n_routed_experts // n_group  # 32
    group_scores_reshaped = scores_for_choice.view(
        bsz_seq_len, n_group, experts_per_group
    )
    
    # Select top-2 experts per group and sum their scores
    group_scores = group_scores_reshaped.topk(2, dim=-1)[0].sum(dim=-1)
    
    # Select top-4 groups out of 8
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    
    # Create group mask
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    
    # Expand mask to expert level
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(bsz_seq_len, n_group, experts_per_group)
        .reshape(bsz_seq_len, n_routed_experts)
    )
    
    # Step 5: Mask out non-selected groups and perform final top-k
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    _, topk_idx = torch.topk(tmp_scores, k=num_experts_per_tok, dim=-1, sorted=False)
    
    # Step 6: Gather final weights and normalize
    topk_weight = scores.gather(1, topk_idx)
    
    # Normalize weights
    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
    topk_weight = topk_weight / denominator
    
    # Apply routing scaling factor
    topk_weight = topk_weight * routed_scaling_factor
    
    return topk_idx, topk_weight
```

## Workloads

```jsonl
{"uuid": "88dd6f57-c3e9-5e68-b200-ee2d4f5706c6", "axes": {"num_tokens": 4352}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.3366, "sol": 0.0047}}}
{"uuid": "b511c68b-ebcc-52a5-b27c-106bfab45771", "axes": {"num_tokens": 5888}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.3872, "sol": 0.0062}}}
{"uuid": "ffbd91e2-38d4-5f22-91c9-2d7938be0e93", "axes": {"num_tokens": 1280}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2278, "sol": 0.0017}}}
{"uuid": "b351f3e4-04a0-5520-baf0-6c85bdbd8f0b", "axes": {"num_tokens": 768}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2082, "sol": 0.0012}}}
{"uuid": "69769b62-6836-5f1f-a101-c8e812ba785b", "axes": {"num_tokens": 4864}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.3522, "sol": 0.0052}}}
{"uuid": "1a0b4955-627d-581f-9af0-ccf615a3c24a", "axes": {"num_tokens": 1536}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2288, "sol": 0.0020}}}
{"uuid": "fb92c9e3-9417-57af-95cc-26e3db1285b3", "axes": {"num_tokens": 2816}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2668, "sol": 0.0032}}}
{"uuid": "4e4b94c8-f89b-5a80-be4d-559cbd3d0054", "axes": {"num_tokens": 3328}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2859, "sol": 0.0037}}}
{"uuid": "b1f2ba64-383d-526e-8015-e458ce800be6", "axes": {"num_tokens": 2560}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2813, "sol": 0.0030}}}
{"uuid": "f4e85037-6072-5ada-b0be-cba08978f3fa", "axes": {"num_tokens": 7168}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.4438, "sol": 0.0075}}}
{"uuid": "cb80f630-5166-5661-a86e-941d81a44b7d", "axes": {"num_tokens": 4608}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.3234, "sol": 0.0050}}}
{"uuid": "87c7fa88-efc9-524f-9d1f-5adb3b62b426", "axes": {"num_tokens": 640}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2083, "sol": 0.0011}}}
{"uuid": "41bd6659-996c-534b-a88f-0ba45e6443ea", "axes": {"num_tokens": 6656}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.4043, "sol": 0.0070}}}
{"uuid": "6c54d23b-50b1-5203-952d-c7bf0212c4c5", "axes": {"num_tokens": 256}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2029, "sol": 0.0007}}}
{"uuid": "9970a4e3-9d4e-578e-a19b-7ca1ae7935fc", "axes": {"num_tokens": 8192}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.5626, "sol": 0.0084}}}
{"uuid": "5bd75987-8af7-55bb-aae9-5867ff1b05e0", "axes": {"num_tokens": 1024}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2185, "sol": 0.0015}}}
```
