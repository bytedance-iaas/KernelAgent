"""Auto-generated from problem.md by scripts/problem_md.py - do not edit.

KernelBench-format problem: `Model` wraps the reference `run(...)`;
`get_inputs()` builds the canonical (first) workload; `WORKLOADS` +
`build_workload_inputs(i)` expose every workload for the test harness.
"""

import json

import torch
import torch.nn as nn

PROBLEM_NAME = '011_fp8_moe_gate_routing'
CUSTOM_INPUTS_ENTRYPOINT = 'get_inputs'
BLOB_ROOT = None   # override with env SOLBENCH_BLOB_ROOT
AXES = json.loads(r"""{
    "num_tokens": {
        "type": "var",
        "value": null,
        "description": "Number of tokens (batch_size * seq_len)"
    },
    "hidden_size": {
        "type": "const",
        "value": 7168,
        "description": "Hidden dimension size"
    },
    "n_routed_experts": {
        "type": "const",
        "value": 256,
        "description": "Total number of routed experts"
    },
    "num_experts_per_tok": {
        "type": "const",
        "value": 8,
        "description": "Number of experts selected per token"
    },
    "n_group": {
        "type": "const",
        "value": 8,
        "description": "Number of expert groups"
    },
    "topk_group": {
        "type": "const",
        "value": 4,
        "description": "Number of top groups to select"
    },
    "hidden_blocks": {
        "type": "expr",
        "value": "hidden_size // 128",
        "description": "Number of 128-element blocks in hidden dimension"
    },
    "expert_blocks": {
        "type": "expr",
        "value": "n_routed_experts // 128",
        "description": "Number of 128-element blocks in expert dimension"
    }
}""")
INPUT_SPECS = json.loads(r"""{
    "hidden_states": {
        "shape": [
            "num_tokens",
            "hidden_size"
        ],
        "dtype": "bfloat16",
        "description": "Input hidden states",
        "role": "input"
    },
    "weight": {
        "shape": [
            "n_routed_experts",
            "hidden_size"
        ],
        "dtype": "bfloat16",
        "description": "Gating projection weight matrix",
        "role": "input"
    },
    "e_score_correction_bias": {
        "shape": [
            "n_routed_experts"
        ],
        "dtype": "bfloat16",
        "description": "Score correction bias for noaux_tc routing",
        "role": "input"
    },
    "scale_x": {
        "shape": [
            "num_tokens",
            "hidden_blocks"
        ],
        "dtype": "float32",
        "description": "Blockwise scales for input activation (BlockWise1x128)",
        "role": "input"
    },
    "scale_w": {
        "shape": [
            "hidden_blocks",
            "expert_blocks"
        ],
        "dtype": "float32",
        "description": "Blockwise scales for weight (BlockWise128x128)",
        "role": "input"
    },
    "routed_scaling_factor": {
        "shape": null,
        "dtype": "float32",
        "description": "Scaling factor applied to final routing weights",
        "role": "scalar"
    }
}""")
WORKLOADS = [json.loads(line) for line in r"""
{"uuid": "88dd6f57-c3e9-5e68-b200-ee2d4f5706c6", "axes": {"num_tokens": 4352}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.3366, "sol": 0.0047}}}
{"uuid": "b511c68b-ebcc-52a5-b27c-106bfab45771", "axes": {"num_tokens": 5888}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.3872, "sol": 0.0062}}}
{"uuid": "ffbd91e2-38d4-5f22-91c9-2d7938be0e93", "axes": {"num_tokens": 1280}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2278, "sol": 0.0017}}}
{"uuid": "b351f3e4-04a0-5520-baf0-6c85bdbd8f0b", "axes": {"num_tokens": 768}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2082, "sol": 0.0012}}}
{"uuid": "69769b62-6836-5f1f-a101-c8e812ba785b", "axes": {"num_tokens": 4864}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.3522, "sol": 0.0052}}}
{"uuid": "1a0b4955-627d-581f-9af0-ccf615a3c24a", "axes": {"num_tokens": 1536}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2288, "sol": 0.002}}}
{"uuid": "fb92c9e3-9417-57af-95cc-26e3db1285b3", "axes": {"num_tokens": 2816}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2668, "sol": 0.0032}}}
{"uuid": "4e4b94c8-f89b-5a80-be4d-559cbd3d0054", "axes": {"num_tokens": 3328}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2859, "sol": 0.0037}}}
{"uuid": "b1f2ba64-383d-526e-8015-e458ce800be6", "axes": {"num_tokens": 2560}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2813, "sol": 0.003}}}
{"uuid": "f4e85037-6072-5ada-b0be-cba08978f3fa", "axes": {"num_tokens": 7168}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.4438, "sol": 0.0075}}}
{"uuid": "cb80f630-5166-5661-a86e-941d81a44b7d", "axes": {"num_tokens": 4608}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.3234, "sol": 0.005}}}
{"uuid": "87c7fa88-efc9-524f-9d1f-5adb3b62b426", "axes": {"num_tokens": 640}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2083, "sol": 0.0011}}}
{"uuid": "41bd6659-996c-534b-a88f-0ba45e6443ea", "axes": {"num_tokens": 6656}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.4043, "sol": 0.007}}}
{"uuid": "6c54d23b-50b1-5203-952d-c7bf0212c4c5", "axes": {"num_tokens": 256}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2029, "sol": 0.0007}}}
{"uuid": "9970a4e3-9d4e-578e-a19b-7ca1ae7935fc", "axes": {"num_tokens": 8192}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.5626, "sol": 0.0084}}}
{"uuid": "5bd75987-8af7-55bb-aae9-5867ff1b05e0", "axes": {"num_tokens": 1024}, "inputs": {"hidden_states": {"type": "custom"}, "weight": {"type": "custom"}, "e_score_correction_bias": {"type": "custom"}, "scale_x": {"type": "custom"}, "scale_w": {"type": "custom"}, "routed_scaling_factor": {"type": "custom"}}, "tolerance": {"max_atol": 0.005, "max_rtol": 0.02, "required_matched_ratio": 0.99}, "latency": {"b200": {"baseline": 0.2185, "sol": 0.0015}}}
""".strip().splitlines()]

_DTYPES = {
    "float32": torch.float32, "float": torch.float32,
    "float16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "float64": torch.float64,
    "float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2,
    "int8": torch.int8, "int16": torch.int16, "int32": torch.int32,
    "int64": torch.int64, "uint8": torch.uint8, "bool": torch.bool,
}

_INT_RANGES = {
    torch.int8: (-128, 128), torch.int16: (-1024, 1024),
    torch.int32: (-1024, 1024), torch.int64: (-1024, 1024),
    torch.uint8: (0, 256),
}


def _resolve_shape(shape, axis_values):
    return [d if isinstance(d, int) else int(axis_values[d]) for d in shape]


def _rand_tensor(shape, dtype, device):
    # Mirrors SOL-ExecBench core/bench/io.py:_rand_tensor
    if dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return torch.randn(shape, dtype=torch.float32, device=device) \
            .clamp_(-2.0, 2.0).to(dtype)
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    low, high = _INT_RANGES[dtype]
    return torch.randint(low, high, shape, dtype=dtype, device=device)




def _heuristic_tensor(name, shape, dtype, device, description=""):
    """Name-based input heuristics, ported from SOL-ExecBench
    core/bench/io.py:_generate_heuristic_tensor (same check order).
    Returns None when no heuristic applies."""
    import math
    if not dtype.is_floating_point or dtype in (
            torch.float8_e4m3fn, torch.float8_e5m2):
        return None
    desc = (description or "").lower()
    base = name.rstrip("0123456789")

    def _is_norm_param(suffix):
        if name == "norm" + suffix:
            return True
        if name.endswith(suffix):
            prefix = name[: -len(suffix)].rstrip("0123456789")
            return prefix.endswith(("norm", "layernorm"))
        return False

    if _is_norm_param("_weight"):
        return torch.ones(shape, dtype=dtype, device=device)
    if _is_norm_param("_bias"):
        return torch.zeros(shape, dtype=dtype, device=device)
    if (len(shape) >= 2 and shape[-1] == shape[-2]
            and (name in ("attention_mask", "causal_mask")
                 or ("attention mask" in desc and "causal" in desc))):
        mask = torch.zeros(shape, dtype=dtype, device=device)
        causal = torch.triu(torch.ones(shape[-1], shape[-1], device=device),
                            diagonal=1).bool()
        mask[..., causal] = torch.finfo(dtype).min
        return mask
    if name in ("x_mask", "text_mask", "aspect_ratio_mask", "drop_mask",
                "attention_mask") or (name.endswith("_mask") and "mask" in desc):
        return torch.randint(0, 2, shape, device=device).to(dtype)
    if name in ("cos", "sin", "cos_cached", "sin_cached", "rope_cos", "rope_sin"):
        t = torch.randn(shape, dtype=torch.float32, device=device)             .clamp_(-math.pi, math.pi)
        fn = torch.cos if name in ("cos", "cos_cached", "rope_cos") else torch.sin
        return fn(t).to(dtype)
    if (name in ("rstd", "std", "variance", "var")
            or name.endswith(("_rstd", "_std", "_variance", "_var"))
            or base in ("rstd", "std", "variance", "var")
            or base.endswith(("_rstd", "_std", "_var"))):
        return torch.randn(shape, dtype=dtype, device=device).abs() + 0.1
    if name in ("A", "A_log", "A_cumsum", "g"):
        if name == "A_cumsum":
            return torch.empty(shape, dtype=dtype, device=device)                 .uniform_(-0.1, 0.0).cumsum(dim=-1)
        if name == "A_log":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, -1.0)
        if name == "A":
            return torch.empty(shape, dtype=dtype, device=device).uniform_(-1.0, -0.001)
        return torch.empty(shape, dtype=dtype, device=device).uniform_(-5.0, 0.0)
    if name in ("attn_weights", "attention_weights", "routing_weights")             or "softmax" in desc:
        logits = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.softmax(logits, dim=-1).to(dtype)
    if len(shape) >= 2 and (name == "weight" or name.endswith(
            ("_weight", "_weights", "_proj", "_projs", "_proj_weight",
             "_proj_weights", "_weight_matrix"))):
        return torch.randn(shape, dtype=dtype, device=device)             / math.sqrt(shape[-1])
    return None


_ST_CACHE = {}


def _load_safetensor(rel_path: str, tensor_key: str, device: str):
    import os
    from safetensors.torch import load_file
    root = os.environ.get("SOLBENCH_BLOB_ROOT", BLOB_ROOT)
    if root is None:
        raise RuntimeError("safetensors input needs SOLBENCH_BLOB_ROOT")
    full = os.path.join(root, rel_path)
    if full not in _ST_CACHE:
        _ST_CACHE[full] = load_file(full)
    return _ST_CACHE[full][tensor_key].to(device)


def build_workload_inputs(workload_idx: int = 0, device: str = "cuda"):
    """Build the ordered input list for one workload (deterministic)."""
    wl = WORKLOADS[workload_idx]
    axis_values = dict(wl.get("axes", {}))
    for name, ax in AXES.items():
        if ax["type"] == "const":
            axis_values[name] = ax["value"]
    for name, ax in AXES.items():  # exprs resolve in declaration order
        if ax["type"] == "expr":
            axis_values[name] = int(eval(ax["value"], {"__builtins__": {}},
                                         dict(axis_values)))
    torch.manual_seed(workload_idx)

    # SOL-ExecBench custom inputs: fn(axes_and_scalars, device) -> dict
    custom = None
    if CUSTOM_INPUTS_ENTRYPOINT:
        scalars = {n: g["value"] for n, g in wl["inputs"].items()
                   if g.get("type") == "scalar"}
        custom = _CUSTOM_INPUTS_FN({**axis_values, **scalars},
                                    torch.device(device))

    args = []
    for name, spec in INPUT_SPECS.items():
        gen = wl["inputs"].get(name, {"type": "random"})
        kind = gen.get("type", "random")
        if kind == "custom":
            val = custom[name]
            args.append(val.to(device) if torch.is_tensor(val) else val)
            continue
        if kind == "safetensors":
            args.append(_load_safetensor(gen["path"], gen["tensor_key"], device))
            continue
        if spec["shape"] is None or kind == "scalar":
            args.append(gen.get("value"))
            continue
        shape = _resolve_shape(spec["shape"], axis_values)
        dtype = _DTYPES[spec["dtype"]]
        if kind == "random":
            t = _heuristic_tensor(name, tuple(shape), dtype, torch.device(device),
                                  spec.get("description"))
            args.append(t if t is not None else _rand_tensor(shape, dtype, device))
        elif kind == "zeros":
            args.append(torch.zeros(shape, dtype=dtype, device=device))
        else:
            raise ValueError(f"unsupported input generator: {gen}")
    return args


def workload_tolerance(workload_idx: int = 0):
    tol = WORKLOADS[workload_idx].get("tolerance", {})
    return float(tol.get("max_atol", 1e-2)), float(tol.get("max_rtol", 5e-2))


def gpu_key():
    """Normalized GPU key for latency-target lookup (e.g. 'h200', 'b200')."""
    name = torch.cuda.get_device_name(0).lower()
    for wl in WORKLOADS:
        for key in wl.get("latency", {}):
            if key.lower() in name:
                return key
    return None


def workload_latency(workload_idx: int = 0, key: str = None):
    """Latency spec for this workload on the given/current GPU, or None
    when the workload carries none for it. Returns a dict:
      baseline  - ms of the recorded baseline implementation
      target    - optional hard goal (ms): pass iff measured <= target
      sol       - optional speed-of-light estimate (ms, e.g. from SOLAR);
                  gated via the SOL-Score S = 1/(1+(t-sol)/(baseline-sol))
                  (1.0 at SOL, 0.5 at baseline)
      min_score - SOL-Score needed to pass when gating on `sol`
                  (default 0.5 = match the baseline)"""
    key = key or gpu_key()
    if key is None:
        return None
    spec = WORKLOADS[workload_idx].get("latency", {}).get(key)
    if spec is None:
        return None
    return dict(baseline=float(spec["baseline"]),
                target=float(spec["target"]) if "target" in spec else None,
                sol=float(spec["sol"]) if "sol" in spec else None,
                min_score=float(spec.get("min_score", 0.5)))


# ---- reference implementation (verbatim from problem.md) ---- #

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

# Snapshot the custom-inputs entrypoint BEFORE the KernelBench wrappers
# below shadow names like `get_inputs`.
_CUSTOM_INPUTS_FN = (globals()[CUSTOM_INPUTS_ENTRYPOINT]
                     if CUSTOM_INPUTS_ENTRYPOINT else None)


class Model(nn.Module):
    """Reference model: forwards straight to run()."""

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return run(*args)


def get_init_inputs():
    return []


def get_inputs():
    return build_workload_inputs(0)
