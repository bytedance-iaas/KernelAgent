"""Auto-generated from problem.md by scripts/problem_md.py - do not edit.

KernelBench-format problem: `Model` wraps the reference `run(...)`;
`get_inputs()` builds the canonical (first) workload; `WORKLOADS` +
`build_workload_inputs(i)` expose every workload for the test harness.
"""

import json

import torch
import torch.nn as nn

PROBLEM_NAME = '033_post_norm_residual'
CUSTOM_INPUTS_ENTRYPOINT = None
BLOB_ROOT = None   # override with env SOLBENCH_BLOB_ROOT
AXES = json.loads(r"""{
    "batch_size": {
        "type": "var",
        "value": null,
        "description": "Batch size"
    },
    "seq_len": {
        "type": "var",
        "value": null,
        "description": "Sequence length"
    },
    "hidden_size": {
        "type": "const",
        "value": 4096,
        "description": "Hidden dimension size"
    }
}""")
INPUT_SPECS = json.loads(r"""{
    "sublayer_output": {
        "shape": [
            "batch_size",
            "seq_len",
            "hidden_size"
        ],
        "dtype": "bfloat16",
        "description": "Output from attention or MLP sublayer",
        "role": "input"
    },
    "residual": {
        "shape": [
            "batch_size",
            "seq_len",
            "hidden_size"
        ],
        "dtype": "bfloat16",
        "description": "Residual connection input",
        "role": "input"
    },
    "weight": {
        "shape": [
            "hidden_size"
        ],
        "dtype": "bfloat16",
        "description": "Learned scale parameter for RMSNorm",
        "role": "input"
    },
    "eps": {
        "shape": null,
        "dtype": "float32",
        "description": "Epsilon for numerical stability in RMSNorm",
        "role": "scalar"
    }
}""")
WORKLOADS = [json.loads(line) for line in r"""
{"uuid": "0e60aff3-9424-553b-99ac-4e1657d5cc6b", "axes": {"batch_size": 16, "seq_len": 1024}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0063, "max_rtol": 0.05}, "latency": {"h200": {"baseline": 0.98, "target": 0.12}}}
{"uuid": "371a388c-51f0-5416-a9eb-926337939aee", "axes": {"batch_size": 8, "seq_len": 2048}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0063, "max_rtol": 0.05}}
{"uuid": "11183480-fb43-5c20-a887-7226134c5fc1", "axes": {"batch_size": 32, "seq_len": 256}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0034, "max_rtol": 0.05}}
{"uuid": "ad827ab9-fb43-5e7f-8ab3-c5ca544ad5cb", "axes": {"batch_size": 8, "seq_len": 997}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.009800000000000001, "max_rtol": 0.05}}
{"uuid": "8496a51c-a1a2-5fd9-a168-15e5d1b62a3a", "axes": {"batch_size": 16, "seq_len": 512}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0034, "max_rtol": 0.05}}
{"uuid": "1078ccd0-3870-5c7e-af42-4aeb7ab8d1ed", "axes": {"batch_size": 4, "seq_len": 2048}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0034, "max_rtol": 0.05}}
{"uuid": "1f59fd2c-b24d-5298-adfd-d06581bf7a8a", "axes": {"batch_size": 1, "seq_len": 131}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 1e-05, "max_rtol": 0.05}}
{"uuid": "da811c4c-e55f-5f55-8a81-9ac4c96b12ba", "axes": {"batch_size": 2, "seq_len": 2053}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.005900000000000001, "max_rtol": 0.05}}
{"uuid": "a36d6608-4971-5f89-ad4c-70b7cfc2cd14", "axes": {"batch_size": 2, "seq_len": 4096}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0034, "max_rtol": 0.05}}
{"uuid": "97064acd-a78b-5ff6-879d-d57a186b425b", "axes": {"batch_size": 8, "seq_len": 512}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.005900000000000001, "max_rtol": 0.05}}
{"uuid": "99216e04-e7bb-5c17-945b-f46cf9f37ca6", "axes": {"batch_size": 4, "seq_len": 128}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 1e-05, "max_rtol": 0.05}}
{"uuid": "e82f3830-1095-5d83-8d44-b192ffc2e898", "axes": {"batch_size": 1, "seq_len": 1024}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0044, "max_rtol": 0.05}}
{"uuid": "2d375849-8474-5f30-97cf-fc5810de29e8", "axes": {"batch_size": 2, "seq_len": 293}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 1e-05, "max_rtol": 0.05}}
{"uuid": "a7de2a7b-5bc4-5e2d-811a-79bfacb87605", "axes": {"batch_size": 2, "seq_len": 2048}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.005900000000000001, "max_rtol": 0.05}}
{"uuid": "72f1b676-a464-5873-9b06-bf18fe6883ee", "axes": {"batch_size": 8, "seq_len": 256}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.00073, "max_rtol": 0.05}}
{"uuid": "c192baac-31dd-5462-b926-a68d0d155d5f", "axes": {"batch_size": 1, "seq_len": 128}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 1e-05, "max_rtol": 0.05}}
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
    """(baseline_ms, target_ms) for this workload on the given/current GPU,
    or None when the workload carries no target for it."""
    key = key or gpu_key()
    if key is None:
        return None
    spec = WORKLOADS[workload_idx].get("latency", {}).get(key)
    if spec is None:
        return None
    return float(spec["baseline"]), float(spec["target"])


# ---- reference implementation (verbatim from problem.md) ---- #

import torch

@torch.no_grad()
def run(sublayer_output: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Post-normalization residual connection: output = residual + RMSNorm(sublayer_output)

    RMSNorm computation:
    1. Compute variance: mean of squared values along hidden dimension
    2. Normalize: x * rsqrt(variance + eps)
    3. Apply learned scale (weight parameter)
    4. Add residual connection

    Args:
        sublayer_output: Output from attention or MLP sublayer [batch, seq_len, hidden_size]
        residual: Residual connection input [batch, seq_len, hidden_size]
        weight: Learned scale parameter [hidden_size]
        eps: Epsilon for numerical stability

    Returns:
        Output tensor with residual added [batch, seq_len, hidden_size]
    """
    # Store input dtype for final conversion
    input_dtype = sublayer_output.dtype

    # RMSNorm computation in float32 for numerical stability
    normalized = sublayer_output.to(torch.float32)

    # Compute variance: mean of squared values along hidden dimension
    # Shape: [batch, seq_len, 1]
    variance = normalized.pow(2).mean(-1, keepdim=True)

    # Normalize: x * rsqrt(variance + eps)
    # rsqrt is more efficient than 1/sqrt
    normalized = normalized * torch.rsqrt(variance + eps)

    # Apply learned scale (weight parameter)
    normalized = weight.to(torch.float32) * normalized

    # Convert back to input dtype
    normalized = normalized.to(input_dtype)

    # Add residual connection
    output = residual + normalized

    return output

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
