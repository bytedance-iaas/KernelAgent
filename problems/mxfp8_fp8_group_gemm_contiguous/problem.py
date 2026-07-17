"""Auto-generated from problem.md by scripts/problem_md.py - do not edit.

KernelBench-format problem: `Model` wraps the reference `run(...)`;
`get_inputs()` builds the canonical (first) workload; `WORKLOADS` +
`build_workload_inputs(i)` expose every workload for the test harness.
"""

import json

import torch
import torch.nn as nn

PROBLEM_NAME = 'mxfp8_fp8_group_gemm_contiguous'
CUSTOM_INPUTS_ENTRYPOINT = 'get_inputs'
BLOB_ROOT = None   # override with env SOLBENCH_BLOB_ROOT
AXES = json.loads(r"""{
    "groups": {
        "type": "var",
        "value": null,
        "description": "Number of expert groups (one B matrix per group)"
    },
    "m_per_group": {
        "type": "var",
        "value": null,
        "description": "Rows of A per group (contiguous)"
    },
    "n": {
        "type": "var",
        "value": null,
        "description": "Output columns / rows of each B matrix"
    },
    "k": {
        "type": "var",
        "value": null,
        "description": "Contraction dimension (multiple of 128)"
    },
    "gran_k": {
        "type": "const",
        "value": 32,
        "description": "MXFP8 scale block size along K (UE8M0 per 32 K)"
    },
    "m": {
        "type": "expr",
        "value": "groups * m_per_group",
        "description": "Total rows of A"
    },
    "k_scale": {
        "type": "expr",
        "value": "k // gran_k",
        "description": "Number of UE8M0 scale bytes along K"
    }
}""")
INPUT_SPECS = json.loads(r"""{
    "a_data": {
        "shape": [
            "m",
            "k"
        ],
        "dtype": "float8_e4m3fn",
        "description": "Quantized activation A (row-major)",
        "role": "input"
    },
    "a_scale": {
        "shape": [
            "m",
            "k_scale"
        ],
        "dtype": "uint8",
        "description": "UE8M0 (e8m0) block scales for A, one per 32 K",
        "role": "input"
    },
    "b_data": {
        "shape": [
            "groups",
            "n",
            "k"
        ],
        "dtype": "float8_e4m3fn",
        "description": "Quantized per-group weights B",
        "role": "input"
    },
    "b_scale": {
        "shape": [
            "groups",
            "n",
            "k_scale"
        ],
        "dtype": "uint8",
        "description": "UE8M0 (e8m0) block scales for B, one per 32 K",
        "role": "input"
    },
    "grouped_layout": {
        "shape": [
            "m"
        ],
        "dtype": "int32",
        "description": "Per-row group id (contiguous m-grouping)",
        "role": "input"
    }
}""")
WORKLOADS = [json.loads(line) for line in r"""
{"uuid": "1b3c0a10-0001-4a00-9000-mxfp8ctg0001", "axes": {"groups": 2, "m_per_group": 128, "n": 48, "k": 128}, "inputs": {"a_data": {"type": "custom"}, "a_scale": {"type": "custom"}, "b_data": {"type": "custom"}, "b_scale": {"type": "custom"}, "grouped_layout": {"type": "custom"}}, "tolerance": {"max_atol": 0.5, "max_rtol": 0.05, "required_matched_ratio": 0.97}}
{"uuid": "1b3c0a10-0002-4a00-9000-mxfp8ctg0002", "axes": {"groups": 2, "m_per_group": 128, "n": 128, "k": 512}, "inputs": {"a_data": {"type": "custom"}, "a_scale": {"type": "custom"}, "b_data": {"type": "custom"}, "b_scale": {"type": "custom"}, "grouped_layout": {"type": "custom"}}, "tolerance": {"max_atol": 1.0, "max_rtol": 0.05, "required_matched_ratio": 0.97}}
{"uuid": "1b3c0a10-0003-4a00-9000-mxfp8ctg0003", "axes": {"groups": 4, "m_per_group": 128, "n": 1024, "k": 1024}, "inputs": {"a_data": {"type": "custom"}, "a_scale": {"type": "custom"}, "b_data": {"type": "custom"}, "b_scale": {"type": "custom"}, "grouped_layout": {"type": "custom"}}, "tolerance": {"max_atol": 2.0, "max_rtol": 0.05, "required_matched_ratio": 0.97}}
{"uuid": "1b3c0a10-0004-4a00-9000-mxfp8ctg0004", "axes": {"groups": 4, "m_per_group": 512, "n": 1024, "k": 1024}, "inputs": {"a_data": {"type": "custom"}, "a_scale": {"type": "custom"}, "b_data": {"type": "custom"}, "b_scale": {"type": "custom"}, "grouped_layout": {"type": "custom"}}, "tolerance": {"max_atol": 2.0, "max_rtol": 0.05, "required_matched_ratio": 0.97}}
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


# --- MXFP8 / UE8M0 helpers (pure PyTorch, self-contained) ---------------- #

def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round a positive fp32 scale up to the next power of two, returned as
    an fp32 value whose mantissa is zero (a pure UE8M0/e8m0 scale)."""
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def _e8m0_u8_from_fp32(sf: torch.Tensor) -> torch.Tensor:
    """Extract the biased fp32 exponent byte of a power-of-two scale."""
    return ((sf.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def _fp32_from_e8m0_u8(sf_u8: torch.Tensor) -> torch.Tensor:
    """UE8M0 byte -> fp32 scale = 2^(byte - 127) = reinterpret(byte << 23)."""
    return (sf_u8.to(torch.int32) << 23).view(torch.float32)


def _per_token_cast_to_mxfp8(x: torch.Tensor, gran_k: int = 32):
    """Quantize a [rows, k] tensor to float8_e4m3fn with per-(row, 32-K-block)
    UE8M0 scales. Mirrors DeepGEMM's per_token_cast_to_fp8(use_ue8m0=True)."""
    rows, k = x.shape
    assert k % gran_k == 0
    xv = x.float().view(rows, k // gran_k, gran_k)
    amax = xv.abs().amax(dim=2).clamp(1e-4)
    sf = _ceil_to_ue8m0(amax / 448.0)                       # fp32 power-of-two
    x_fp8 = (xv * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(rows, k)
    return x_fp8.contiguous(), _e8m0_u8_from_fp32(sf)


def _dequant(x_fp8: torch.Tensor, sf_u8: torch.Tensor, gran_k: int) -> torch.Tensor:
    """Dequantize FP8 data with UE8M0 block scales to fp32."""
    sf_fp32 = _fp32_from_e8m0_u8(sf_u8)
    group_idx = torch.arange(x_fp8.size(-1), device=x_fp8.device) // gran_k
    return x_fp8.float() * sf_fp32[..., group_idx]


def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict:
    """Build one workload's quantized inputs (deterministic; the harness
    seeds torch before calling this)."""
    groups = int(axes_and_scalars["groups"])
    m_per_group = int(axes_and_scalars["m_per_group"])
    n = int(axes_and_scalars["n"])
    k = int(axes_and_scalars["k"])
    gran_k = 32
    m = groups * m_per_group

    a_ref = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    b_ref = torch.randn((groups, n, k), device=device, dtype=torch.bfloat16)

    a_data, a_scale = _per_token_cast_to_mxfp8(a_ref, gran_k)
    b_data = torch.empty((groups, n, k), device=device, dtype=torch.float8_e4m3fn)
    b_scale = torch.empty((groups, n, k // gran_k), device=device, dtype=torch.uint8)
    for g in range(groups):
        b_data[g], b_scale[g] = _per_token_cast_to_mxfp8(b_ref[g], gran_k)

    grouped_layout = torch.arange(
        groups, device=device, dtype=torch.int32
    ).repeat_interleave(m_per_group)

    return {
        "a_data": a_data,
        "a_scale": a_scale,
        "b_data": b_data,
        "b_scale": b_scale,
        "grouped_layout": grouped_layout,
    }


@torch.no_grad()
def run(a_data, a_scale, b_data, b_scale, grouped_layout):
    """SM90 MXFP8-FP8 M-grouped contiguous GEMM (NT), FP32 accumulation.

    Args:
        a_data:         [m, k]           float8_e4m3fn quantized A
        a_scale:        [m, k // 32]     uint8 UE8M0 block scales for A
        b_data:         [groups, n, k]   float8_e4m3fn quantized B
        b_scale:        [groups, n, k//32] uint8 UE8M0 block scales for B
        grouped_layout: [m]              int32 per-row group id (contiguous)
    Returns:
        d: [m, n] bfloat16, row i = dequant(A[i]) @ dequant(B[group(i)]).T
    """
    gran_k = 32
    m, k = a_data.shape
    groups, n, _ = b_data.shape

    a_deq = _dequant(a_data, a_scale, gran_k)                # [m, k] fp32
    out = torch.zeros((m, n), device=a_data.device, dtype=torch.bfloat16)
    gl = grouped_layout.to(torch.long)
    for g in range(groups):
        rows = gl == g
        if not bool(rows.any()):
            continue
        b_deq = _dequant(b_data[g], b_scale[g], gran_k)      # [n, k] fp32
        out[rows] = (a_deq[rows] @ b_deq.t()).to(torch.bfloat16)
    return out

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
