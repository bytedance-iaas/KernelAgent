"""Load flashinfer-bench contest definitions from JSON files.

Each JSON file under the contest definitions directory has the shape:
  {
    "name": "<key>",
    "description": "...",
    "axes": { "<name>": {"type": "const"|"var", "value": <int>}, ... },
    "inputs": { "<name>": {"shape": [...] | null, "dtype": "...", "optional": bool}, ... },
    "outputs": { "<name>": {"shape": [...], "dtype": "..."}, ... },
    "reference": "<full Python source with run() function>"
  }

Variable-type axes get sensible defaults so get_inputs() can create tensors.
Scalar inputs (shape=null) are generated as Python literals in get_inputs().
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from flashinfer.adapter import DefinitionSpec, TensorSpec


# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "bfloat16":       "torch.bfloat16",
    "float32":        "torch.float32",
    "float16":        "torch.float16",
    "float8_e4m3fn":  "torch.float8_e4m3fn",
    "int8":           "torch.int8",
    "uint8":          "torch.uint8",
    "int32":          "torch.int32",
    "int64":          "torch.int64",
}


# ---------------------------------------------------------------------------
# Per-definition defaults for variable-type axes
# ---------------------------------------------------------------------------

_VAR_DEFAULTS: dict[str, dict[str, int]] = {
    "gdn_decode_qk4_v8_d128_k_last": {
        "batch_size": 16,
    },
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048": {
        "seq_len": 32,
    },
    "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64": {
        "num_tokens": 16,
        "num_pages":  512,
    },
    "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64": {
        "batch_size":    16,
        "max_num_pages": 64,
        "num_pages":     512,
    },
    "gdn_prefill_qk4_v8_d128_k_last": {
        "total_seq_len":   1024,
        "num_seqs":        4,
        "len_cu_seqlens":  5,
    },
}


# ---------------------------------------------------------------------------
# Default init expressions for scalar inputs (shape=null)
# ---------------------------------------------------------------------------

_SCALAR_DEFAULTS: dict[str, str] = {
    "scale":                 "1.0 / math.sqrt(head_size)",
    "sm_scale":              "1.0 / math.sqrt(576)",   # 1/sqrt(512+64)
    "local_expert_offset":   "0",
    "routed_scaling_factor": "1.0",
}


# ---------------------------------------------------------------------------
# Per-definition, per-input custom initialization expressions.
# Evaluated with axes constants in scope (they shadow the default zeros init).
# ---------------------------------------------------------------------------

_CUSTOM_INIT: dict[str, dict[str, str]] = {
    # sparse_indices must be varied real indices into the KV cache, not zeros.
    # All-zero indices collapse attention to a single KV position, letting
    # trivially-wrong kernels (e.g. stride=0 broadcast of cache[0]) pass tests.
    "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64": {
        "sparse_indices": (
            "(torch.arange(num_tokens * topk, dtype=torch.int32, device='cuda')"
            " % (num_pages * page_size)).view(num_tokens, topk)"
        ),
    },
    # seq_lens needs realistic values; block_table maps batch→pages
    "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64": {
        "seq_lens": (
            "torch.full((batch_size,), max_num_pages * page_size,"
            " dtype=torch.int32, device='cuda')"
        ),
        "block_table": (
            "torch.arange(num_pages, dtype=torch.int32, device='cuda')"
            ".unsqueeze(0).expand(batch_size, -1)"
            "[:, :max_num_pages].contiguous()"
        ),
    },
    # cu_seqlens must be a valid cumulative sum, not zeros
    "gdn_prefill_qk4_v8_d128_k_last": {
        "cu_seqlens": (
            "torch.tensor([i * (total_seq_len // num_seqs)"
            " for i in range(num_seqs + 1)],"
            " dtype=torch.int64, device='cuda')"
        ),
    },
}


# ---------------------------------------------------------------------------
# Tolerances per definition
# ---------------------------------------------------------------------------

_TOLERANCES: dict[str, tuple[float, float]] = {
    "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048": (1e-1, 1e-1),
    "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64":                   (0.0, 0.0),
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_contest_json(json_path: Path) -> DefinitionSpec:
    """Parse a contest definition JSON file into a DefinitionSpec."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    name        = data["name"]
    description = data.get("description", name)
    reference_source = data.get("reference", "").replace("\r\n", "\n")

    # ------------------------------------------------------------------
    # Build axes dict (const values; var axes use per-definition defaults)
    # ------------------------------------------------------------------
    axes_raw     = data.get("axes", {})
    var_defaults = _VAR_DEFAULTS.get(name, {})
    axes: dict[str, int] = {}
    for ax_name, ax_spec in axes_raw.items():
        if isinstance(ax_spec, dict):
            if ax_spec.get("type") == "const":
                axes[ax_name] = int(ax_spec["value"])
            elif ax_name in var_defaults:
                axes[ax_name] = var_defaults[ax_name]
            # var axes without a default are skipped (not needed for shapes)
        elif isinstance(ax_spec, (int, float)):
            axes[ax_name] = int(ax_spec)

    def resolve_dim(dim: Any) -> Any:
        if isinstance(dim, int):
            return dim
        if isinstance(dim, str):
            try:
                return int(eval(dim, {"__builtins__": {}}, axes))
            except Exception:
                return dim  # keep as unevaluated string
        return dim

    # ------------------------------------------------------------------
    # Parse inputs (maintain JSON order — matches run() signature)
    # ------------------------------------------------------------------
    inputs_raw = data.get("inputs", {})
    tensor_inputs: list[TensorSpec] = []
    scalar_inputs: dict[str, str] = {}
    input_order:   list[str]       = list(inputs_raw.keys())
    custom_init    = _CUSTOM_INIT.get(name, {})

    custom_tensor_init: dict[str, str] = {}

    for inp_name, inp_spec in inputs_raw.items():
        shape    = inp_spec.get("shape")
        dtype_str = inp_spec.get("dtype", "float32")
        dtype    = _DTYPE_MAP.get(dtype_str, f"torch.{dtype_str}")

        if shape is None:
            # Scalar / non-tensor input
            scalar_inputs[inp_name] = _SCALAR_DEFAULTS.get(inp_name, "0")
        else:
            resolved = [resolve_dim(d) for d in shape]
            tensor_inputs.append(TensorSpec(inp_name, resolved, dtype))
            if inp_name in custom_init:
                custom_tensor_init[inp_name] = custom_init[inp_name]

    # ------------------------------------------------------------------
    # Parse outputs
    # ------------------------------------------------------------------
    outputs_raw = data.get("outputs", {})
    outputs: list[TensorSpec] = []
    for out_name, out_spec in outputs_raw.items():
        shape    = out_spec.get("shape") or []
        dtype_str = out_spec.get("dtype", "float32")
        dtype    = _DTYPE_MAP.get(dtype_str, f"torch.{dtype_str}")
        resolved = [resolve_dim(d) for d in shape]
        outputs.append(TensorSpec(out_name, resolved, dtype))

    atol, rtol = _TOLERANCES.get(name, (1e-2, 1e-2))

    return DefinitionSpec(
        key=name,
        description=description,
        inputs=tensor_inputs,
        outputs=outputs,
        reference_source=reference_source,
        input_order=input_order,
        scalar_inputs=scalar_inputs,
        custom_tensor_init=custom_tensor_init,
        axes=axes,
        atol=atol,
        rtol=rtol,
    )


def discover_contest_jsons(contest_root: Path) -> dict[str, Path]:
    """Recursively find all .json definition files under contest_root."""
    result = {}
    for p in sorted(contest_root.rglob("*.json")):
        result[p.stem] = p
    return result
