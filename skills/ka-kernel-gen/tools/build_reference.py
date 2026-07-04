#!/usr/bin/env python3
"""
Build PyTorch reference code and problem description from a subgraph JSON item.

Generates a structured problem description and reference implementation
for a single subgraph, suitable for passing to a Triton kernel generator.

Usage:
    python build_reference.py --subgraph '{"id": "sg_1", "type": "conv_relu", ...}'
    python build_reference.py --subgraph-file /path/to/subgraph_item.json --platform cuda

Output:
    JSON to stdout: {"problem_description": "...", "reference_code": "..."}

Logic ported from: Fuser/dispatch_kernel_agent.py
    (_build_reference_code, _synthesize_problem_description)
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any


def _shape_list(shape: Any) -> list[str]:
    if isinstance(shape, list):
        return [str(x) for x in shape]
    return [str(shape)] if shape is not None else []


def _fmt_shape(shape: Any) -> str:
    return "[" + ", ".join(_shape_list(shape)) + "]"


def _py_tuple(arr: Any) -> str:
    vals = _shape_list(arr)
    if not vals:
        return "()"
    if len(vals) == 1:
        return f"({vals[0]},)"
    return f"({', '.join(vals)})"


def _pick_weights(item: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    ws = (
        item.get("weights_fused")
        or item.get("weights_original")
        or item.get("weights")
        or {}
    )
    if not isinstance(ws, dict):
        return {}
    out: dict[str, Any] = {}
    for k in keys:
        if k in ws:
            out[k] = ws[k]
    for k, v in ws.items():
        out.setdefault(k, v)
    return out


def build_reference_code(item: dict[str, Any]) -> tuple[str, list[str]]:
    """Return (reference_code_str, param_names) implementing the subgraph."""
    ops: list[dict[str, Any]] = [
        op for op in (item.get("ops") or []) if isinstance(op, dict)
    ]
    lines: list[str] = ["import torch", "import torch.nn.functional as F", ""]
    params: list[str] = []

    inputs_multi = item.get("inputs")
    input_names: list[str]
    if isinstance(inputs_multi, list) and inputs_multi:
        input_names = [f"x{i}" for i in range(len(inputs_multi))]
        header = f"def reference({', '.join(input_names)}"
    else:
        input_names = ["x"]
        header = "def reference(x"

    body: list[str] = []
    cur = input_names[0] if input_names else "x"

    for op in ops:
        kind = str(op.get("op"))
        if kind == "conv2d":
            wmap = _pick_weights(item, ["conv_weight", "weight", "bias"])
            w = (
                "conv_weight"
                if "conv_weight" in wmap
                else ("weight" if "weight" in wmap else "conv_weight")
            )
            b = "bias" if "bias" in wmap else None
            op_args: list[str] = [cur, w]
            if b:
                op_args.append(b)
            stride = _py_tuple(op.get("stride", (1, 1)))
            padding = _py_tuple(op.get("padding", (0, 0)))
            dilation = _py_tuple(op.get("dilation", (1, 1)))
            groups = str(op.get("groups", 1))
            body.append(
                f"{cur} = F.conv2d({', '.join(op_args)}, stride={stride}, padding={padding}, dilation={dilation}, groups={groups})"
            )
            params.extend([p for p in [w, b] if p])
        elif kind == "conv_transpose2d":
            wmap = _pick_weights(item, ["conv_transpose.weight", "weight", "bias"])
            w = (
                "conv_transpose_weight"
                if "conv_transpose.weight" in wmap
                else ("weight" if "weight" in wmap else "conv_transpose_weight")
            )
            b = "bias" if "bias" in wmap else None
            op_args = [cur, w]
            if b:
                op_args.append(b)
            stride = _py_tuple(op.get("stride", (1, 1)))
            padding = _py_tuple(op.get("padding", (0, 0)))
            dilation = _py_tuple(op.get("dilation", (1, 1)))
            outpad = _py_tuple(op.get("output_padding", (0, 0)))
            groups = str(op.get("groups", 1))
            body.append(
                f"{cur} = F.conv_transpose2d({', '.join(op_args)}, stride={stride}, padding={padding}, dilation={dilation}, output_padding={outpad}, groups={groups})"
            )
            params.extend([p for p in [w, b] if p])
        elif kind in ("relu", "tanh", "sigmoid"):
            body.append(f"{cur} = torch.{kind}({cur})")
        elif kind == "max_pool2d":
            k = _py_tuple(op.get("kernel_size", (2, 2)))
            s = _py_tuple(op.get("stride", (2, 2)))
            p = _py_tuple(op.get("padding", (0, 0)))
            d = _py_tuple(op.get("dilation", (1, 1)))
            ceil = bool(op.get("ceil_mode", False))
            body.append(
                f"{cur} = F.max_pool2d({cur}, kernel_size={k}, stride={s}, padding={p}, dilation={d}, ceil_mode={str(ceil).lower()})"
            )
        elif kind == "avg_pool2d":
            k = _py_tuple(op.get("kernel_size", (2, 2)))
            s = _py_tuple(op.get("stride", (2, 2)))
            p = _py_tuple(op.get("padding", (0, 0)))
            body.append(
                f"{cur} = F.avg_pool2d({cur}, kernel_size={k}, stride={s}, padding={p})"
            )
        elif kind == "batch_norm":
            wmap = _pick_weights(
                item,
                [
                    "batch_norm.weight", "batch_norm.bias",
                    "batch_norm.running_mean", "batch_norm.running_var",
                    "weight", "bias", "running_mean", "running_var",
                ],
            )
            w = "bn_weight"
            b = "bn_bias"
            rm = "bn_running_mean"
            rv = "bn_running_var"
            eps = op.get("eps", 1e-5)
            momentum = op.get("momentum", 0.1)
            body.append(
                f"{cur} = F.batch_norm({cur}, {rm}, {rv}, {w}, {b}, training=False, momentum={momentum}, eps={eps})"
            )
            params.extend([w, b, rm, rv])
        elif kind == "group_norm":
            w = "gn_weight"
            b = "gn_bias"
            num_groups = int(op.get("num_groups", 1))
            eps = op.get("eps", 1e-5)
            body.append(
                f"{cur} = F.group_norm({cur}, {num_groups}, {w}, {b}, eps={eps})"
            )
            params.extend([w, b])
        elif kind in ("add", "sum"):
            if len(input_names) >= 2:
                body.append(f"{cur} = {input_names[0]} + {input_names[1]}")
            else:
                body.append(f"{cur} = {cur} + {cur}")
        elif kind == "gemm":
            w = "linear_weight"
            b = "linear_bias"
            body.append(f"{cur} = torch.nn.functional.linear({cur}, {w}, {b})")
            params.extend([w, b])
        else:
            body.append(
                f"# TODO: op '{kind}' not explicitly handled; update generator if needed"
            )

    header += "):\n"
    lines.append(header)
    if not body:
        body = ["return x"]
    indented = ["    " + ln for ln in body]
    indented.append("    return " + cur)
    lines.extend(indented)
    return "\n".join(lines) + "\n", params


def synthesize_problem_description(
    item: dict[str, Any],
    platform_name: str = "cuda",
    device_string: str = "cuda",
) -> str:
    """Create a problem description for Triton kernel generation."""
    id_ = str(item.get("id", "unknown"))
    type_ = str(item.get("type", ""))
    layout = item.get("data_layout") or "NCHW"
    dtype = item.get("dtype") or "float32"
    input_shape = item.get("input_shape")
    output_shape = item.get("output_shape")
    inputs_multi = item.get("inputs")
    weights_fused = item.get("weights_fused")
    weights_orig = item.get("weights_original")
    source = item.get("source") or {}

    ref_code, _ = build_reference_code(item)

    header = textwrap.dedent(
        f"""
        Implement a Triton kernel that computes the following subgraph end-to-end.

        Subgraph ID: {id_}
        Type: {type_}
        Data layout: {layout}
        DType: {dtype}
        Target Platform: {platform_name}
        Device String: {device_string}

        Shapes:
        - input: {_fmt_shape(inputs_multi[0]) if isinstance(inputs_multi, list) else _fmt_shape(input_shape)}
        {("- input2: " + _fmt_shape(inputs_multi[1])) if isinstance(inputs_multi, list) and len(inputs_multi) > 1 else ""}
        - output: {_fmt_shape(output_shape)}

        Weights (fused): {json.dumps(weights_fused, indent=2) if isinstance(weights_fused, dict) else "null"}
        Weights (original): {json.dumps(weights_orig, indent=2) if isinstance(weights_orig, dict) else "null"}

        Operations in order (with parameters):
        {json.dumps(item.get("ops", []), indent=2)}

        Requirements:
        - Return a complete Python file with a @triton.jit kernel and a wrapper function named kernel_function(...).
        - kernel_function must accept input tensor(s) and any required weights/bias parameters (match shapes above).
        - Implement the exact semantics of the listed ops in the given order for the provided shapes.
        - Use {layout} layout and {dtype} dtype semantics.
        - Allocate inputs, weights, intermediates, and outputs on device='{device_string}'.
        - The test will import kernel_function and compare to the reference implementation below.

        Test tolerance policy:
        - Default tolerances: rtol=1e-3, atol=1e-3.
        - Absolute cap: NEVER exceed rtol=1e-2 or atol=1e-2.

        Reference PyTorch implementation (exact semantics to match):
        """
    ).strip()

    src_code_block = ""
    if isinstance(source, dict) and source.get("code"):
        mod = source.get("module", "Model")
        code = str(source.get("code"))
        src_code_block = f"\nOriginal source snippet ({mod}):\n```python\n{code}\n```\n"

    problem = header + "\n\n```python\n" + ref_code + "```\n" + src_code_block
    return problem


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build reference code and problem description from a subgraph"
    )
    p.add_argument(
        "--subgraph",
        default=None,
        help="Subgraph JSON as string",
    )
    p.add_argument(
        "--subgraph-file",
        default=None,
        help="Path to subgraph JSON file (single item or array with --index)",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index into subgraphs array (default: 0)",
    )
    p.add_argument(
        "--platform",
        default="cuda",
        choices=["cuda", "xpu"],
        help="Target platform (default: cuda)",
    )
    args = p.parse_args(argv)

    if args.subgraph:
        item = json.loads(args.subgraph)
    elif args.subgraph_file:
        data = json.loads(Path(args.subgraph_file).read_text(encoding="utf-8"))
        if isinstance(data, list):
            if args.index >= len(data):
                print(json.dumps({"error": f"Index {args.index} out of range ({len(data)} items)"}))
                return 1
            item = data[args.index]
        else:
            item = data
    else:
        print(json.dumps({"error": "Provide --subgraph or --subgraph-file"}))
        return 2

    device_map = {"cuda": "cuda", "xpu": "xpu"}
    device_string = device_map.get(args.platform, "cuda")

    ref_code, param_names = build_reference_code(item)
    desc = synthesize_problem_description(
        item,
        platform_name=args.platform,
        device_string=device_string,
    )

    result = {
        "problem_description": desc,
        "reference_code": ref_code,
        "param_names": param_names,
        "subgraph_id": item.get("id", "unknown"),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
