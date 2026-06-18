#!/usr/bin/env python3
"""
Static complexity analysis for KernelBench-style problem files.

Parses a PyTorch problem file's AST and extracts complexity features
to recommend whether to use the direct KernelAgent path or the full
Fuser pipeline (extract → dispatch → compose).

Usage:
    python analyze_problem.py --problem /path/to/problem.py

Output:
    JSON to stdout with complexity features and routing recommendation.

Logic ported from: Fuser/auto_agent.py (analyze_problem_code + Complexity)
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# ── Token sets (from auto_agent.py) ──────────────────────────────────────────

_HARD_OP_TOKENS = {
    "conv_transpose2d",
    "multiheadattention",
    "scaled_dot_product_attention",
    "attention",
    "softmax",
    "einsum",
    "group_norm",
}

_CONV_OP_TOKENS = {"conv2d", "conv1d", "conv3d"}
_POOL_OP_TOKENS = {
    "max_pool2d",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
}
_ACT_TOKENS = {"relu", "gelu", "tanh", "sigmoid", "silu", "mish", "leaky_relu"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _dotted_name(n: ast.AST) -> str:
    parts: list[str] = []
    cur = n
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    parts.reverse()
    return ".".join(parts)


# ── Complexity dataclass ─────────────────────────────────────────────────────

@dataclass
class Complexity:
    has_control_flow: bool
    has_attention_like: bool
    has_conv_transpose: bool
    has_group_norm: bool
    has_conv: bool
    pool_ops: int
    act_ops: int
    chain_len_estimate: int
    raw_op_names: dict[str, int]

    def route_to_fuser(self) -> bool:
        """Return True if the problem should be routed to the Fuser pipeline."""
        # Primary triggers
        if self.has_attention_like:
            if (
                not self.has_control_flow
                and not self.has_group_norm
                and not self.has_conv_transpose
                and self.chain_len_estimate <= 3
            ):
                return False
            return True
        if self.has_conv_transpose:
            return True
        if self.has_control_flow:
            return True
        if self.has_group_norm and (self.has_conv or self.pool_ops > 0):
            return True
        if self.chain_len_estimate >= 4:
            return True
        return False

    def route_recommendation(self) -> str:
        """Return one of: kernelagent, fuser."""
        return "fuser" if self.route_to_fuser() else "kernelagent"

    def route_reason(self) -> str:
        """Return human-readable reason for the routing decision."""
        if self.has_attention_like:
            if self.route_to_fuser():
                return "Attention-like patterns with complex surrounding context"
            return "Self-contained attention block; direct KernelAgent should handle"
        if self.has_conv_transpose:
            return "conv_transpose2d present; route to Fuser"
        if self.has_control_flow:
            return "Control flow in forward(); route to Fuser"
        if self.has_group_norm and (self.has_conv or self.pool_ops > 0):
            return "GroupNorm with conv/pool chains; route to Fuser"
        if self.chain_len_estimate >= 4:
            return f"Long op chain (est. {self.chain_len_estimate} steps); route to Fuser"
        return "Short/simple op chain; direct KernelAgent path"


# ── Analysis function ────────────────────────────────────────────────────────

def analyze_problem_code(code: str) -> Complexity:
    """Analyze a PyTorch problem file and return complexity features."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback: plain text scan
        txt = code.lower()
        has_attention_like = any(tok in txt for tok in _HARD_OP_TOKENS)
        has_conv_transpose = "conv_transpose2d" in txt
        has_group_norm = "groupnorm" in txt or "group_norm" in txt
        has_conv = any(t in txt for t in _CONV_OP_TOKENS)
        pool_ops = sum(txt.count(t) for t in _POOL_OP_TOKENS)
        act_ops = sum(txt.count(t) for t in _ACT_TOKENS)
        chain_len_estimate = 0
        for ln in txt.splitlines():
            s = ln.strip()
            if s.startswith("x =") and any(
                t in s
                for t in (
                    list(_CONV_OP_TOKENS)
                    + list(_POOL_OP_TOKENS)
                    + list(_ACT_TOKENS)
                    + ["matmul", "bmm", "einsum"]
                )
            ):
                chain_len_estimate += 1
        return Complexity(
            has_control_flow=(" if " in txt or " for " in txt or " while " in txt),
            has_attention_like=has_attention_like,
            has_conv_transpose=has_conv_transpose,
            has_group_norm=has_group_norm,
            has_conv=has_conv,
            pool_ops=pool_ops,
            act_ops=act_ops,
            chain_len_estimate=chain_len_estimate,
            raw_op_names={},
        )

    # AST path
    has_control_flow = False
    raw_op_counts: dict[str, int] = {}
    has_attention_like = False
    has_conv_transpose = False
    has_group_norm = False
    has_conv = False
    pool_ops = 0
    act_ops = 0
    chain_len_estimate = 0

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
            nonlocal has_control_flow
            if node.name == "forward":
                for n in ast.walk(node):
                    if isinstance(n, (ast.If, ast.For, ast.While)):
                        has_control_flow = True
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> Any:
            nonlocal has_attention_like, has_conv_transpose, has_group_norm, \
                has_conv, pool_ops, act_ops, chain_len_estimate
            try:
                name = _dotted_name(node.func).lower()
            except Exception:
                name = ""
            if name:
                raw_op_counts[name] = raw_op_counts.get(name, 0) + 1
                base = name.split(".")[-1]
                if base in _HARD_OP_TOKENS:
                    has_attention_like = True
                if base == "conv_transpose2d":
                    has_conv_transpose = True
                if base in _CONV_OP_TOKENS:
                    has_conv = True
                if base == "group_norm":
                    has_group_norm = True
                if base in _POOL_OP_TOKENS:
                    pool_ops += 1
                if base in _ACT_TOKENS:
                    act_ops += 1
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> Any:
            nonlocal chain_len_estimate
            try:
                is_x = any(
                    isinstance(t, ast.Name) and t.id == "x" for t in node.targets
                )
                is_call = isinstance(node.value, ast.Call)
                if is_x and is_call:
                    chain_len_estimate += 1
            except Exception:
                pass
            self.generic_visit(node)

    _Visitor().visit(tree)
    return Complexity(
        has_control_flow=has_control_flow,
        has_attention_like=has_attention_like,
        has_conv_transpose=has_conv_transpose,
        has_group_norm=has_group_norm,
        has_conv=has_conv,
        pool_ops=pool_ops,
        act_ops=act_ops,
        chain_len_estimate=chain_len_estimate,
        raw_op_names=raw_op_counts,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Analyze a KernelBench problem file for routing complexity"
    )
    p.add_argument(
        "--problem", required=True, help="Path to the problem .py file"
    )
    args = p.parse_args(argv)

    problem_path = Path(args.problem).resolve()
    if not problem_path.is_file():
        print(json.dumps({"error": f"File not found: {problem_path}"}))
        return 2

    code = problem_path.read_text(encoding="utf-8")
    cx = analyze_problem_code(code)

    result = asdict(cx)
    result["route_recommendation"] = cx.route_recommendation()
    result["route_reason"] = cx.route_reason()
    result["problem_path"] = str(problem_path)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
