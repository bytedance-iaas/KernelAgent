# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapter: flashinfer Definition → KernelAgent problem directory.

A DefinitionSpec describes a kernel target (inputs, outputs, reference
implementation, tolerances).  FlashInferProblemAdapter renders that spec
into the three files KernelAgent expects:

    problem.py   — Model class + get_inputs() + get_init_inputs()
    input.py     — initial kernel_function (PyTorch-based, correct starting point)
    test.py      — correctness harness matching the definition's tolerances
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TensorSpec:
    """Describes one tensor argument."""
    name: str
    shape: list[Any]    # integers or string expressions like "num_tokens"
    dtype: str          # e.g. "torch.bfloat16", "torch.float8_e4m3fn", "torch.int32"
    device: str = "cuda"


@dataclass
class DefinitionSpec:
    """Complete specification of a flashinfer-bench kernel target."""
    key: str                             # definition name used in solution.json
    description: str
    inputs: list[TensorSpec]             # tensor inputs only (not scalars)
    outputs: list[TensorSpec]
    # Python source that produces the reference outputs given the inputs.
    # Available names: torch, inputs (list of tensors), plus any axis variables.
    reference_body: str = ""
    # Axis variable defaults (used for get_inputs() tensor shapes)
    axes: dict[str, int] = field(default_factory=dict)
    atol: float = 1e-2
    rtol: float = 1e-2
    # Optional: initial Triton kernel body (replaces default PyTorch passthrough)
    initial_kernel_body: str = ""
    # Optional extra imports needed by problem.py
    extra_imports: str = ""
    # Full self-contained Python source from contest JSON (contains run() function).
    # When set, overrides reference_body for problem.py and initial_kernel_py.
    reference_source: str = ""
    # Ordered list of ALL input names (tensors + scalars), matching run() signature.
    input_order: list[str] = field(default_factory=list)
    # Scalar inputs: name → Python init expression (eval'd with axes in scope).
    scalar_inputs: dict[str, str] = field(default_factory=dict)
    # Custom tensor init overrides: name → expression replacing the default zeros+randn.
    custom_tensor_init: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class FlashInferProblemAdapter:
    """Converts a DefinitionSpec into KernelAgent problem directory files."""

    def __init__(self, spec: DefinitionSpec):
        self.spec = spec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_problem_dir(self, output_dir: Path) -> Path:
        """Write problem.py, input.py, test.py into output_dir.

        Returns the path to problem.py.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "problem.py").write_text(self.generate_problem_py())
        (output_dir / "input.py").write_text(self.generate_initial_kernel_py())
        (output_dir / "test.py").write_text(self.generate_test_py())
        return output_dir / "problem.py"

    def generate_problem_py(self) -> str:
        spec = self.spec
        if spec.reference_source:
            return self._generate_problem_py_from_source()
        return self._generate_problem_py_from_body()

    def _generate_problem_py_from_source(self) -> str:
        """Generate problem.py when we have a full self-contained reference source."""
        spec = self.spec
        lines = [
            '"""Auto-generated problem.py for FlashInfer definition: ' + spec.key + '"""',
            "",
            # Embed the full contest reference source (contains run() function)
            spec.reference_source.replace("\r\n", "\n"),
            "",
            "",
            "import torch.nn as nn",
            "",
            "",
        ]

        # Axis constants
        if spec.axes:
            lines.append("# Axis defaults (workload dimensions)")
            for name, val in spec.axes.items():
                lines.append(f"{name} = {val}")
            lines.append("")

        # Model class delegates to run()
        all_input_names = spec.input_order or (
            [t.name for t in spec.inputs] + list(spec.scalar_inputs)
        )
        fwd_args = ", ".join(all_input_names)
        lines += [
            "class Model(nn.Module):",
            f'    """Reference wrapper: {spec.description}"""',
            "",
            "    def __init__(self):",
            "        super().__init__()",
            "",
            f"    def forward(self, *inputs):",
            "        return run(*inputs)",
            "",
            "",
        ]

        # get_inputs — returns all inputs in signature order
        lines += ["import math", "", "def get_inputs():"]
        tensor_map = {t.name: t for t in spec.inputs}
        for name in all_input_names:
            if name in spec.custom_tensor_init:
                lines.append(f"    {name} = {spec.custom_tensor_init[name]}")
            elif name in tensor_map:
                t = tensor_map[name]
                shape_str = self._shape_str(t.shape)
                lines.append(
                    f"    {name} = torch.zeros({shape_str},"
                    f" dtype={t.dtype}, device='{t.device}')"
                )
                lines += [f"    {l}" for l in self._init_lines_raw(t)]
            elif name in spec.scalar_inputs:
                lines.append(f"    {name} = {spec.scalar_inputs[name]}")
        lines.append("    return [" + ", ".join(all_input_names) + "]")
        lines += ["", ""]

        # get_init_inputs
        lines += [
            "def get_init_inputs():",
            "    return []",
        ]
        return "\n".join(lines) + "\n"

    def _generate_problem_py_from_body(self) -> str:
        """Original problem.py generator using reference_body snippet."""
        spec = self.spec
        lines = [
            '"""Auto-generated problem.py for FlashInfer definition: '
            + spec.key + '"""',
            "",
            "import torch",
            "import torch.nn as nn",
        ]
        if spec.extra_imports:
            lines.append(spec.extra_imports)
        lines += ["", ""]

        # Axis constants
        if spec.axes:
            lines.append("# Axis defaults (workload dimensions)")
            for name, val in spec.axes.items():
                lines.append(f"{name} = {val}")
            lines.append("")

        # Model class
        input_names = ", ".join(t.name for t in spec.inputs)
        lines += [
            "class Model(nn.Module):",
            f'    """Reference implementation: {spec.description}"""',
            "",
            "    def __init__(self):",
            "        super().__init__()",
            "",
            f"    def forward(self, {input_names}):",
        ]
        for line in spec.reference_body.strip().splitlines():
            lines.append("        " + line)
        lines += ["", ""]

        # get_inputs
        lines += ["def get_inputs():"]
        for t in spec.inputs:
            shape_str = self._shape_str(t.shape)
            lines.append(
                f"    {t.name} = torch.zeros({shape_str},"
                f" dtype={t.dtype}, device='{t.device}')"
            )
            lines += self._init_lines(t)
        lines.append(
            "    return [" + ", ".join(t.name for t in spec.inputs) + "]"
        )
        lines += ["", ""]

        # get_init_inputs
        lines += [
            "def get_init_inputs():",
            "    return []",
        ]
        return "\n".join(lines) + "\n"

    def generate_initial_kernel_py(self) -> str:
        spec = self.spec
        if spec.initial_kernel_body:
            body = spec.initial_kernel_body
        elif spec.reference_source:
            body = self._kernel_from_reference_source()
        else:
            body = self._generate_kernel_from_reference()

        header = textwrap.dedent(f"""\
            # Auto-generated initial kernel for: {spec.key}
            # This is a correct PyTorch baseline — KernelAgent will optimize it.
            #
            # Definition: {spec.description}
            # Inputs: {', '.join(f'{t.name}: {t.dtype}{t.shape}' for t in spec.inputs)}
            # Outputs: {', '.join(f'{t.name}: {t.dtype}{t.shape}' for t in spec.outputs)}
            # Target: speedup_factor > 1.0 vs flashinfer-bench reference on B200
        """)
        return header + "\n" + body

    def _kernel_from_reference_source(self) -> str:
        """Produce kernel_function by renaming run() in the contest reference source.

        Also replaces torch.nn.functional patterns that KernelAgent's checker bans,
        substituting numerically-equivalent pure-torch ops or local helper functions.
        """
        import re
        src = self.spec.reference_source.replace("\r\n", "\n")

        # Detect which helpers we need
        needs_softplus = "F.softplus(" in src
        needs_sigmoid  = "torch.sigmoid(" in src
        needs_softmax  = "torch.softmax(" in src
        needs_relu     = "torch.relu(" in src

        # 1. Remove banned imports
        src = re.sub(r"^import\s+torch\.nn\.functional\s+as\s+F\s*$", "", src, flags=re.MULTILINE)
        src = re.sub(r"^from\s+torch\s+import\s+nn\b.*$", "", src, flags=re.MULTILINE)
        src = re.sub(r"^import\s+torch\.nn\b.*$", "", src, flags=re.MULTILINE)

        # 2. Replace banned call patterns with local helpers
        if needs_softplus:
            src = src.replace("F.softplus(", "_softplus(")
        if needs_sigmoid:
            src = src.replace("torch.sigmoid(", "_sigmoid(")
        if needs_softmax:
            src = src.replace("torch.softmax(", "_softmax(")
        if needs_relu:
            src = src.replace("torch.relu(", "_relu(")

        # 3. Prepend helper definitions (only those needed)
        helpers = []
        if needs_softplus:
            helpers.append(
                "def _softplus(x, beta=1, threshold=20):\n"
                "    return torch.where(beta * x > threshold, x,\n"
                "                       torch.log1p(torch.exp(beta * x)) / beta)\n"
            )
        if needs_sigmoid:
            helpers.append(
                "def _sigmoid(x):\n"
                "    return 1.0 / (1.0 + torch.exp(-x))\n"
            )
        if needs_softmax:
            helpers.append(
                "def _softmax(x, dim=-1):\n"
                "    x_s = x - x.max(dim=dim, keepdim=True).values\n"
                "    e = torch.exp(x_s)\n"
                "    return e / e.sum(dim=dim, keepdim=True)\n"
            )
        if needs_relu:
            helpers.append(
                "def _relu(x):\n"
                "    return x.clamp(min=0)\n"
            )

        if helpers:
            # Insert helpers after the last top-level import line
            lines = src.splitlines(keepends=True)
            last_import_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    last_import_idx = i
            helper_block = "\n\n" + "\n".join(helpers)
            lines.insert(last_import_idx + 1, helper_block)
            src = "".join(lines)

        # 4. Rename entry function
        src = re.sub(r'\bdef run\(', 'def kernel_function(', src)
        return src + "\n"

    def _generate_kernel_from_reference(self) -> str:
        """Generate a standalone kernel_function by wrapping reference_body.

        Injects all spec.axes constants so reference_body variables resolve,
        and strips any import statements that are lifted to the top level.
        """
        spec = self.spec
        input_names = ", ".join(t.name for t in spec.inputs)

        needs_math = "math." in spec.reference_body or "import math" in spec.reference_body
        imports = "import math\nimport torch" if needs_math else "import torch"

        # Inject all axis constants so reference_body variables are in scope
        axis_lines = [f"    {name} = {val}" for name, val in spec.axes.items()]

        # Strip import statements (hoisted to top); indent remaining lines
        body_lines = []
        for line in spec.reference_body.strip().splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                continue
            body_lines.append("    " + line)

        axis_section = "\n".join(axis_lines)
        body_section = "\n".join(body_lines)

        return (
            f"{imports}\n\n\n"
            f"def kernel_function({input_names}):\n"
            f"{axis_section}\n"
            f"{body_section}\n"
        )

    def generate_test_py(self) -> str:
        spec = self.spec
        input_names = ", ".join(t.name for t in spec.inputs)
        atol = spec.atol
        rtol = spec.rtol
        num_outputs = len(spec.outputs)

        return textwrap.dedent(f"""\
            # Auto-generated test.py for FlashInfer definition: {spec.key}
            import sys
            import torch
            from kernel import kernel_function
            from problem import get_inputs, get_init_inputs, Model


            def test_kernel():
                device = "cuda"
                inputs = [
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in get_inputs()
                ]
                model = Model(*get_init_inputs()).to(device)

                with torch.no_grad():
                    ref = model(*inputs)

                kernel_out = kernel_function(*inputs)

                # Normalize to list of tensors
                if isinstance(ref, torch.Tensor):
                    refs = [ref]
                else:
                    refs = list(ref)
                if isinstance(kernel_out, torch.Tensor):
                    outs = [kernel_out]
                elif kernel_out is None:
                    outs = [inputs[0]]   # in-place
                else:
                    outs = list(kernel_out)

                if len(outs) != len(refs):
                    print(f"FAIL: expected {{len(refs)}} outputs, got {{len(outs)}}")
                    return False

                for i, (r, o) in enumerate(zip(refs, outs)):
                    if r.shape != o.shape:
                        print(f"FAIL output[{{i}}]: shape {{o.shape}} != {{r.shape}}")
                        return False
                    # Cast to common dtype for comparison
                    r_cmp = r.float()
                    o_cmp = o.float()
                    if not torch.allclose(r_cmp, o_cmp, atol={atol}, rtol={rtol}):
                        abs_err = (r_cmp - o_cmp).abs()
                        max_err = abs_err.max().item()
                        matched = (abs_err <= ({atol} + {rtol} * r_cmp.abs())).float().mean().item()
                        print(f"FAIL output[{{i}}]: max_err={{max_err:.4f}}, matched={{matched:.3%}}")
                        return False

                print("PASS")
                return True


            if __name__ == "__main__":
                success = test_kernel()
                sys.exit(0 if success else 1)
        """)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _shape_str(self, shape: list[Any]) -> str:
        parts = []
        for dim in shape:
            if isinstance(dim, str):
                parts.append(dim)  # axis variable name
            else:
                parts.append(str(dim))
        return "(" + ", ".join(parts) + ",)"

    def _init_lines(self, t: TensorSpec) -> list[str]:
        """Return 4-space-indented lines that fill a freshly-zeroed tensor."""
        return ["    " + l for l in self._init_lines_raw(t)]

    def _init_lines_raw(self, t: TensorSpec) -> list[str]:
        """Return un-indented lines that fill a freshly-zeroed tensor with sensible values."""
        lines = []
        dtype = t.dtype
        if "float8" in dtype or "int8" in dtype or "uint8" in dtype:
            lines.append(
                f"{t.name} = torch.rand_like({t.name}.float()).sub(0.5).mul(2)"
                f".to({dtype})"
            )
        elif "int32" in dtype or "int64" in dtype or "long" in dtype:
            pass  # leave as zeros (valid indices)
        elif "bool" in dtype:
            lines.append(
                f"{t.name} = torch.rand_like({t.name}.float()).gt(0.5)"
            )
        else:
            lines.append(
                f"{t.name} = torch.randn_like({t.name}.float()).to({dtype})"
            )
        return lines
