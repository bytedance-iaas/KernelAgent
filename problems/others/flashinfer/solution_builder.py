"""Builds a flashinfer-bench solution.json from a kernel + definition spec.

The submission format is:
    {
        "name": "...",
        "definition": "...",
        "author": "...",
        "spec": {
            "language": "python",
            "target_hardware": "<set via flashinfer_target_hardware in yaml>",
            "entry_point": "kernel.py::kernel_function",
            "destination_passing_style": false,
            "dependencies": []
        },
        "sources": [
            {"path": "kernel.py", "content": "..."}
        ]
    }

The kernel_function signature must match:
    kernel_function(*inputs) -> tensor
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SolutionBuilder:
    """Assemble a flashinfer-bench solution.json.

    Parameters
    ----------
    definition_key:
        The flashinfer-bench definition name (e.g. "moe_fp8_block_scale_e256_...").
    kernel_code:
        Python source of the optimized kernel (must define kernel_function).
    author:
        Submission author string.
    name:
        Human-readable solution name (defaults to "{definition_key} by {author}").
    target_hardware:
        Hardware target string, e.g. "nvidia_h200" or "nvidia_b200".
        Set via flashinfer_target_hardware in the strategy yaml config.
    dependencies:
        List of pip-installable package names needed by the kernel.
    extra_sources:
        Additional {"path": ..., "content": ...} entries beyond kernel.py.
    """

    def __init__(
        self,
        definition_key: str,
        kernel_code: str,
        author: str = "kernelAgent",
        name: str = "",
        target_hardware: str = "",
        dependencies: list[str] | None = None,
        extra_sources: list[dict[str, str]] | None = None,
    ):
        if not target_hardware:
            raise ValueError(
                "target_hardware is required (e.g. 'nvidia_h200'). "
                "Set flashinfer_target_hardware in your strategy yaml config."
            )
        self.definition_key = definition_key
        self.kernel_code = kernel_code
        self.author = author
        self.name = name or f"{definition_key} by {author}"
        self.target_hardware = target_hardware
        self.dependencies = dependencies or []
        self.extra_sources = extra_sources or []

    def build(self) -> dict[str, Any]:
        """Return the solution dict ready for JSON serialization."""
        sources = [{"path": "kernel.py", "content": self.kernel_code}]
        sources.extend(self.extra_sources)

        # target_hardware must be a list per flashinfer-bench BuildSpec schema
        hw = self.target_hardware if isinstance(self.target_hardware, list) else [self.target_hardware]

        return {
            "name": self.name,
            "definition": self.definition_key,
            "author": self.author,
            "spec": {
                "language": "python",
                "target_hardware": hw,
                "entry_point": "kernel.py::kernel_function",
                "destination_passing_style": False,
                "dependencies": self.dependencies,
            },
            "sources": sources,
        }

    def save(self, output_path: Path) -> Path:
        """Write solution.json to output_path (creates parents as needed).

        If output_path is a directory, writes solution.json inside it.
        Returns the actual file path written.
        """
        output_path = Path(output_path)
        if output_path.is_dir() or not output_path.suffix:
            output_path = output_path / "solution.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        solution = self.build()
        output_path.write_text(json.dumps(solution, indent=2) + "\n")
        return output_path

    @classmethod
    def from_problem_dir(
        cls,
        problem_dir: Path,
        definition_key: str,
        kernel_filename: str = "kernel.py",
        **kwargs: Any,
    ) -> "SolutionBuilder":
        """Load kernel source from a KernelAgent problem directory.

        Looks for `kernel_filename` (the optimized kernel) in problem_dir.
        """
        kernel_path = Path(problem_dir) / kernel_filename
        if not kernel_path.exists():
            # Fall back to input.py (initial kernel)
            kernel_path = Path(problem_dir) / "input.py"
        kernel_code = kernel_path.read_text()
        return cls(definition_key=definition_key, kernel_code=kernel_code, **kwargs)
