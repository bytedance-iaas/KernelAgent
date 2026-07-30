#!/usr/bin/env python3
"""
Render Jinja2 prompt templates from the ka-kernel-gen skill's templates/ directory.

Standalone: depends only on the Python standard library and jinja2
(`pip install jinja2`). Does NOT import the triton_kernel_agent package,
so the skill folder can be copied to any machine/project and used as-is.

Usage:
    python render_template.py --template test_generation \
        --vars '{"problem_description": "...", "target_platform": "cuda"}'

    python render_template.py --template language_guidelines \
        --vars '{"kernel_language": "tilelang"}'

    python render_template.py --template kernel_generation \
        --vars '{"problem_description": "...", "test_code": "...",
                 "kernel_language": "triton", "no_cusolver": false}'

Available templates: test_generation, kernel_generation, kernel_refinement,
                     kernel_optimization, language_guidelines

Template directory resolution order:
    1. --templates-dir CLI flag
    2. KERNELAGENT_TEMPLATES_DIR environment variable
    3. <skill dir>/templates (sibling of this script's directory)

Output:
    Rendered prompt text to stdout (or --output file).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:  # pragma: no cover
    print(
        "error: jinja2 is required. Install it with: pip install jinja2",
        file=sys.stderr,
    )
    sys.exit(2)

_DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


# ---------------------------------------------------------------------------
# Platform configuration (vendored subset of
# triton_kernel_agent/platform_config.py — keep in sync)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlatformConfig:
    name: str
    device_string: str
    kernel_guidance: str


_XPU_KERNEL_GUIDANCE = """\
## Intel XPU-Specific Optimizations

You are generating a Triton kernel for Intel XPU (Xe GPUs). Follow these guidelines:

1. **Device Context**: Use 'xpu' as the device instead of 'cuda'
2. **Memory Hierarchy**: Intel Xe has different cache sizes - optimize accordingly
3. **Thread Configuration**:
   - Subgroup size is typically 8, 16, or 32 (flexible)
   - num_warps: typically 4, 8, or 16 for Intel GPUs
   - BLOCK_SIZE: prefer 64, 128, 256, or 512
4. **Optimal Block Sizes**: Start with 128-256 for most kernels
5. **Data Types**: Intel supports fp32, fp16, bf16 (fp8 varies by generation)"""

PLATFORMS: dict[str, PlatformConfig] = {
    "cuda": PlatformConfig(name="cuda", device_string="cuda", kernel_guidance=""),
    "xpu": PlatformConfig(
        name="xpu", device_string="xpu", kernel_guidance=_XPU_KERNEL_GUIDANCE
    ),
}


# ---------------------------------------------------------------------------
# Kernel language configuration (vendored subset of
# triton_kernel_agent/kernel_language_config.py — keep in sync)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelLanguageConfig:
    name: str
    display_name: str
    generation_template: str
    refinement_template: str
    guidelines_template: str


KERNEL_LANGUAGES: dict[str, KernelLanguageConfig] = {
    "triton": KernelLanguageConfig(
        name="triton",
        display_name="Triton",
        generation_template="backend/triton/kernel_generation.j2",
        refinement_template="backend/triton/kernel_refinement.j2",
        guidelines_template="backend/triton/guidelines.j2",
    ),
    "tilelang": KernelLanguageConfig(
        name="tilelang",
        display_name="TileLang",
        generation_template="backend/tilelang/kernel_generation.j2",
        refinement_template="backend/tilelang/kernel_refinement.j2",
        guidelines_template="backend/tilelang/guidelines.j2",
    ),
    "cutedsl": KernelLanguageConfig(
        name="cutedsl",
        display_name="cuteDSL",
        generation_template="backend/cutedsl/kernel_generation.j2",
        refinement_template="backend/cutedsl/kernel_refinement.j2",
        guidelines_template="backend/cutedsl/guidelines.j2",
    ),
}


def _get_platform(name: str) -> PlatformConfig:
    if name not in PLATFORMS:
        available = ", ".join(sorted(PLATFORMS.keys()))
        raise ValueError(f"Unknown platform '{name}'. Available: {available}")
    return PLATFORMS[name]


def _get_kernel_language(name: str) -> KernelLanguageConfig:
    key = name.strip().lower()
    if key not in KERNEL_LANGUAGES:
        available = ", ".join(sorted(KERNEL_LANGUAGES.keys()))
        raise ValueError(f"Unknown kernel language '{name}'. Available: {available}")
    return KERNEL_LANGUAGES[key]


class TemplateRenderer:
    """Minimal standalone replacement for triton_kernel_agent.PromptManager."""

    def __init__(
        self,
        templates_dir: Path,
        platform: PlatformConfig,
        language: KernelLanguageConfig,
    ):
        if not templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {templates_dir}")
        self.templates_dir = templates_dir
        self.platform = platform
        self.language = language
        self.env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _render(self, template_file: str, **variables) -> str:
        return self.env.get_template(template_file).render(**variables)

    def render_language_guidelines(self) -> str:
        return self._render(self.language.guidelines_template)

    def render_test_generation(self, variables: dict) -> str:
        return self._render(
            "test_generation.j2",
            problem_description=variables.get("problem_description", ""),
            provided_test_code=variables.get("provided_test_code"),
            device_string=self.platform.device_string,
            kernel_language=self.language.name,
            kernel_language_display=self.language.display_name,
        )

    def render_kernel_generation(self, variables: dict) -> str:
        guidelines = self.render_language_guidelines()
        return self._render(
            self.language.generation_template,
            problem_description=variables.get("problem_description", ""),
            test_code=variables.get("test_code", ""),
            # Backend templates are inconsistent about the variable name;
            # pass the guidelines under every name in use.
            guidelines=guidelines,
            backend_guidelines=guidelines,
            language_guidelines=guidelines,
            kernel_guidance=self.platform.kernel_guidance,
            no_cusolver=variables.get("no_cusolver", False),
        )

    def render_kernel_refinement(self, variables: dict) -> str:
        guidelines = self.render_language_guidelines()
        return self._render(
            self.language.refinement_template,
            problem_description=variables.get("problem_description", ""),
            test_code=variables.get("test_code", ""),
            kernel_code=variables.get("kernel_code", ""),
            error_info=variables.get("error_info", {}),
            history_context=variables.get("history_context"),
            guidelines=guidelines,
            backend_guidelines=guidelines,
            language_guidelines=guidelines,
            kernel_guidance=self.platform.kernel_guidance,
            no_cusolver=variables.get("no_cusolver", False),
        )

    def render_kernel_optimization(self, variables: dict) -> str:
        bottleneck = {
            "category": variables.get("category", ""),
            "summary": variables.get("summary", ""),
            "reasoning": variables.get("reasoning", ""),
            "root_cause": variables.get("root_cause", {}),
            "recommended_fix": variables.get("recommended_fix", {}),
        }
        return self._render(
            "kernel_optimization.j2",
            problem_description=variables.get("problem_description", ""),
            kernel_code=variables.get("kernel_code", ""),
            gpu_specs=variables.get("gpu_specs", {}),
            roofline=variables.get("roofline", {}),
            bottleneck=bottleneck,
            pytorch_baseline_ms=variables.get("pytorch_baseline_ms"),
            current_best_ms=variables.get("current_best_ms"),
            error_feedback=variables.get("error_feedback"),
            recent_attempts=variables.get("recent_attempts"),
            reflexions=variables.get("reflexions"),
            rag_context=variables.get("rag_context"),
            grid_analysis=variables.get("grid_analysis"),
            kernel_language=self.language.name,
            guidelines=self.render_language_guidelines(),
        )

    def render_generic(self, template_name: str, variables: dict) -> str:
        """Render an arbitrary .j2 file from the templates directory."""
        template_file = template_name
        if not template_file.endswith(".j2"):
            template_file += ".j2"
        return self._render(template_file, **variables)


def _render_template(
    template_name: str, variables: dict, templates_dir: Path
) -> str:
    platform = _get_platform(variables.pop("target_platform", "cuda"))
    # "kernel_backend" is accepted as a legacy alias for "kernel_language".
    language_name = variables.pop(
        "kernel_language", variables.pop("kernel_backend", "triton")
    )
    language = _get_kernel_language(language_name)

    renderer = TemplateRenderer(templates_dir, platform, language)

    if template_name == "test_generation":
        return renderer.render_test_generation(variables)
    elif template_name == "kernel_generation":
        return renderer.render_kernel_generation(variables)
    elif template_name == "kernel_refinement":
        return renderer.render_kernel_refinement(variables)
    elif template_name == "kernel_optimization":
        return renderer.render_kernel_optimization(variables)
    elif template_name in ("language_guidelines", "backend_guidelines"):
        return renderer.render_language_guidelines()
    else:
        return renderer.render_generic(template_name, variables)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Render Jinja2 prompt templates from the skill's templates directory"
    )
    p.add_argument(
        "--template",
        required=True,
        help="Template name (test_generation, kernel_generation, kernel_refinement, "
        "kernel_optimization, language_guidelines, or a .j2 file name)",
    )
    p.add_argument(
        "--vars",
        default="{}",
        help="JSON string of template variables",
    )
    p.add_argument(
        "--vars-file",
        default=None,
        help="Path to JSON file with template variables (overrides --vars)",
    )
    p.add_argument(
        "--templates-dir",
        default=None,
        help="Templates directory (default: $KERNELAGENT_TEMPLATES_DIR or "
        "the templates directory next to this script's directory)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output file path (prints to stdout if not set)",
    )
    args = p.parse_args(argv)

    if args.vars_file:
        variables = json.loads(Path(args.vars_file).read_text(encoding="utf-8"))
    else:
        variables = json.loads(args.vars)

    templates_dir = Path(
        args.templates_dir
        or os.environ.get("KERNELAGENT_TEMPLATES_DIR")
        or _DEFAULT_TEMPLATES_DIR
    )

    rendered = _render_template(args.template, variables, templates_dir)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"Rendered to: {args.output}", file=sys.stderr)
    else:
        print(rendered)

    return 0


if __name__ == "__main__":
    sys.exit(main())
