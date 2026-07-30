#!/usr/bin/env python3
"""
Render Jinja2 templates from the triton_kernel_agent/templates/ directory.

Wraps the existing PromptManager to render prompt templates with provided
variables, outputting the rendered text to stdout.

Usage:
    python render_template.py --template test_generation \
        --vars '{"problem_description": "...", "device_string": "cuda"}'

    python render_template.py --template backend_guidelines \
        --vars '{"target_platform": "cuda", "kernel_backend": "triton"}'

    python render_template.py --template kernel_generation \
        --vars '{"problem_description": "...", "test_code": "...", "no_cusolver": false}'

Available templates: test_generation, kernel_generation, kernel_refinement,
                     kernel_optimization, backend_guidelines

Output:
    Rendered prompt text to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path so we can import from triton_kernel_agent
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from triton_kernel_agent.platform_config import get_platform
from triton_kernel_agent.kernel_backend_config import get_kernel_backend
from triton_kernel_agent.prompt_manager import PromptManager


def _render_template(template_name: str, variables: dict) -> str:
    """Render a template using the project's PromptManager."""

    # Use PromptManager for proper rendering
    platform = variables.pop("target_platform", "cuda")
    platform_config = get_platform(platform)

    kernel_backend = variables.pop("kernel_backend", "triton")
    _ = get_kernel_backend(kernel_backend) # Just to validate

    pm = PromptManager(target_platform=platform_config, kernel_backend=kernel_backend)

    if template_name == "test_generation":
        return pm.render_test_generation_prompt(
            problem_description=variables.get("problem_description", ""),
            provided_test_code=variables.get("provided_test_code"),
        )
    elif template_name == "kernel_generation":
        #TODO: Don't support custom guideslines for now
        return pm.render_kernel_generation_prompt(
            problem_description=variables.get("problem_description", ""),
            test_code=variables.get("test_code", ""),
            backend_guidelines=None,
            no_cusolver=variables.get("no_cusolver", False),
        )
    elif template_name == "kernel_refinement":
        #TODO: Don't support custom guideslines for now
        return pm.render_kernel_refinement_prompt(
            problem_description=variables.get("problem_description", ""),
            test_code=variables.get("test_code", ""),
            kernel_code=variables.get("kernel_code", ""),
            error_info=variables.get("error_info", {}),
            history_context=variables.get("history_context"),
            backend_guidelines=None,
            no_cusolver=variables.get("no_cusolver", False),
        )
    elif template_name == "backend_guidelines":
        return pm.render_backend_guidelines()
    elif template_name == "kernel_optimization":
        return pm.render_kernel_optimization_prompt(
            problem_description=variables.get("problem_description", ""),
            kernel_code=variables.get("kernel_code", ""),
            gpu_specs=variables.get("gpu_specs", {}),
            roofline=variables.get("roofline", {}),
            category=variables.get("category", ""),
            summary=variables.get("summary", ""),
            reasoning=variables.get("reasoning", ""),
            root_cause=variables.get("root_cause", {}),
            recommended_fix=variables.get("recommended_fix", {}),
            pytorch_baseline_ms=variables.get("pytorch_baseline_ms"),
            current_best_ms=variables.get("current_best_ms"),
            error_feedback=variables.get("error_feedback"),
            recent_attempts=variables.get("recent_attempts"),
            reflexions=variables.get("reflexions"),
            rag_context=variables.get("rag_context"),
        )
    else:
        # Generic rendering via PromptManager.render_custom_template
        return pm.render_custom_template(template_name, **variables)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Render Jinja2 prompt templates from triton_kernel_agent"
    )
    p.add_argument(
        "--template",
        required=True,
        help="Template name (e.g., test_generation, kernel_generation, backend_guidelines, etc.)",
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
        "--output",
        default=None,
        help="Output file path (prints to stdout if not set)",
    )
    args = p.parse_args(argv)

    if args.vars_file:
        variables = json.loads(Path(args.vars_file).read_text(encoding="utf-8"))
    else:
        variables = json.loads(args.vars)

    rendered = _render_template(args.template, variables)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"Rendered to: {args.output}", file=sys.stderr)
    else:
        print(rendered)

    return 0


if __name__ == "__main__":
    sys.exit(main())
