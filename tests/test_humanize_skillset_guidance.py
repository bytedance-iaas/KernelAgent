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

"""Verify that the Humanize skillset guidance remains repo-local and pointer-oriented."""

import re
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    tomllib = None


_REPO_ROOT = Path(__file__).resolve().parent.parent
_HUMANIZE_GUIDANCE = _REPO_ROOT / "skillset/skills/humanize.md"

_CANONICAL_FILES = [
    "triton_kernel_agent/templates/triton_guidelines.j2",
    "triton_kernel_agent/templates/backend/cutedsl/guidelines.j2",
    "triton_kernel_agent/templates/tilelang/guidelines.j2",
    "triton_kernel_agent/platform_config.py",
    "triton_kernel_agent/kernel_backend_config.py",
    "Fuser/config/autoagent_default.yml",
    "triton_kernel_agent/templates/kernel_optimization.j2",
    "triton_kernel_agent/templates/reflexion_prompt.j2",
    "CONTRIBUTING.md",
    "README.md",
]

_CANONICAL_DIRS = [
    "triton_kernel_agent/templates/backend/cutedsl/",
    "triton_kernel_agent/templates/tilelang/",
    "examples/configs/",
]

_REQUIRED_GUIDANCE_PATTERNS = {
    "Humanize is described as Claude Code-compatible": r"Claude Code-compatible",
    "Humanize has a Claude Code skill entry point": r"\.claude/skills/humanize/SKILL\.md",
    "Humanize is positioned as skill-plugin workflow support": r"skill-plugin",
    "Humanize is positioned for single-agent sessions": r"single-agent sessions",
    "Humanize supports human-in-the-loop planning/review": r"human-in-the-loop",
    "Humanize stays outside KernelAgent runtime": r"does not run inside KernelAgent",
    "Humanize is excluded from runtime dependencies": r"runtime dependenc(?:y|ies)",
    "Plans include affected-subsystem context": r"affects kernel generation",
    "Plans include target-platform context": r"target platform",
    "Plans include target-backend context": r"target kernel backend",
    "Plans include Release 1 SM90 FP8 GEMM context": r"SM90 FP8 GEMM",
    "Plans include artifact-change context": r"artifact changes",
    "Plans include validation context": r"validation plan",
    "Reviews include cuteDSL backend context": r"cuteDSL",
    "Reviews include TileLang backend context": r"TileLang",
    "Plans mention example config validation templates": r"examples/configs/",
}

_SUPPORTED_HUMANIZE_WORKFLOWS = [
    "/humanize:gen-idea",
    "/humanize:gen-plan",
    "/humanize:refine-plan",
    "/humanize:start-rlcr-loop",
    "/humanize:ask-codex",
]


def _read_guidance() -> str:
    assert _HUMANIZE_GUIDANCE.exists(), (
        "skillset/skills/humanize.md must exist as the authoritative guidance"
    )
    return _HUMANIZE_GUIDANCE.read_text(encoding="utf-8")


def _dependency_name(dependency: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", dependency.strip())
    return match.group(0).lower().replace("_", "-") if match else ""


def _project_dependencies(pyproject_text: str) -> list[str]:
    if tomllib is not None:
        pyproject = tomllib.loads(pyproject_text)
        return pyproject["project"].get("dependencies", [])

    project_match = re.search(r"(?ms)^\[project\]\s*(.*?)(?:^\[|\Z)", pyproject_text)
    assert project_match, "pyproject.toml must contain a [project] section"

    dependencies_match = re.search(
        r"(?ms)^dependencies\s*=\s*\[(.*?)^\]",
        project_match.group(1),
    )
    assert dependencies_match, "pyproject.toml [project] section must contain dependencies"

    return [
        match.group(2)
        for match in re.finditer(r"(['\"])(.*?)\1", dependencies_match.group(1))
    ]


def test_humanize_guidance_references_existing_canonical_sources():
    guidance = _read_guidance()

    for relative_path in _CANONICAL_FILES:
        assert relative_path in guidance, f"humanize guidance must reference {relative_path}"
        assert (_REPO_ROOT / relative_path).is_file(), (
            f"Referenced file does not exist: {relative_path}"
        )

    for relative_path in _CANONICAL_DIRS:
        assert relative_path in guidance, f"humanize guidance must reference {relative_path}"
        assert (_REPO_ROOT / relative_path).is_dir(), (
            f"Referenced directory does not exist: {relative_path}"
        )


def test_humanize_guidance_keeps_runtime_boundary_context():
    guidance = _read_guidance()

    for description, pattern in _REQUIRED_GUIDANCE_PATTERNS.items():
        assert re.search(pattern, guidance, re.IGNORECASE), (
            f"humanize guidance must show: {description}"
        )


def test_humanize_guidance_documents_supported_workflows():
    guidance = _read_guidance()

    assert "Supported Humanize Workflows" in guidance
    for workflow in _SUPPORTED_HUMANIZE_WORKFLOWS:
        assert workflow in guidance, f"humanize guidance must document {workflow}"

    assert re.search(r"repo-grounded idea", guidance, re.IGNORECASE), (
        "gen-idea guidance must keep ideas grounded in this repository"
    )
    assert re.search(r"reviewer notes|Codex findings", guidance, re.IGNORECASE), (
        "refine-plan guidance must preserve reviewer/Codex refinement context"
    )
    assert re.search(r"default independent reviewer", guidance, re.IGNORECASE), (
        "ask-codex guidance must describe Codex as the independent reviewer path"
    )


def test_humanize_skillset_guidance_is_not_ignored_by_git():
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "skillset/skills/humanize.md"],
        cwd=_REPO_ROOT,
        check=False,
    )

    assert result.returncode == 1, "skillset/skills/humanize.md must not be ignored by git"


def test_humanize_is_not_a_runtime_dependency():
    dependencies = _project_dependencies(
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    dependency_names = {_dependency_name(dependency) for dependency in dependencies}
    dependency_text = "\n".join(dependencies).lower()

    assert "humanize" not in dependency_names, "humanize must not be a runtime dependency"
    assert "polyarch/humanize" not in dependency_text, (
        "PolyArch/humanize must not be referenced from runtime dependencies"
    )
