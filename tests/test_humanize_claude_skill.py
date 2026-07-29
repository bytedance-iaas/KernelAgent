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

"""Verify the Claude Code Humanize skill entry point stays a thin, repo-local wrapper."""

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CLAUDE_HUMANIZE_SKILL = _REPO_ROOT / ".claude/skills/humanize/SKILL.md"
_ROOT_CLAUDE_GUIDANCE = _REPO_ROOT / "CLAUDE.md"


def _read_claude_humanize_skill() -> str:
    assert _CLAUDE_HUMANIZE_SKILL.exists(), (
        "Claude Code must have a repo-local Humanize skill entry point"
    )
    return _CLAUDE_HUMANIZE_SKILL.read_text(encoding="utf-8")


def test_claude_code_humanize_skill_entrypoint_points_to_guidance():
    skill = _read_claude_humanize_skill()

    assert re.search(r"(?m)^---$", skill), "Claude Code skill must have frontmatter"
    assert re.search(r"(?m)^name:\s*humanize\s*$", skill), (
        "Claude Code skill must be named humanize"
    )
    assert re.search(r"(?m)^allowed-tools:\s*$", skill), (
        "Claude Code skill must declare allowed tools like the existing repo skills"
    )
    assert re.search(r"(?m)^\s+- Bash\s*$", skill), (
        "Claude Code Humanize skill must allow Bash for Humanize runtime scripts"
    )
    assert "skillset/skills/humanize.md" in skill, (
        "Claude Code Humanize skill must point to skillset/skills/humanize.md"
    )
    assert "Do not proceed until you have read" in skill, (
        "Claude Code Humanize skill must follow the repo skill wrapper pattern"
    )
    assert "$ARGUMENTS" in skill, "Claude Code Humanize skill must pass user arguments through"
    assert re.search(r"fallback behavior", skill, re.IGNORECASE), (
        "Skill must delegate runtime fallback behavior to skillset/skills/humanize.md"
    )


def test_claude_humanize_skill_is_not_ignored_by_git():
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", ".claude/skills/humanize/SKILL.md"],
        cwd=_REPO_ROOT,
        check=False,
    )

    assert result.returncode == 1, ".claude/skills/humanize/SKILL.md must not be ignored by git"


def test_root_claude_guidance_is_absent_and_ignored_for_local_overrides():
    assert not _ROOT_CLAUDE_GUIDANCE.exists(), (
        "root CLAUDE.md should not be committed because it can affect local agent settings"
    )

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", "CLAUDE.md"],
        cwd=_REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0, "root CLAUDE.md should remain ignored for local overrides"
