# humanize — Claude Code Skill: Humanize Workflow Orchestrator

Repo-local entry point for [PolyArch/humanize](https://github.com/PolyArch/humanize)-style
developer workflows on KernelAgent: repo-grounded idea generation, planning,
plan refinement, RLCR implementation loops, and Codex-reviewed change
proposals.

Humanize is developer workflow tooling, not part of KernelAgent's runtime.
It must never be added as a runtime dependency, and must not be used as
part of generated kernel execution, benchmarking, profiling, or production
runtime — see `SKILL.md` for the full scope and runtime-boundary rules.

## Architecture

```
humanize/
└── SKILL.md   # Entry point: scope, invocation, workflow mapping, review checklist
```

Unlike the `ka-kernel-*` skills, this skill has no `steps/` or `tools/` —
it is a single guidance document that Claude Code (and Codex, as an
independent reviewer) read directly to stay consistent about KernelAgent's
subsystems, target platforms/languages, and validation expectations when
running Humanize workflows.

## Usage

Invoke with natural language, e.g.:

- "Use the humanize skill to generate a plan from `<draft.md>`"
- "Use the humanize skill to ask Codex to review this KernelAgent change"

If the PolyArch/humanize plugin runtime is installed, requests map to the
equivalent `/humanize:*` command (`gen-idea`, `gen-plan`, `refine-plan`,
`start-rlcr-loop`, `ask-codex`). If it is not installed, `SKILL.md` still
provides planning/review structure and states which Humanize runtime
command would be required.
