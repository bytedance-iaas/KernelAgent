# Humanize Guidance For KernelAgent

Use this when running PolyArch/humanize workflows on KernelAgent tasks. Humanize is a Claude Code-compatible workflow aid that can be invoked through the repo-local `.claude/skills/humanize/SKILL.md` skill or through Humanize plugin slash commands when an agent runtime exposes them; it does not run inside KernelAgent and should not be added as a runtime dependency.

Humanize uses Codex as an independent reviewer during its review loop, so review checklist items below are intended for both the developer preparing the plan and Codex evaluating the implementation.

## Scope

- Use Humanize for repository changes: Python orchestration, Fuser pipeline changes, optimization-loop logic, prompts/templates, tests, examples, docs, and CI.
- Treat Humanize as developer workflow support for Claude Code skills, skill-plugin sessions, single-agent sessions, and human-in-the-loop planning/review, not as the hosted multi-agent service or kernel hub deliverable.
- Do not use Humanize as part of generated kernel execution, benchmarking, profiling, or production runtime.
- Treat existing correctness tests, benchmark gates, and profiling data as authoritative over reviewer opinions.

## Claude Code Entry Point

- Claude Code should use `.claude/skills/humanize/SKILL.md` as the repo-local skill entry point, then read this file for the authoritative KernelAgent-specific contract.
- In Claude Code, users may invoke the skill with natural language such as "Use the humanize skill to generate a plan from `<draft.md>`" or "Use the humanize skill to ask Codex to review this KernelAgent change".
- If the PolyArch/humanize plugin runtime is installed, map those requests to the equivalent `/humanize:*` or Humanize script workflow. If it is not installed, still follow this file for planning/review structure and explain which Humanize runtime command would be required.
- Keep `.claude/skills/humanize/SKILL.md` as a thin Claude Code wrapper with frontmatter, `allowed-tools`, a critical instruction to read `skillset/skills/humanize.md`, `$ARGUMENTS` passthrough, and no duplicated workflow policy.

## Context To Include In Humanize Drafts/Plans

- State whether the change affects kernel generation, Fuser orchestration, optimization, profiling, benchmarking, UI scripts, examples, or packaging.
- Include the target platform: CUDA, XPU, or platform-neutral.
- Include the target kernel backend when relevant: Triton, cuteDSL, TileLang, or backend-neutral.
- For Release #1 work, state whether the task targets SM90 FP8 GEMM and which of the Triton/cuteDSL/TileLang generation paths it exercises.
- Include expected artifact changes under `.fuse/`, `.optimize/`, `triton_kernel_logs/`, or worker artifact directories.
- Include the validation plan: relevant `CONTRIBUTING.md` test/example commands, import smoke checks if applicable, and baseline/post-change performance expectations for performance-sensitive changes.
- Use `examples/configs/` as validation run templates when relevant, such as running `examples/run_opt_manager.py` with an example kernel directory and config.

## Supported Humanize Workflows

- `/humanize:gen-idea`: use for repo-grounded idea discovery before a concrete plan exists. Ideas should identify the affected KernelAgent subsystem, target platform, target kernel backend, validation path, and whether the idea advances Release #1 SM90 FP8 GEMM, multi-DSL generation, or optimization tooling.
- `/humanize:gen-plan`: use for turning a selected idea or draft into an implementation plan. Plans must include the context checklist above and keep Humanize outside KernelAgent runtime dependencies.
- `/humanize:refine-plan`: use when reviewer notes, user annotations, or Codex findings require plan cleanup. Refinement must preserve the target platform/backend context, runtime boundary, validation requirements, and original draft intent.
- `/humanize:start-rlcr-loop`: use for implementation rounds after a plan is ready. RLCR work should follow the plan's task routing, maintain deterministic validation, and treat Codex review feedback as a gate before completion.
- `/humanize:ask-codex`: use as the default independent reviewer for backend/template changes, Fuser orchestration changes, runtime dependency changes, verification changes, performance-sensitive changes, and Humanize guidance changes.

## Review Checklist For Codex/Humanize

- Correctness: enforce the runtime constraints in the selected backend guidance: `triton_kernel_agent/templates/triton_guidelines.j2`, `triton_kernel_agent/templates/backend/cutedsl/guidelines.j2`, or `triton_kernel_agent/templates/tilelang/guidelines.j2`.
- Verification: changes should preserve strict PASS/sentinel-based verification semantics and avoid weakening tests.
- Performance: benchmark/profiling changes should respect existing warmup/repeat, timeout, lock, and semaphore constraints.
- Prompt changes: avoid duplicating backend guidance templates; prefer references or targeted deltas.
- Platform support: when touching device allocation or backend behavior, state whether the change follows or updates `triton_kernel_agent/platform_config.py`.
- Kernel backend support: when touching Triton, cuteDSL, or TileLang generation paths, state whether the change follows or updates `triton_kernel_agent/kernel_backend_config.py`.
- Artifacts: preserve reproducibility and avoid changing artifact schemas without tests or migration notes.
- Dependencies: keep Humanize optional; do not add it to runtime dependencies.

## Existing Sources To Check Before Adding New Guidance

- `triton_kernel_agent/templates/triton_guidelines.j2` for Triton coding constraints and examples.
- `triton_kernel_agent/platform_config.py` for CUDA/XPU platform guidance.
- `triton_kernel_agent/kernel_backend_config.py` for Triton, cuteDSL, and TileLang backend routing and composition requirements.
- `Fuser/config/autoagent_default.yml` for default Fuser routing, target platform, and kernel backend settings.
- `triton_kernel_agent/templates/backend/cutedsl/` for cuteDSL generation/refinement/guideline templates.
- `triton_kernel_agent/templates/tilelang/` for TileLang generation/refinement/guideline templates.
- `triton_kernel_agent/templates/kernel_optimization.j2` for optimization prompt context.
- `triton_kernel_agent/templates/reflexion_prompt.j2` for review/reflection expectations.
- `CONTRIBUTING.md` for tests, style, PR expectations, and performance notes.
- `README.md` for supported workflows and artifact locations.
