# ka-kernel-gen — Claude Code Skill for KernelAgent Kernel Generation

A standalone, self-contained Claude Code skill that replicates the KernelAgent
End-to-End Kernel Generation pipeline without relying on remote MCP servers,
external LLM API calls, or the KernelAgent Python package.

Everything the skill needs — instructions, tool scripts, and prompt templates —
lives inside this directory. Copy or symlink this one folder anywhere and it
works as-is. The only requirements are Python 3.9+ and `jinja2`.

## Architecture

Claude Code acts as the LLM "brain" via the skill files, while deterministic
Python scripts handle the "hands" work (AST analysis, code execution, template
rendering, JSON parsing, artifact management).

```
ka-kernel-gen/
├── SKILL.md                    # Entry point: mode dispatch + orchestration
├── steps/                      # Pipeline step instructions
│   ├── 00_route_problem.md     # Analyze complexity → pick route
│   ├── 01_fuse_model.md        # Rewrite PyTorch into fusable subgraphs
│   ├── 02_extract_subgraphs.md # Extract subgraph JSON from fused code
│   ├── 03_generate_kernel.md   # Generate kernel + test for one subgraph
│   ├── 04_refine_kernel.md     # Iterative verify-and-fix loop
│   └── 05_compose_kernels.md   # Stitch subgraph kernels together
├── tools/                      # Local Python scripts (invoked by the skill)
│   ├── analyze_problem.py      # Static AST analysis + routing recommendation
│   ├── run_candidate.py        # Execute Python file, classify PASS/FAIL
│   ├── extract_json.py         # Parse JSON from fenced code blocks
│   ├── dedup_subgraphs.py      # Deduplicate subgraphs by shape signature
│   ├── build_reference.py      # Build reference code from subgraph JSON
│   ├── render_template.py      # Render Jinja2 prompt templates (standalone)
│   └── manage_artifacts.py     # Create/manage run directories
├── templates/                  # Vendored Jinja2 prompt templates
│   ├── test_generation.j2
│   ├── kernel_optimization.j2
│   ├── reflexion_prompt.j2
│   └── backend/{triton,tilelang,cutedsl}/
│       ├── guidelines.j2
│       ├── kernel_generation.j2
│       └── kernel_refinement.j2
└── sync_templates.sh           # Re-vendor templates (KernelAgent repo only)
```

The skill references its bundled tools, templates, and step files via
`${CLAUDE_SKILL_DIR}` (the directory containing the active SKILL.md —
substituted by Claude Code), so it is location-independent.

## Installation

Claude Code discovers skills from a project's `.claude/skills/` directory.
A `.claude/skills/<name>` entry may be a **symlink**, so from any project root:

```bash
mkdir -p .claude/skills
ln -s /path/to/skills/ka-kernel-gen .claude/skills/ka-kernel-gen
```

(Copying the directory instead of symlinking also works — the skill is fully
self-contained.)

In the KernelAgent repo this is already set up: `.claude/skills/ka-kernel-gen`
is a symlink to `skills/ka-kernel-gen`.

## Usage

The skill dispatches on its arguments:

```
/ka-kernel-gen /path/to/KernelBench/level1/19_ReLU.py     # full flow: analyze → route → generate → verify
/ka-kernel-gen Implement ReLU over a 1D tensor of length 1024   # direct path from a text description
/ka-kernel-gen analyze /path/to/problem.py                # routing analysis only
/ka-kernel-gen refine /path/to/workdir                    # refinement loop on an existing kernel + test
```

Or run the tools directly:

```bash
python skills/ka-kernel-gen/tools/analyze_problem.py --problem /path/to/problem.py
python skills/ka-kernel-gen/tools/render_template.py --template language_guidelines \
  --vars '{"kernel_language": "tilelang"}'
```

## Mapping to Original Pipeline

| Original Component | Skill Step | Tool(s) |
|---|---|---|
| `Fuser.auto_agent` | `SKILL.md` + `steps/00_route_problem.md` | `analyze_problem.py` |
| `Fuser.orchestrator` + `Fuser.worker` | `steps/01_fuse_model.md` | `run_candidate.py` |
| `Fuser.subgraph_extractor` | `steps/02_extract_subgraphs.md` | `extract_json.py`, `dedup_subgraphs.py` |
| `Fuser.dispatch_kernel_agent` | `steps/03_generate_kernel.md` | `build_reference.py`, `render_template.py` |
| `triton_kernel_agent.worker` | `steps/04_refine_kernel.md` | `run_candidate.py` |
| `Fuser.compose_end_to_end` | `steps/05_compose_kernels.md` | `run_candidate.py` |
| `triton_kernel_agent.agent` | `SKILL.md` Path A (direct) | `render_template.py`, `run_candidate.py` |

## Keeping Templates in Sync

The Jinja2 templates under `templates/` are vendored copies of
`triton_kernel_agent/templates/`. When working inside the KernelAgent repo,
re-sync them after upstream template changes:

```bash
./skills/ka-kernel-gen/sync_templates.sh
```

(The script exits with a clear error when the skill is used outside the
KernelAgent repo — the vendored copies are then the source of truth.)

## Requirements

- Python 3.9+
- Jinja2 (`pip install jinja2`) — only needed by `render_template.py`;
  all other tools are stdlib-only
- GPU access for kernel verification (CUDA or XPU)
