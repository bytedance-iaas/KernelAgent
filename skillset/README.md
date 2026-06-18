# Claude Code Skills — KernelAgent Kernel Generation

This directory contains **Claude Code skills** and **local tool scripts** that replicate
the KernelAgent End-to-End Kernel Generation pipeline without relying on remote MCP
servers or external LLM API calls.

## Architecture

Claude Code acts as the LLM "brain" via skill files, while deterministic Python
scripts handle the "hands" work (AST analysis, code execution, template rendering,
JSON parsing, artifact management).

```
skillset/
├── skills/            # Claude Code skill markdown files
│   ├── main_kernel_gen.md      # Orchestrator: routes and chains all steps
│   ├── 00_route_problem.md     # Analyze problem complexity → pick route
│   ├── 01_fuse_model.md        # Rewrite PyTorch into fusable subgraphs
│   ├── 02_extract_subgraphs.md # Extract subgraph JSON from fused code
│   ├── 03_generate_kernel.md   # Generate Triton kernel for one subgraph
│   ├── 04_refine_kernel.md     # Iterative kernel refinement loop
│   ├── 05_compose_kernels.md   # Stitch subgraph kernels into final program
│   └── 06_direct_kernel.md     # Direct path bypassing Fuser
├── tools/             # Local Python scripts (invoked by skills)
│   ├── analyze_problem.py      # Static AST analysis + routing recommendation
│   ├── run_candidate.py        # Execute Python file, classify PASS/FAIL
│   ├── extract_json.py         # Parse JSON from fenced code blocks
│   ├── dedup_subgraphs.py      # Deduplicate subgraphs by shape signature
│   ├── build_reference.py      # Build reference code from subgraph JSON
│   ├── render_template.py      # Render Jinja2 prompt templates
│   └── manage_artifacts.py     # Create/manage run directories
└── templates/
    └── README.md               # Template sourcing strategy
```

## Quick Start

### 1. Analyze a Problem

```bash
python skillset/tools/analyze_problem.py \
  --problem /path/to/KernelBench/level1/19_ReLU.py
```

### 2. Use with Claude Code

Point Claude Code at the relevant skill:

```
@skill skillset/skills/main_kernel_gen.md
Generate a Triton kernel for /path/to/problem.py
```

Or invoke individual pipeline steps:

```
@skill skillset/skills/06_direct_kernel.md
Generate a Triton kernel for: "Implement ReLU over a 1D tensor of length 1024"
```

## Mapping to Original Pipeline

| Original Component | Skill | Tool(s) |
|---|---|---|
| `Fuser.auto_agent` | `main_kernel_gen.md` + `00_route_problem.md` | `analyze_problem.py` |
| `Fuser.orchestrator` + `Fuser.worker` | `01_fuse_model.md` | `run_candidate.py` |
| `Fuser.subgraph_extractor` | `02_extract_subgraphs.md` | `extract_json.py`, `dedup_subgraphs.py` |
| `Fuser.dispatch_kernel_agent` | `03_generate_kernel.md` | `build_reference.py`, `render_template.py` |
| `triton_kernel_agent.worker` | `04_refine_kernel.md` | `run_candidate.py` |
| `Fuser.compose_end_to_end` | `05_compose_kernels.md` | `run_candidate.py` |
| `triton_kernel_agent.agent` | `06_direct_kernel.md` | `render_template.py`, `run_candidate.py` |

## Requirements

- Python 3.8+
- The KernelAgent project installed (`pip install -e .` from project root)
- Jinja2 (`pip install jinja2`) for template rendering
- GPU access for kernel verification (CUDA or XPU)
