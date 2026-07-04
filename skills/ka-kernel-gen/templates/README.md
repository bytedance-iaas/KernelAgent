# Templates

Vendored copies of the Jinja2 prompt templates from
`triton_kernel_agent/templates/`, so the ka-kernel-gen skill works standalone
(outside the KernelAgent repo, without the `triton_kernel_agent` package).

When working inside the KernelAgent repo, re-sync after upstream template
changes:

```bash
./skills/ka-kernel-gen/sync_templates.sh
```

`render_template.py` resolves the templates directory in this order:
`--templates-dir` flag → `KERNELAGENT_TEMPLATES_DIR` env var → this directory.

## Available Templates

### Shared (language-agnostic)
- `test_generation.j2` — Generates test code for kernel verification
- `kernel_optimization.j2` — Generates optimized kernel based on profiling
- `reflexion_prompt.j2` — Generates reflexion/self-critique prompt for iterative refinement

### Per-language (`backend/<language>/`)

Templates are organized under `backend/triton/`, `backend/tilelang/`, and
`backend/cutedsl/`. The `kernel_language` variable selects which set is used.

| Template | Purpose |
|---|---|
| `guidelines.j2` | Language-specific programming guidelines (included by generation/refinement templates) |
| `kernel_generation.j2` | Generates initial kernel implementation for this language |
| `kernel_refinement.j2` | Generates refined kernel based on error feedback for this language |

Supported languages: **triton**, **tilelang**, **cutedsl**

## Usage

```bash
# Test code generation (language-agnostic)
python skills/ka-kernel-gen/tools/render_template.py \
  --template test_generation \
  --vars '{"problem_description": "...", "target_platform": "cuda"}'

# Kernel generation — select language via kernel_language
python skills/ka-kernel-gen/tools/render_template.py \
  --template kernel_generation \
  --vars '{"problem_description": "...", "test_code": "...", "kernel_language": "triton"}'

# Language guidelines only
python skills/ka-kernel-gen/tools/render_template.py \
  --template language_guidelines \
  --vars '{"kernel_language": "tilelang"}'
```
