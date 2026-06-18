# Template Sourcing Strategy

The local tool `render_template.py` reads Jinja2 templates directly from the
existing project directory:

    triton_kernel_agent/templates/

**No copies are maintained here.** This ensures templates stay in sync with the
upstream project and avoids duplication.

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
python skillset/tools/render_template.py \
  --template test_generation \
  --vars '{"problem_description": "...", "device_string": "cuda"}'

# Kernel generation — select language via kernel_language
python skillset/tools/render_template.py \
  --template kernel_generation \
  --vars '{"problem_description": "...", "test_code": "...", "kernel_language": "triton"}'

# Language guidelines only
python skillset/tools/render_template.py \
  --template language_guidelines \
  --vars '{"target_platform": "cuda", "kernel_language": "tilelang"}'
```
