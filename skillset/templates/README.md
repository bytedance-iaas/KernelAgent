# Template Sourcing Strategy

The local tool `render_template.py` reads Jinja2 templates directly from the
existing project directory:

    triton_kernel_agent/templates/

**No copies are maintained here.** This ensures templates stay in sync with the
upstream project and avoids duplication.

## Available Templates

### Shared (backend-agnostic)
- `test_generation.j2` — Generates test code for kernel verification
- `kernel_optimization.j2` — Generates optimized kernel based on profiling
- `reflexion_prompt.j2` — Generates reflexion/self-critique prompt for iterative refinement

### Per-backend (`backend/<backend>/`)

Templates are organized under `backend/triton/`, `backend/tilelang/`, and
`backend/cutedsl/`. The `kernel_backend` variable selects which set is used.

| Template | Purpose |
|---|---|
| `guidelines.j2` | Backend-specific programming guidelines (included by generation/refinement templates) |
| `kernel_generation.j2` | Generates initial kernel implementation for this backend |
| `kernel_refinement.j2` | Generates refined kernel based on error feedback for this backend |

Supported backends: **triton**, **tilelang**, **cutedsl**

## Usage

```bash
# Test code generation (backend-agnostic)
python skillset/tools/render_template.py \
  --template test_generation \
  --vars '{"problem_description": "...", "device_string": "cuda"}'

# Kernel generation — select backend via kernel_backend
python skillset/tools/render_template.py \
  --template kernel_generation \
  --vars '{"problem_description": "...", "test_code": "...", "kernel_backend": "triton"}'

# Backend guidelines only
python skillset/tools/render_template.py \
  --template backend_guidelines \
  --vars '{"target_platform": "cuda", "kernel_backend": "tilelang"}'
```
