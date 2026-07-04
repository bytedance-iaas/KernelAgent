# ka-kernel-parser — Claude Code Skill: Kernel → KernelBench Problem

Parses an existing GPU kernel — **Triton**, **PyTorch**, or **CUTLASS/CUDA** —
into the KernelBench problem format: a pure-PyTorch `Model(nn.Module)`
reference with `get_init_inputs()` / `get_inputs()`. The Model is not meant
to be fast — it is meant to be *exactly as accurate as the source kernel*:
besides the format contract and a random-input equivalence check, the
kernel's **own unit tests are ported** to run against the Model with
identical inputs, seeds, and tolerances.

Together with its siblings this completes the KernelAgent skill pipeline:

```
existing kernel ──ka-kernel-parser──▶ problem.py ──ka-kernel-gen──▶ verified kernel ──ka-kernel-opt──▶ fast kernel
```

## Architecture

```
ka-kernel-parser/
├── SKILL.md                    # Entry point: analyze → generate → validate
├── steps/
│   ├── 01_analyze_kernel.md    # Extract semantics/shapes/unit tests per source type
│   ├── 02_generate_problem.md  # KernelBench format contract + conventions
│   └── 03_validate.md          # Contract check + equivalence + ported unit tests
└── tools/
    ├── validate_problem.py     # Format contract + one forward pass → specs JSON
    ├── check_equivalence.py    # Model vs kernel on same seeded inputs → allclose
    └── run_candidate.py        # Generic runner for ad-hoc equivalence harnesses
```

## Output Format (KernelBench)

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, <hyperparams>): ...   # weights/layers live here
    def forward(self, <tensors>): ...        # PURE PyTorch reference math

batch_size = ...                             # module-level workload constants

def get_inputs():      return [torch.rand(...)]   # forward args (CPU)
def get_init_inputs(): return [<hyperparams>]     # constructor args
```

Levels: 1 = single op, 2 = fused op sequence, 3 = full architecture,
4 = pretrained HF model. Canonical examples:
`examples/KernelBench/KernelBench/level{1..4}/`.

## Installation

```bash
mkdir -p .claude/skills
ln -s /path/to/skills/ka-kernel-parser .claude/skills/ka-kernel-parser
```

## Usage

```
/ka-kernel-parser path/to/triton_kernel.py
/ka-kernel-parser path/to/gemm.cu --output problems/my_gemm.py
/ka-kernel-parser parse every kernel under kernels/ into problems/
```

Tools standalone:

```bash
python skills/ka-kernel-parser/tools/validate_problem.py --problem problem.py
python skills/ka-kernel-parser/tools/check_equivalence.py \
  --problem problem.py --kernel kernel.py --entry kernel_function
```

## Notes

- **Unit-test porting**: every case from the kernel's existing tests is
  duplicated against the Model — same input construction, seeds,
  parametrizations, and the original tolerances (exact equality stays
  exact). When the kernel is Python-callable, each case also asserts
  `Model ≈ kernel` directly on the test's own inputs. The ported file
  (`test_<name>.py`) is standalone (prints `PASS`, exits 0) and doubles as
  the `test.py` required by `ka-kernel-opt`.
- CUTLASS/C++ sources without Python bindings get format validation only;
  the equivalence check is skipped with an explicit note (or done via a
  hand-written harness when bindings exist).
- FP8/quantized kernels are referenced via their dequantized math with
  loosened, documented tolerances.
- Prefers codegraph (`codegraph_explore` MCP / `codegraph explore` CLI) over
  grep when hunting shapes/callers in large repos with a `.codegraph/` index.

## Requirements

- Python 3.9+, `torch` for the validation tools
- The source kernel's DSL importable (and a GPU for GPU-only DSLs) when
  running the equivalence check
