# Step: Generate the KernelBench Problem File

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-parser skill directory.

## Purpose
Write the problem file in the exact KernelBench format (see
`examples/KernelBench/KernelBench/level{1..4}/` in the KernelAgent repo for
canonical examples).

## Format Contract

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    One-sentence description of the operation.
    """
    def __init__(self, <hyperparams...>):
        super(Model, self).__init__()
        # nn.* layers and nn.Parameter weights only

    def forward(self, <runtime inputs...>) -> torch.Tensor:
        """
        Args / Returns docstring with shapes.
        """
        # PURE PyTorch ops — this is the reference implementation
        return ...

# module-level workload constants
batch_size = ...
M = ...

def get_inputs():
    return [torch.rand(...), ...]     # forward args, in order

def get_init_inputs():
    return [<constructor args, in order>]  # [] if none
```

Rules:
1. **Pure PyTorch only** in the file — no `triton`/`cutlass`/`tilelang`
   imports, no custom CUDA. The Model IS the reference implementation the
   original kernel will be compared against.
2. `forward` takes the runtime tensors; weights/bias live in `__init__` as
   `nn.Parameter` / `nn.*` layers; hyperparameters flow through
   `get_init_inputs()`.
3. Workload shapes are module-level constants named like the examples
   (`batch_size`, `M`, `K`, `N`, `in_channels`, ...), matching the analysis
   from step 01.
4. `get_inputs()` creates CPU tensors with `torch.rand`/`torch.randn`
   (`torch.randint` for integer inputs) — no device placement, no dtype
   casts (the harness handles device/dtype). If the source kernel only
   makes sense in a reduced dtype, add a comment noting it.
5. Keep the file self-contained and deterministic apart from the random
   inputs — no I/O, no globals beyond the shape constants.
6. Docstrings: class docstring describes the op; forward docstring lists
   Args/Returns with shapes (match the level-1/2 example style).

## Naming and Level

- Default filename: `<Description_In_PascalCase>.py`, or
  `<N>_<Description>.py` if the user wants KernelBench numbering; write to
  `OUTPUT_PATH` if given.
- Level classification (report it, and use it if filing into a level dir):
  - **1**: single operator (matmul, conv, softmax, reduction, loss)
  - **2**: short fused sequence (conv + activation + bias, gemm + epilogue)
  - **3**: full architecture (MLP, ResNet block/network, attention block)
  - **4**: pretrained HuggingFace model wrapper

## Fidelity Notes
- Model the *logical* op, not the kernel's schedule: tile sizes, swizzling,
  pipelining, TMA usage do not appear in the problem file.
- Fused kernels: forward applies the ops in the kernel's fusion order.
- Non-standard numerics (FP8 scaling, stochastic rounding): express the
  dequantized equivalent and document the difference in the docstring.
