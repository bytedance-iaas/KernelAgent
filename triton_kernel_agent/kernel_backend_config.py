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

"""Kernel backend registry for generated kernel source dialects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelBackendConfig:
    """Configuration for the language/framework used to write generated kernels."""

    name: str
    display_name: str
    generation_template: str
    refinement_template: str
    guidelines_template: str
    composition_label: str
    composition_requirements: str
    composition_tips: str


_TRITON_COMPOSITION_REQUIREMENTS = """\
- Provide at least one @triton.jit kernel and a top-level Python wrapper
  named kernel_function(...). This wrapper must accept the same primary
  input tensor(s) as the model and any required weights/biases with shapes
  implied by the problem; it should orchestrate Triton kernel(s) and
  return the final output tensor.
- No PyTorch math path: kernel_function MUST compute the final outputs
  using your Triton kernels only. Do NOT implement or fall back to
  torch.nn / torch.nn.functional / torch.* ops
  sigmoid, etc.) for producing the final result. Using PyTorch for
  reference comparisons is allowed only inside the self-test.
- No imports beyond torch, triton, triton.language as tl, and stdlib. No I/O."""

_TRITON_COMPOSITION_TIPS = """\
- Use tl.load/tl.store with masks for boundary conditions.
- Favor coalesced memory access; tile by blocks; compute grid from shape.
- Common Triton pitfalls to avoid:
  * Do NOT call tl.broadcast on Python scalars; tl.maximum(x, 0.0) works.
  * Prefer scalar constants directly in elementwise ops (no explicit broadcast needed).
  * Keep BLOCK_SIZE power-of-two; mask stores at tail."""

_CUTEDSL_COMPOSITION_REQUIREMENTS = """\
- Provide at least one cuteDSL/CuTe DSL kernel or kernel launcher and a top-level
  Python wrapper named kernel_function(...). This wrapper must accept the same
  primary input tensor(s) as the model and any required weights/biases with
  shapes implied by the problem; it should orchestrate cuteDSL kernel(s) and
  return the final output tensor.
- No PyTorch math path: kernel_function MUST compute the final outputs using
  cuteDSL kernels only. Do NOT implement or fall back to torch.nn /
  torch.nn.functional / torch.* compute ops for producing the final result.
  Using PyTorch for reference comparisons is allowed only inside the self-test.
- Imports may include torch, cuteDSL/CuTe DSL modules, and stdlib only. No I/O."""

_CUTEDSL_COMPOSITION_TIPS = """\
- Keep shape/layout metadata explicit in the wrapper and launch cuteDSL kernels
  for all numerical work.
- Prefer tiled memory layouts and coalesced global-memory access.
- Keep the Python wrapper limited to allocation, argument checks, and launch
  configuration; all elementwise, reduction, matmul, convolution, and pooling
  work must happen inside cuteDSL kernels."""

_TILELANG_COMPOSITION_REQUIREMENTS = """\
- Provide at least one TileLang JIT kernel builder and a top-level Python wrapper
  named kernel_function(...). The wrapper must accept the same primary input
  tensor(s) as the model and any required weights/biases with shapes implied by
  the problem; it should allocate outputs, compile/cache TileLang kernels, launch
  them, and return the final output tensor.
- No PyTorch math path: kernel_function MUST compute the final outputs using
  TileLang kernels only. Do NOT implement or fall back to torch.nn /
  torch.nn.functional / torch.* compute ops for producing the final result.
  Using PyTorch for reference comparisons is allowed only inside the self-test.
- Imports may include torch, tilelang, tilelang.language as T, functools, math,
  and stdlib only. No I/O."""

_TILELANG_COMPOSITION_TIPS = """\
- Use `@tilelang.jit` builders that return `@T.prim_func` kernels.
- Use `with T.Kernel(..., threads=...) as ...` to define launch grids and thread
  blocks.
- Use TileLang primitives such as `T.alloc_shared`, `T.alloc_fragment`,
  `T.copy`, `T.gemm`, `T.clear`, `T.Parallel`, and `T.Pipelined` where
  appropriate.
- Keep the Python wrapper limited to allocation, argument checks, kernel
  specialization/launch, and returning tensors; all numerical work must happen
  in TileLang kernels."""


KERNEL_BACKENDS: dict[str, KernelBackendConfig] = {
    "triton": KernelBackendConfig(
        name="triton",
        display_name="Triton",
        generation_template="kernel_generation.j2",
        refinement_template="kernel_refinement.j2",
        guidelines_template="triton_guidelines.j2",
        composition_label="Triton",
        composition_requirements=_TRITON_COMPOSITION_REQUIREMENTS,
        composition_tips=_TRITON_COMPOSITION_TIPS,
    ),
    "cutedsl": KernelBackendConfig(
        name="cutedsl",
        display_name="cuteDSL",
        generation_template="backend/cutedsl/kernel_generation.j2",
        refinement_template="backend/cutedsl/kernel_refinement.j2",
        guidelines_template="backend/cutedsl/guidelines.j2",
        composition_label="cuteDSL",
        composition_requirements=_CUTEDSL_COMPOSITION_REQUIREMENTS,
        composition_tips=_CUTEDSL_COMPOSITION_TIPS,
    ),
    "tilelang": KernelBackendConfig(
        name="tilelang",
        display_name="TileLang",
        generation_template="backend/tilelang/kernel_generation.j2",
        refinement_template="backend/tilelang/kernel_refinement.j2",
        guidelines_template="backend/tilelang/guidelines.j2",
        composition_label="TileLang",
        composition_requirements=_TILELANG_COMPOSITION_REQUIREMENTS,
        composition_tips=_TILELANG_COMPOSITION_TIPS,
    ),    
}


def get_kernel_backend(name: str) -> KernelBackendConfig:
    """Get kernel backend configuration by name."""
    key = name.strip().lower()
    if key not in KERNEL_BACKENDS:
        available = ", ".join(sorted(KERNEL_BACKENDS.keys()))
        raise ValueError(f"Unknown kernel backend '{name}'. Available: {available}")
    return KERNEL_BACKENDS[key]


def get_kernel_backend_choices() -> list[str]:
    """Get available kernel backend names for CLI choices."""
    return sorted(KERNEL_BACKENDS.keys())
