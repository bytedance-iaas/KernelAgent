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

"""
Platform configuration registry for multi-backend support.

Usage:
    from triton_kernel_agent.platform_config import get_platform, get_platform_choices

    platform = get_platform("xpu")
    print(platform.device_string)  # "xpu"
    print(platform.guidance_block)  # Intel XPU-specific guidance
"""

from dataclasses import dataclass, field

DEFAULT_PLATFORM = "cuda"


@dataclass(frozen=True)
class PlatformConfig:
    """Configuration for a specific hardware platform/backend."""

    name: str
    device_string: str
    guidance_block: str
    kernel_guidance: str
    cuda_hacks_to_strip: tuple = field(default_factory=tuple)


# Platform-specific constants
_XPU_GUIDANCE = """\
**CRITICAL PLATFORM REQUIREMENTS FOR INTEL XPU:**
- Default tensor allocations to device='xpu' (never 'cuda'); CPU is allowed only when necessary.
- Check availability with: hasattr(torch, 'xpu') and torch.xpu.is_available()
- Do NOT monkey-patch torch.cuda or torch.device
- Do NOT set TRITON_BACKENDS environment variable
- Do NOT import or disable XPUDriver
- Use torch.xpu.synchronize() if synchronization is needed
- Intel XPU subgroup size is typically 16 (not 32 like CUDA warps)
- Preferred block sizes: 64, 128, 256, or 512"""

_XPU_KERNEL_GUIDANCE = """\
## Intel XPU-Specific Optimizations

You are generating a Triton kernel for Intel XPU (Xe GPUs). Follow these guidelines:

1. **Device Context**: Use 'xpu' as the device instead of 'cuda'
2. **Memory Hierarchy**: Intel Xe has different cache sizes - optimize accordingly
3. **Thread Configuration**:
   - Subgroup size is typically 8, 16, or 32 (flexible)
   - num_warps: typically 4, 8, or 16 for Intel GPUs
   - BLOCK_SIZE: prefer 64, 128, 256, or 512
4. **Optimal Block Sizes**: Start with 128-256 for most kernels
5. **Data Types**: Intel supports fp32, fp16, bf16 (fp8 varies by generation)"""

_XPU_CUDA_HACKS = (
    "torch.cuda.is_available = lambda: True",
    "_orig_torch_device = torch.device",
    "_real_torch_device = torch.device",
    "def _fake_torch_device",
    "torch.device = _fake_torch_device",
    'os.environ["TRITON_BACKENDS"] = "cuda"',
    "from triton.backends.intel.driver import XPUDriver",
    "XPUDriver.is_available = classmethod(lambda cls: False)",
)

_CUTEDSL_GUIDANCE = """\
**CRITICAL REQUIREMENTS FOR CuteDSL KERNELS:**
- Import: `import cutlass; import cutlass.cute as cute`
- TWO-FUNCTION PATTERN (mandatory):
    1. `@cute.kernel` — GPU-side function, args typed as `cute.Tensor`
    2. `@cute.jit`    — host-side launcher, calls `kernel(...).launch(grid=..., block=...)`
  `cute.compile()` MUST target the `@cute.jit` function, NOT the bare `@cute.kernel`.
- Convert PyTorch tensors: `from_dlpack(tensor, assumed_align=16)`
  (call `.contiguous()` first if needed; do this in the Python wrapper, not the kernel)
- Inside `@cute.kernel`, index tensors directly: `gA[i]`, `gA[row, col]`
- Compile-time constants inside the kernel: `BLK: cutlass.Constexpr = 256`
- Shared memory: `cute.arch.alloc_smem(Float32, N)` where N is a plain Python int
  (module-level constant). Wrap result with `cute.make_tensor(ptr, cute.make_layout(N))`
  before indexing with runtime values.
- Type casts: `cutlass.BFloat16(f32_val)`, `Float32(bf16_val)`, `cutlass.Int32(x)`
- Thread indexing: `tid_x, _, _ = cute.arch.thread_idx()` (returns 3-tuple)
- Warp reduction: `acc = cute.arch.warp_reduction_sum(acc)` for FP32 sum
- Compile-time loops: `for i in cutlass.range_constexpr(N):` (N must be Python int)
- Expose `kernel_function(*args) -> torch.Tensor` that caches and calls the compiled host fn
- Do NOT import or call any Triton APIs (`triton`, `triton.language`, `tl.*`)"""

_CUTEDSL_KERNEL_GUIDANCE = """\
## CuteDSL Optimization Guidelines

You are generating a CuteDSL kernel. Follow these patterns exactly.

1. **Required imports**
   ```python
   import torch
   import cutlass
   import cutlass.cute as cute
   from cutlass import Float32, BFloat16, Int32
   from cutlass.cute.runtime import from_dlpack
   ```

2. **Two-function structure** (MANDATORY — cute.compile targets @cute.jit, not @cute.kernel)
   ```python
   @cute.kernel
   def _kernel(gA: cute.Tensor, gOut: cute.Tensor):
       BLK: cutlass.Constexpr = _BLK   # compile-time constant from module level
       tid_x, _, _ = cute.arch.thread_idx()
       bid_x, _, _ = cute.arch.block_idx()
       ...

   @cute.jit
   def _host(gA: cute.Tensor, gOut: cute.Tensor):
       M = gA.shape[0]
       _kernel(gA, gOut).launch(grid=(M, 1, 1), block=(_BLK, 1, 1))

   _COMPILE_CACHE: dict = {}
   def kernel_function(A: torch.Tensor) -> torch.Tensor:
       A   = A.contiguous()
       Out = torch.empty_like(A)
       mA, mOut = from_dlpack(A, assumed_align=16), from_dlpack(Out, assumed_align=16)
       key = (A.shape, A.dtype)
       if key not in _COMPILE_CACHE:
           _COMPILE_CACHE[key] = cute.compile(_host, mA, mOut)
       _COMPILE_CACHE[key](mA, mOut)
       return Out
   ```

3. **Shared memory** — size argument MUST be a plain Python int
   ```python
   _WARPS = 8   # module-level Python int, not cutlass.Constexpr

   smem_ptr = cute.arch.alloc_smem(Float32, _WARPS)          # plain int required
   smem     = cute.make_tensor(smem_ptr, cute.make_layout(_WARPS))  # wrap for runtime indexing
   smem[warp_id] = acc      # warp_id is runtime — OK via cute.Tensor
   cute.arch.sync_threads()
   ```

4. **Warp reduction and type casting**
   ```python
   acc = cute.arch.warp_reduction_sum(acc)   # FP32 butterfly reduction
   gOut[row] = cutlass.BFloat16(acc)         # FP32 → BF16 cast
   ```

5. **Compile-time loops** — argument must be a plain Python int
   ```python
   for i in cutlass.range_constexpr(_WARPS):   # compiler-unrolled
       total = total + smem[i]
   ```

6. **Tile sizes and thread counts**: powers of 2 (64, 128, 256)
7. **Prefer persistent kernels** for memory-bandwidth-bound ops
8. **Use TMEM** (SM100) for MMA accumulators; avoid stores until epilogue"""


# Platform registry
PLATFORMS: dict[str, PlatformConfig] = {
    "cuda": PlatformConfig(
        name="cuda",
        device_string="cuda",
        guidance_block="",
        kernel_guidance="",
        cuda_hacks_to_strip=(),
    ),
    "cutedsl": PlatformConfig(
        name="cutedsl",
        device_string="cuda",
        guidance_block=_CUTEDSL_GUIDANCE,
        kernel_guidance=_CUTEDSL_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=(),
    ),
    "xpu": PlatformConfig(
        name="xpu",
        device_string="xpu",
        guidance_block=_XPU_GUIDANCE,
        kernel_guidance=_XPU_KERNEL_GUIDANCE,
        cuda_hacks_to_strip=_XPU_CUDA_HACKS,
    ),
}


def get_platform(name: str) -> PlatformConfig:
    """Get platform configuration by name."""
    if name not in PLATFORMS:
        available = ", ".join(sorted(PLATFORMS.keys()))
        raise ValueError(f"Unknown platform '{name}'. Available: {available}")
    return PLATFORMS[name]


def get_platform_choices() -> list[str]:
    """Get list of available platform names for CLI choices."""
    return sorted(PLATFORMS.keys())
