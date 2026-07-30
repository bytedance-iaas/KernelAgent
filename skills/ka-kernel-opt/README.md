# ka-kernel-opt — Claude Code Skill for KernelAgent Kernel Optimization

A standalone, self-contained Claude Code skill that replicates the KernelAgent
hardware-guided kernel optimization pipeline (`kernel_perf_agent` +
`triton_kernel_agent` opt manager/worker/orchestrator) without external LLM
API calls or the KernelAgent Python package. Claude acts as the diagnosis and
rewrite brain; small deterministic scripts handle NCU profiling, roofline
math, and benchmarking.

Copy or symlink this one folder anywhere and the core loop works as-is; for
the richest results also link the companion skills and kernel repos described
below.

## Architecture

```
ka-kernel-opt/
├── SKILL.md                    # Entry point: mode dispatch + optimization loop
├── steps/
│   ├── 01_baseline.md          # Verify kernel + benchmark eager/compile/initial
│   ├── 02_profile.md           # NCU profiling + roofline/grid analysis
│   ├── 03_diagnose.md          # Bottleneck diagnosis (Claude as GPU perf expert)
│   ├── 04_rewrite.md           # Optimization rewrite w/ pattern library
│   └── 05_verify_accept.md     # Correctness, benchmark, accept/reject, reflexion
├── tools/                      # Deterministic scripts (no LLM, no repo deps)
│   ├── profile_ncu.py          # Run Nsight Compute, parse CSV → metrics JSON
│   ├── roofline.py             # SOL classification + grid analysis + config extraction
│   ├── benchmark.py            # Time kernel / PyTorch eager / torch.compile
│   ├── gpu_specs.py            # GPU spec database lookup / auto-detect
│   ├── program_db.py           # Program database with lineage (best / top-k)
│   ├── run_candidate.py        # Execute test file, classify PASS/FAIL
│   └── kernel_io.py            # Shared problem/kernel loading helpers
└── reference/                  # Real production kernels, per platform
    └── cuda -> ../../../reference/cuda   # flash-attention, flashinfer, sgl-DeepGEMM
```

The original pipeline's three LLM calls (bottleneck diagnosis, kernel rewrite,
reflexion) become skill steps Claude performs directly. The OpenAI-embedding
RAG retrieval is replaced by three richer knowledge sources (see below).

## Companion Knowledge Sources

Instead of a vendored toy corpus, the skill draws on sibling skills and real
kernel repos (all optional — each has a graceful fallback):

| Source | Location | Used for |
|---|---|---|
| **KernelWiki** skill | `.claude/skills/KernelWiki/` | Hopper/Blackwell techniques (TMA, tcgen05, warp specialization, grouped GEMM) with CUTLASS/SGLang/vLLM/FlashInfer PR references — rewrite step |
| **cuda_skill** | `.claude/skills/cuda_skill/` | General CUDA optimization/debugging/profiling practice — rewrite step |
| **ncu-report-skill** | `.claude/skills/ncu-report-skill/` | Deep NCU analysis: full-set reports, `ncu_report` Python API, per-line stalls, B200/sm_100 metric names — profile step escalation |
| **Kernel repos** | `reference/cuda/` (symlink to the repo's platform-specific kernel submodules) | Battle-tested implementations to grep/read before applying a pattern — rewrite step |
| **Learned insights** | `reference/insights/` (symlink to the repo's `insights/` folder) | Checklists distilled from past optimization campaigns — mandatory read before diagnosing or rewriting without profile evidence |

Link the skills into your project's `.claude/skills/` and initialize the
submodules (`git submodule update --init`) to enable them.

## Input Contract

A kernel directory with three files (same as `examples/run_opt_manager.py`):

```
kernel_dir/
├── input.py     # initial kernel — defines kernel_function(...)
├── problem.py   # defines Model(nn.Module), get_inputs(), [get_init_inputs()]
└── test.py      # correctness test: imports `from kernel import kernel_function`,
                 # exits 0 on PASS
```

## Installation

```bash
mkdir -p .claude/skills
ln -s /path/to/skills/ka-kernel-opt .claude/skills/ka-kernel-opt
```

In the KernelAgent repo this is already set up.

## Usage

```
/ka-kernel-opt examples/optimize_01_matvec                 # full optimization loop
/ka-kernel-opt profile examples/optimize_01_matvec        # NCU + roofline report only
/ka-kernel-opt diagnose examples/optimize_01_matvec       # + bottleneck diagnosis, no rewrite
```

Options via natural language: GPU name, max rounds, warmup/repeat, greedy vs
beam strategy, bottleneck override, kernel language (triton/tilelang/cutedsl).

Tools can also run standalone:

```bash
python skills/ka-kernel-opt/tools/gpu_specs.py --detect
python skills/ka-kernel-opt/tools/benchmark.py --mode eager --problem problem.py
python skills/ka-kernel-opt/tools/profile_ncu.py --kernel kernel.py --problem problem.py --workdir ./artifacts
python skills/ka-kernel-opt/tools/roofline.py --metrics artifacts/ncu_metrics.json --gpu-name "NVIDIA H100 NVL 94GB"
```

## Mapping to Original Pipeline

| Original Component | Skill Part |
|---|---|
| `OptimizationManager.run_optimization` (baselines, round loop, round tables, result contract) | `SKILL.md` + `steps/01_baseline.md` |
| `KernelProfiler` + `kernel_perf_agent...ncu_profiler` (incl. PyTorch-baseline NCU) | `tools/profile_ncu.py` (`--target kernel\|eager`) |
| `RooflineAnalyzer` + `_compute_grid_analysis` + `extract_kernel_config` | `tools/roofline.py` |
| `Benchmark` / `timing.py` / `kernel_subprocess.py` | `tools/benchmark.py` |
| `gpu_specs_database` / `get_gpu_specs` | `tools/gpu_specs.py` |
| `JSONProgramDatabase` / `ProgramEntry` (lineage, top-k) | `tools/program_db.py` |
| `BottleneckAnalyzer` + `BOTTLENECK_PROMPT` (judger_prompt) | `steps/03_diagnose.md` (Claude) |
| `kernel_optimization.j2` rewrite call | `steps/04_rewrite.md` (Claude) |
| `verify_with_refinement` + `_update_kernels` accept/reject | `steps/05_verify_accept.md` |
| `reflexion_prompt.j2` | `steps/05_verify_accept.md` (Claude) |
| `RAGPrescriber` + `kernel_opt/database` corpus | KernelWiki + cuda_skill + `reference/cuda/` repos |
| `GreedyStrategy` / `BeamSearchStrategy` (select/update/terminate) | `SKILL.md` round step 1 + beam variant |

## Requirements

- Python 3.9+; tools are stdlib-only in-process
- GPU machine with `torch` + the kernel's DSL (`triton`/`tilelang`/`cutlass`)
- NVIDIA Nsight Compute (`ncu`) for profiling (with counter permissions or sudo)
- GPU specs database covers A100/H100/H200/RTX 4090/RTX 5080 (extend
  `tools/gpu_specs.py` for others)
- Companion skills + kernel-repo submodules for the richest results (see
  Companion Knowledge Sources above); the core loop works without them
