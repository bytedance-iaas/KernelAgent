# Unified Problem Format: `problem.md`

One markdown file per problem replaces both the SOL-ExecBench triplet
(`definition.json` + `reference.py` + `workload.jsonl`) and the KernelBench
triplet (`problem.py` + `test.py` + input builders). The schema mirrors
SOL-ExecBench (`examples/SOL-ExecBench`), with the reference kernel inline
as a PyTorch code block.

Tooling: `scripts/problem_md.py`

```bash
# convert a SOL-ExecBench problem directory
python scripts/problem_md.py from-solbench \
    examples/SOL-ExecBench/data/benchmark/L1/033_post_norm_residual \
    -o problems/post_norm_residual/problem.md

# validate
python scripts/problem_md.py check problems/post_norm_residual/problem.md

# emit KernelBench-compatible problem.py + test.py next to the md
python scripts/problem_md.py materialize problems/post_norm_residual/problem.md
```

Conversion is owned by the **ka-kernel-parser** skill (its "solbench
convert" mode, `steps/04_solbench_convert.md`); `ka-kernel-gen` and
`ka-kernel-opt` consume only the KernelBench output. `materialize`
produces that output: the generated `problem.py` satisfies the
KernelBench contract
(`Model`, `get_inputs()`, `get_init_inputs()`) plus `WORKLOADS` /
`build_workload_inputs(i)` / `workload_tolerance(i)`; the generated
`test.py` imports `from kernel import kernel_function` and checks every
workload with SOL-ExecBench's exact correctness semantics
(`core/bench/correctness.py`): an element matches when
`|out - ref| <= max_atol + max_rtol * |ref|`; the workload passes when
the matched ratio >= `required_matched_ratio` (default 0.99), there are
no nan/inf values, the output is not spuriously all-zero, and
`max_abs_err <= max_error_cap` when a cap is set. Exits 0 on PASS.

## Schema

````markdown
---
name: 033_post_norm_residual          # required, snake_case id
hf_id: allenai/Olmo-3-1025-7B         # optional provenance fields
---

# Title

Free-form description prose (becomes the problem description).

## Axes

| axis | type | value | description |
|------|------|-------|-------------|
| batch_size | var | - | Batch size |
| hidden_size | const | 4096 | Hidden dimension |

`var` axes get their values from each workload line; `const` axes are
fixed here.

## Inputs

| name | shape | dtype | role | description |
|------|-------|-------|------|-------------|
| sublayer_output | [batch_size, seq_len, hidden_size] | bfloat16 | input | ... |
| weight | [hidden_size] | bfloat16 | input | ... |
| eps | scalar | float32 | scalar | ... |

- `shape`: `[dims]` mixing axis names and integer literals, or `scalar`.
- `role`: `input` (runtime tensor, default), `weight` (reserved for a
  future init-input split; currently passed as a runtime arg), `scalar`
  (python scalar, value supplied by the workload line).
- Argument order in this table IS the `run(...)` argument order.

## Outputs

| name | shape | dtype | description |
|------|-------|-------|-------------|
| output | [batch_size, seq_len, hidden_size] | bfloat16 | ... |

## Reference

Exactly one fenced python block defining `run(...)` in pure PyTorch —
the ground truth the kernel is verified against:

```python
import torch

@torch.no_grad()
def run(sublayer_output, residual, weight, eps):
    ...
    return output
```

## Workloads

Exactly one fenced jsonl block, one workload per line, SOL-ExecBench
schema. The FIRST line is the canonical workload (used by
`get_inputs()` and benchmarks); the rest are additional accuracy cases.

```jsonl
{"uuid": "...", "axes": {"batch_size": 16, "seq_len": 1024}, "inputs": {"sublayer_output": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0063, "max_rtol": 0.05}, "latency": {"h200": {"baseline": 0.98, "target": 0.12}}}
```
````

### Performance targets (`latency`)

A workload line may carry an optional `latency` object: per-GPU
`{"<gpu_key>": {"baseline": <ms>, "target": <ms>}}` entries (units:
milliseconds). `gpu_key` is a case-insensitive substring of
`torch.cuda.get_device_name()` — e.g. `"h200"`, `"b200"`, `"h100"`.
`baseline` is what an unoptimized reference achieves; `target` is the
optimization goal.

When workloads carry `latency`, `materialize` additionally emits one
pinned gate per GPU spec in a subfolder: `<problem>/<gpu_key>/
perf_test.py` (e.g. `fp8_group_gemm/b200/perf_test.py`). Each gate
imports `problem`/`kernel` from the parent problem directory,
benchmarks `kernel.kernel_function` (median of CUDA-event timings) on
every workload with a target for that spec, prints a per-workload table
plus one machine-readable JSON line, and exits 0 iff every target is
met — or 2 when run on a different GPU (pinned, never vacuous).
`problem.py` exposes the same data via `gpu_key()` and
`workload_latency(i)`.

The perf gates are GOAL gates, not correctness gates: the
`ka-kernel-opt` pipeline picks the subfolder matching the current GPU,
runs it on each accepted best kernel, and stops with success when it
passes; `test.py` alone decides correctness.

## Input generation semantics

Mirrors SOL-ExecBench `core/bench/io.py`:

- `{"type": "random"}` — `torch.randn` for fp32/fp16/bf16/fp64; fp8 is
  randn(fp32).clamp(±2).to(fp8); bool is randint(0,2); ints use bounded
  randint ranges. Seeded with `torch.manual_seed(workload_idx)` for
  reproducibility.
- `{"type": "scalar", "value": v}` — passes `v` as a python scalar.
- `{"type": "zeros"}` — zero tensor.
- `{"type": "custom"}` — the value comes from the problem's
  `custom_inputs_entrypoint` (front-matter key): a function inside the
  Reference code block called as `fn(axes_and_scalars, device) -> dict`,
  exactly like SOL-ExecBench's `gen_inputs`.
- Name-based heuristics (ported from SOL-ExecBench) override plain randn
  for `random` inputs: norm weights -> ones, norm biases -> zeros, causal
  attention masks, binary masks, rope cos/sin, positive tensors
  (rstd/std/var), SSM decay params (A/A_log/A_cumsum/g), softmax outputs,
  and fan-in-scaled weight matrices.

Axes may be `var` (from the workload line), `const` (fixed value), or
`expr` (arithmetic over other axes in the value column, e.g.
`batch_size * seq_len`, resolved in declaration order). Multi-output
references (run() returning a tuple) are checked output-by-output.

Not yet supported: workload input type `safetensors` (needs external
weight files) and random generation for packed `float4_e2m1fn_x2`
inputs — keep such construction inside `run(...)` helpers or the
custom entrypoint.
