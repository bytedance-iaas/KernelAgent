# Step: Route Problem

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-gen skill directory; the
> shared tool scripts live at `${CLAUDE_SKILL_DIR}/tools/`.

## Purpose
Analyze a KernelBench-style problem file and decide the optimal routing strategy.
Replaces `AutoKernelRouter._llm_decide_route()` from `Fuser/auto_agent.py`.

## When to Use
As the first step of the `ka-kernel-gen` generate mode, before choosing between
the direct KernelAgent path and the full Fuser pipeline — or standalone in
analyze mode.

## Inputs
- `PROBLEM_PATH`: Absolute path to the problem `.py` file

## Workflow

### Step 1: Run Static Analysis

```bash
python "${CLAUDE_SKILL_DIR}/tools/analyze_problem.py" --problem $PROBLEM_PATH
```

This produces JSON with complexity features:
- `has_control_flow`, `has_attention_like`, `has_conv_transpose`, `has_group_norm`
- `has_conv`, `pool_ops`, `act_ops`, `chain_len_estimate`
- `raw_op_names` (top operation counts)
- `route_recommendation` (heuristic-based)

### Step 2: Read and Analyze the Problem Code

Read the problem file and consider:

**Architectural patterns:**
- ResNet blocks (conv → BN → ReLU): if short chain → `kernelagent`
- UNet/decoder (conv_transpose2d): → `fuser`
- Transformer blocks (QKV, softmax, matmul): self-contained SDPA → `kernelagent`; with projections/norms → `fuser`
- MLPs (linear → activation): → `kernelagent`
- Depthwise-separable convs: depends on chain length
- GroupNorm + activation chains: → `fuser` if with conv/pool

**KernelBench level hints:**
- Level 1: strongly bias to `kernelagent` unless complexity triggers hit
- Level 2/3: more likely `fuser`, especially with multi-op compositions

### Step 3: Choose Strategy

Select one of four strategies:

| Strategy | When to Use |
|---|---|
| `kernelagent` | Simple/linear op chains, no triggers |
| `fuser` | Complex patterns present |
| `kernel_then_fuser` | Likely trivial but uncertain |
| `fuser_then_kernel` | Likely complex but worth trying direct |

### Step 4: Output Decision

Output a JSON object:
```json
{
  "route_strategy": "kernelagent|fuser|kernel_then_fuser|fuser_then_kernel",
  "confidence": 0.0-1.0,
  "rationale": "explanation of why this route was chosen",
  "static_features": { ... features from analyze_problem.py ... }
}
```

## Decision Criteria Reference

From the original auto_agent.py routing policy:
- **Route to Fuser if ANY of:**
  - Attention-like patterns (except self-contained SDPA with chain ≤3)
  - `conv_transpose2d` present
  - `group_norm` with conv/pool or long chains (≥4 steps)
  - Explicit control flow in forward
  - Chain length ≥4
- **Otherwise route to KernelAgent**
