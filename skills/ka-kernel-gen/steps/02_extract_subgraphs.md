# Step: Extract Subgraphs from Fused Code

> **Paths:** `${CLAUDE_SKILL_DIR}` is the ka-kernel-gen skill directory; the
> shared tool scripts live at `${CLAUDE_SKILL_DIR}/tools/`.

## Purpose
Analyze fused PyTorch code and emit a JSON array of unique subgraphs with exact
shape signatures. Replaces the LLM call in `Fuser.subgraph_extractor.extract_subgraphs_to_json()`.

## When to Use
After step `01_fuse_model.md` has produced verified fused code, to decompose it into
individual subgraphs that can be independently converted to Triton kernels.

## Inputs
- `FUSED_CODE`: The fused PyTorch code (text content or file path)
- `PROBLEM_PATH`: Path to the original problem file
- `RUN_DIR`: Path to the run directory for saving artifacts

## Workflow

### Step 1: Read Both Files

Read the fused code and the original problem file.

### Step 2: Analyze and Emit Subgraphs JSON

Analyze the fused code and original problem to identify every unique subgraph.
Emit a JSON array where each item follows this schema:

```json
{
  "id": "<string>",
  "type": "<string, e.g. conv_relu, pool, attention>",
  "data_layout": "<NCHW|NHWC|null>",
  "dtype": "<string|null>",
  "ops": [{"op": "<string>", ... op-specific fields ...}],
  "input_shape": [<int|sym>, ...],
  "output_shape": [<int|sym>, ...],
  "weights_fused": {"<name>": [<int|sym>, ...], ...} | null,
  "weights_original": {"<name>": [<int|sym>, ...], ...} | null,
  "count": <int>,
  "where": "<string, e.g. Model.forward stem>",
  "source": {"module": "<string>", "code": "<string>"}
}
```

For multi-input ops (like residual add), use `"inputs": [[...], [...]]` instead of `"input_shape"`.

**Key rules:**
- Treat any shape difference (inputs/outputs/weights) as a distinct subgraph. Count occurrences.
- Populate op-specific fields: `kernel_size`, `stride`, `padding`, `groups`, `bn_fused`, `output_size`, `start_dim`.
- Include both `weights_original` (pre-fusion params like BN gamma/beta/running stats) and `weights_fused` (post-fusion).
- Provide a short `"where"` string (e.g., `Model.forward stem` or `layer2.block3.conv`).
- Provide `"source"` with the smallest contiguous code snippet implementing the subgraph.
- Prefer concrete integers from `get_inputs()` shapes in the problem.

### Step 3: Deduplicate

Save the raw JSON to a temp file and run:
```bash
python "${CLAUDE_SKILL_DIR}/tools/dedup_subgraphs.py" --input $RAW_JSON_PATH --output $RUN_DIR/subgraphs.json
```

### Step 4: Save Artifacts

The deduplicated subgraphs are saved to `$RUN_DIR/subgraphs.json`.

## Output
- Path to `subgraphs.json`
- Number of unique subgraphs found
