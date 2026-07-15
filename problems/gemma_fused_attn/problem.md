---
name: gemma_fused_attn
---

# gemma_fused_attn

Fused Gemma decoder layer (Pi0.5 VLM / prefix side, standard RMSNorm — NOT the
adaRMS action-expert side), parsed from the reference GemmaDecoderLayer built by
`ref_tests.build_reference_layer` (`transformers` gemma, `use_adarms=False`).

One decoder layer:

    r = hidden_states
    h = RMSNorm_in(hidden_states)          # fp32 var, scale by (1 + weight)
    h = SelfAttn(h) + r                    # GQA + rotary, full non-causal prefix
    r = h
    h = RMSNorm_post(h)
    h = MLP(h) + r                         # down(gelu_tanh(gate(h)) * up(h))

Self-attention is GQA (8 query heads, 1 kv head, head_dim 256), scaling
`head_dim**-0.5`, rotary embeddings (theta 1e4, positions = arange(seq_len)),
softmax accumulated in fp32, and an all-zero additive mask (full non-causal
prefix attention). No biases anywhere. bf16 fused kernels reorder fp ops, so
bit-exactness is not expected (~1e-2 relative for bf16).

## Axes

| axis | type | value | description |
|------|------|-------|-------------|
| batch_size | var | - | Batch size |
| seq_len | var | - | Prefix sequence length |
| hidden_size | const | 2048 | Hidden dimension (gemma_2b) |
| intermediate_size | const | 16384 | MLP intermediate dimension |
| num_attention_heads | const | 8 | Query heads |
| num_key_value_heads | const | 1 | KV heads (GQA) |
| head_dim | const | 256 | Per-head dimension |

## Inputs

| name | shape | dtype | role | description |
|------|-------|-------|------|-------------|
| hidden_states | [batch_size, seq_len, hidden_size] | bfloat16 | input | Layer input |
| input_layernorm_weight | [hidden_size] | bfloat16 | input | Pre-attention RMSNorm scale (applied as 1 + weight) |
| q_proj_weight | [2048, hidden_size] | bfloat16 | input | Query projection (num_attention_heads * head_dim, hidden) |
| k_proj_weight | [256, hidden_size] | bfloat16 | input | Key projection (num_key_value_heads * head_dim, hidden) |
| v_proj_weight | [256, hidden_size] | bfloat16 | input | Value projection (num_key_value_heads * head_dim, hidden) |
| o_proj_weight | [hidden_size, 2048] | bfloat16 | input | Output projection (hidden, num_attention_heads * head_dim) |
| post_attention_layernorm_weight | [hidden_size] | bfloat16 | input | Post-attention RMSNorm scale (applied as 1 + weight) |
| gate_proj_weight | [intermediate_size, hidden_size] | bfloat16 | input | MLP gate projection |
| up_proj_weight | [intermediate_size, hidden_size] | bfloat16 | input | MLP up projection |
| down_proj_weight | [hidden_size, intermediate_size] | bfloat16 | input | MLP down projection |
| eps | scalar | float32 | scalar | RMSNorm epsilon |

## Outputs

| name | shape | dtype | description |
|------|-------|-------|-------------|
| output | [batch_size, seq_len, hidden_size] | bfloat16 | Decoder layer output |

## Reference

```python
import torch
import torch.nn.functional as F

_ROPE_THETA = 10000.0


def _rms_norm(x, weight, eps):
    dtype = x.dtype
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normed = x * torch.rsqrt(var + eps)          # bf16 * fp32 -> fp32
    normed = normed * (1.0 + weight.float())
    return normed.to(dtype)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.no_grad()
def run(hidden_states, input_layernorm_weight,
        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
        post_attention_layernorm_weight,
        gate_proj_weight, up_proj_weight, down_proj_weight, eps):
    """Standard Gemma decoder layer (prefix side, use_adarms=False)."""
    bsz, seq_len, hidden_size = hidden_states.shape
    dtype = hidden_states.dtype
    device = hidden_states.device

    head_dim = 256
    num_attention_heads = q_proj_weight.shape[0] // head_dim   # 8
    num_key_value_heads = k_proj_weight.shape[0] // head_dim    # 1
    num_key_value_groups = num_attention_heads // num_key_value_heads
    scaling = head_dim ** -0.5

    # ---- input RMSNorm ----
    residual = hidden_states
    h = _rms_norm(hidden_states, input_layernorm_weight, eps)

    # ---- self attention (GQA + rotary, full prefix / zero mask) ----
    q = F.linear(h, q_proj_weight).view(
        bsz, seq_len, num_attention_heads, head_dim).transpose(1, 2)
    k = F.linear(h, k_proj_weight).view(
        bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    v = F.linear(h, v_proj_weight).view(
        bsz, seq_len, num_key_value_heads, head_dim).transpose(1, 2)

    half = head_dim // 2
    inv_freq = 1.0 / (_ROPE_THETA ** (torch.arange(0, half, device=device).float() * 2.0 / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin

    k = k.repeat_interleave(num_key_value_groups, dim=1)
    v = v.repeat_interleave(num_key_value_groups, dim=1)

    attn = torch.matmul(q, k.transpose(2, 3)) * scaling    # + 0 mask (full prefix)
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(dtype)
    o = torch.matmul(attn, v)
    o = o.transpose(1, 2).reshape(bsz, seq_len, -1)
    o = F.linear(o, o_proj_weight)
    h = residual + o

    # ---- post-attention RMSNorm + MLP ----
    residual = h
    h = _rms_norm(h, post_attention_layernorm_weight, eps)
    gate = F.gelu(F.linear(h, gate_proj_weight), approximate="tanh")
    up = F.linear(h, up_proj_weight)
    h = F.linear(gate * up, down_proj_weight)
    return residual + h
```

## Workloads

```jsonl
{"uuid": "b0e1f2a3-0001-4a00-9000-000000000001", "axes": {"batch_size": 2, "seq_len": 968}, "inputs": {"hidden_states": {"type": "random"}, "input_layernorm_weight": {"type": "random"}, "q_proj_weight": {"type": "random"}, "k_proj_weight": {"type": "random"}, "v_proj_weight": {"type": "random"}, "o_proj_weight": {"type": "random"}, "post_attention_layernorm_weight": {"type": "random"}, "gate_proj_weight": {"type": "random"}, "up_proj_weight": {"type": "random"}, "down_proj_weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.02, "max_rtol": 0.05}}
{"uuid": "b0e1f2a3-0002-4a00-9000-000000000002", "axes": {"batch_size": 1, "seq_len": 512}, "inputs": {"hidden_states": {"type": "random"}, "input_layernorm_weight": {"type": "random"}, "q_proj_weight": {"type": "random"}, "k_proj_weight": {"type": "random"}, "v_proj_weight": {"type": "random"}, "o_proj_weight": {"type": "random"}, "post_attention_layernorm_weight": {"type": "random"}, "gate_proj_weight": {"type": "random"}, "up_proj_weight": {"type": "random"}, "down_proj_weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.02, "max_rtol": 0.05}}
{"uuid": "b0e1f2a3-0003-4a00-9000-000000000003", "axes": {"batch_size": 4, "seq_len": 256}, "inputs": {"hidden_states": {"type": "random"}, "input_layernorm_weight": {"type": "random"}, "q_proj_weight": {"type": "random"}, "k_proj_weight": {"type": "random"}, "v_proj_weight": {"type": "random"}, "o_proj_weight": {"type": "random"}, "post_attention_layernorm_weight": {"type": "random"}, "gate_proj_weight": {"type": "random"}, "up_proj_weight": {"type": "random"}, "down_proj_weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.02, "max_rtol": 0.05}}
```
