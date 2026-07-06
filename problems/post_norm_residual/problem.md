---
name: 033_post_norm_residual
hf_id: allenai/Olmo-3-1025-7B
---

# 033_post_norm_residual

Post-normalization residual connection pattern: output = residual + RMSNorm(sublayer_output). RMSNorm applies learned weight scaling after normalization. Used after both attention and MLP blocks in Olmo3 architecture.

## Axes

| axis | type | value | description |
|------|------|-------|-------------|
| batch_size | var | - | Batch size |
| seq_len | var | - | Sequence length |
| hidden_size | const | 4096 | Hidden dimension size |

## Inputs

| name | shape | dtype | role | description |
|------|-------|-------|------|-------------|
| sublayer_output | [batch_size, seq_len, hidden_size] | bfloat16 | input | Output from attention or MLP sublayer |
| residual | [batch_size, seq_len, hidden_size] | bfloat16 | input | Residual connection input |
| weight | [hidden_size] | bfloat16 | input | Learned scale parameter for RMSNorm |
| eps | scalar | float32 | scalar | Epsilon for numerical stability in RMSNorm |

## Outputs

| name | shape | dtype | description |
|------|-------|-------|-------------|
| output | [batch_size, seq_len, hidden_size] | bfloat16 | Output with residual added after normalization |

## Reference

```python
import torch

@torch.no_grad()
def run(sublayer_output: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Post-normalization residual connection: output = residual + RMSNorm(sublayer_output)
    
    RMSNorm computation:
    1. Compute variance: mean of squared values along hidden dimension
    2. Normalize: x * rsqrt(variance + eps)
    3. Apply learned scale (weight parameter)
    4. Add residual connection
    
    Args:
        sublayer_output: Output from attention or MLP sublayer [batch, seq_len, hidden_size]
        residual: Residual connection input [batch, seq_len, hidden_size]
        weight: Learned scale parameter [hidden_size]
        eps: Epsilon for numerical stability
    
    Returns:
        Output tensor with residual added [batch, seq_len, hidden_size]
    """
    # Store input dtype for final conversion
    input_dtype = sublayer_output.dtype
    
    # RMSNorm computation in float32 for numerical stability
    normalized = sublayer_output.to(torch.float32)
    
    # Compute variance: mean of squared values along hidden dimension
    # Shape: [batch, seq_len, 1]
    variance = normalized.pow(2).mean(-1, keepdim=True)
    
    # Normalize: x * rsqrt(variance + eps)
    # rsqrt is more efficient than 1/sqrt
    normalized = normalized * torch.rsqrt(variance + eps)
    
    # Apply learned scale (weight parameter)
    normalized = weight.to(torch.float32) * normalized
    
    # Convert back to input dtype
    normalized = normalized.to(input_dtype)
    
    # Add residual connection
    output = residual + normalized
    
    return output
```

## Workloads

```jsonl
{"uuid": "0e60aff3-9424-553b-99ac-4e1657d5cc6b", "axes": {"batch_size": 16, "seq_len": 1024}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0063, "max_rtol": 0.05}}
{"uuid": "371a388c-51f0-5416-a9eb-926337939aee", "axes": {"batch_size": 8, "seq_len": 2048}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0063, "max_rtol": 0.05}}
{"uuid": "11183480-fb43-5c20-a887-7226134c5fc1", "axes": {"batch_size": 32, "seq_len": 256}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0034, "max_rtol": 0.05}}
{"uuid": "ad827ab9-fb43-5e7f-8ab3-c5ca544ad5cb", "axes": {"batch_size": 8, "seq_len": 997}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.009800000000000001, "max_rtol": 0.05}}
{"uuid": "8496a51c-a1a2-5fd9-a168-15e5d1b62a3a", "axes": {"batch_size": 16, "seq_len": 512}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0034, "max_rtol": 0.05}}
{"uuid": "1078ccd0-3870-5c7e-af42-4aeb7ab8d1ed", "axes": {"batch_size": 4, "seq_len": 2048}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0034, "max_rtol": 0.05}}
{"uuid": "1f59fd2c-b24d-5298-adfd-d06581bf7a8a", "axes": {"batch_size": 1, "seq_len": 131}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 1e-05, "max_rtol": 0.05}}
{"uuid": "da811c4c-e55f-5f55-8a81-9ac4c96b12ba", "axes": {"batch_size": 2, "seq_len": 2053}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.005900000000000001, "max_rtol": 0.05}}
{"uuid": "a36d6608-4971-5f89-ad4c-70b7cfc2cd14", "axes": {"batch_size": 2, "seq_len": 4096}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0034, "max_rtol": 0.05}}
{"uuid": "97064acd-a78b-5ff6-879d-d57a186b425b", "axes": {"batch_size": 8, "seq_len": 512}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.005900000000000001, "max_rtol": 0.05}}
{"uuid": "99216e04-e7bb-5c17-945b-f46cf9f37ca6", "axes": {"batch_size": 4, "seq_len": 128}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 1e-05, "max_rtol": 0.05}}
{"uuid": "e82f3830-1095-5d83-8d44-b192ffc2e898", "axes": {"batch_size": 1, "seq_len": 1024}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.0044, "max_rtol": 0.05}}
{"uuid": "2d375849-8474-5f30-97cf-fc5810de29e8", "axes": {"batch_size": 2, "seq_len": 293}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 1e-05, "max_rtol": 0.05}}
{"uuid": "a7de2a7b-5bc4-5e2d-811a-79bfacb87605", "axes": {"batch_size": 2, "seq_len": 2048}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.005900000000000001, "max_rtol": 0.05}}
{"uuid": "72f1b676-a464-5873-9b06-bf18fe6883ee", "axes": {"batch_size": 8, "seq_len": 256}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 0.00073, "max_rtol": 0.05}}
{"uuid": "c192baac-31dd-5462-b926-a68d0d155d5f", "axes": {"batch_size": 1, "seq_len": 128}, "inputs": {"sublayer_output": {"type": "random"}, "residual": {"type": "random"}, "weight": {"type": "random"}, "eps": {"type": "scalar", "value": 1e-06}}, "tolerance": {"max_atol": 1e-05, "max_rtol": 0.05}}
```
