"""KernelBench problem: fused Gemma decoder layer (Pi0.5 VLM / prefix side).

Reference math is a standard `transformers` GemmaDecoderLayer on the prefix
(standard RMSNorm, NOT the adaRMS action-expert side), i.e. built with
`use_adarms=False`. Parsed from problems/gemma_fused_attn/ref_tests.py whose
`build_reference_layer` is the accuracy contract.

Structure reproduced (GemmaDecoderLayer.forward, use_adarms=False):
    r = x
    h = RMSNorm_in(x)
    h = Attn(h) + r
    r = h
    h = RMSNorm_post(h)
    h = MLP(h) + r

RMSNorm : fp32 variance, scale by (1 + weight).
MLP     : down(gelu_tanh(gate(h)) * up(h)), no bias.
Attn    : GQA (8 q heads, 1 kv head, head_dim 256), rotary (theta 1e4,
          positions = arange(seq)), scaling = head_dim**-0.5, softmax in fp32,
          full non-causal prefix attention (additive zero mask), no bias.

Because the reference is exercised with position_ids = arange(seq) and an
all-zero additive attention mask (full prefix attention), the layer is a pure
function of the hidden states and the weights: rotary tables and the (no-op)
mask are computed inside forward.  bf16 fused kernels reorder fp ops, so
bit-exactness is not expected (~1e-2 relative for bf16).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms_norm(x, weight, eps):
    """Gemma RMSNorm: fp32 variance, scale by (1 + weight); back to x.dtype."""
    dtype = x.dtype
    var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normed = x * torch.rsqrt(var + eps)                 # bf16 * fp32 -> fp32
    normed = normed * (1.0 + weight.float())
    return normed.to(dtype)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Model(nn.Module):
    """Standard Gemma decoder layer (prefix side, standard RMSNorm)."""

    def __init__(self, hidden_size, intermediate_size, num_attention_heads,
                 num_key_value_heads, head_dim, eps, rope_theta):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim ** -0.5
        self.eps = eps
        self.rope_theta = rope_theta

        q_dim = num_attention_heads * head_dim
        kv_dim = num_key_value_heads * head_dim

        # RMSNorm scales (gemma applies 1 + weight)
        self.input_layernorm_weight = nn.Parameter(torch.zeros(hidden_size))
        self.post_attention_layernorm_weight = nn.Parameter(torch.zeros(hidden_size))

        # Attention projections (no bias)
        self.q_proj_weight = nn.Parameter(torch.empty(q_dim, hidden_size))
        self.k_proj_weight = nn.Parameter(torch.empty(kv_dim, hidden_size))
        self.v_proj_weight = nn.Parameter(torch.empty(kv_dim, hidden_size))
        self.o_proj_weight = nn.Parameter(torch.empty(hidden_size, q_dim))

        # MLP projections (no bias)
        self.gate_proj_weight = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        self.up_proj_weight = nn.Parameter(torch.empty(intermediate_size, hidden_size))
        self.down_proj_weight = nn.Parameter(torch.empty(hidden_size, intermediate_size))

        for p in (self.q_proj_weight, self.k_proj_weight, self.v_proj_weight,
                  self.o_proj_weight, self.gate_proj_weight, self.up_proj_weight,
                  self.down_proj_weight):
            nn.init.normal_(p, std=0.02)

    def _rope_tables(self, seq_len, device, dtype):
        """cos/sin for positions = arange(seq_len), gemma default rope."""
        half = self.head_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, half, device=device).float() * 2.0 / self.head_dim)
        )
        pos = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(pos, inv_freq)                  # [seq, half]
        emb = torch.cat((freqs, freqs), dim=-1)             # [seq, head_dim]
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, hidden_states) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden_size]
        Returns:
            [batch, seq, hidden_size]
        """
        bsz, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # ---- input RMSNorm ----
        residual = hidden_states
        h = _rms_norm(hidden_states, self.input_layernorm_weight, self.eps)

        # ---- self attention (GQA + rotary, full prefix / zero mask) ----
        q = F.linear(h, self.q_proj_weight).view(
            bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = F.linear(h, self.k_proj_weight).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = F.linear(h, self.v_proj_weight).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._rope_tables(seq_len, hidden_states.device, dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)                 # [1,1,seq,head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        # GQA: expand kv heads
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        attn = torch.matmul(q, k.transpose(2, 3)) * self.scaling  # + 0 mask (full prefix)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(dtype)
        o = torch.matmul(attn, v)                           # [b,heads,seq,head_dim]
        o = o.transpose(1, 2).reshape(bsz, seq_len, -1)
        o = F.linear(o, self.o_proj_weight)
        h = residual + o

        # ---- post-attention RMSNorm + MLP ----
        residual = h
        h = _rms_norm(h, self.post_attention_layernorm_weight, self.eps)
        gate = F.gelu(F.linear(h, self.gate_proj_weight), approximate="tanh")
        up = F.linear(h, self.up_proj_weight)
        h = F.linear(gate * up, self.down_proj_weight)
        return residual + h


# ---- pi0.5 VLM (gemma_2b) workload, from openpi get_config("gemma_2b") ----
hidden_size = 2048
intermediate_size = 16384
num_attention_heads = 8
num_key_value_heads = 1          # gemma_2b GQA
head_dim = 256
eps = 1e-6
rope_theta = 10000.0
batch_size = 2
seq_len = 968                    # prefix tokens (256*3 imgs + 200 lang)


def get_inputs():
    # bf16 in the real workload; harness handles device/dtype.
    return [torch.randn(batch_size, seq_len, hidden_size)]


def get_init_inputs():
    return [hidden_size, intermediate_size, num_attention_heads,
            num_key_value_heads, head_dim, eps, rope_theta]
