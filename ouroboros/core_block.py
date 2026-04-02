"""CoreBlock — shared transformer layer for Ouroboros.

Standard Llama-style transformer block with all linear layers wrapped in
DynamicLoRALinear for runtime weight modification by the Controller.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ouroboros.config import OuroborosConfig
from ouroboros.dynamic_lora import DynamicLoRALinear


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, seq, d_model]
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).

    Uses complex number implementation for efficiency.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 500000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency tensor
        # freqs: [max_seq_len, head_dim // 2] as complex numbers
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build the complex frequency cache."""
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, head_dim // 2]
        # Complex exponentials: e^(i*theta) = cos(theta) + i*sin(theta)
        emb = torch.polar(torch.ones_like(freqs), freqs)  # [seq_len, head_dim // 2]
        self.register_buffer("freqs_cis", emb, persistent=False)

    def forward(self, seq_len: int) -> Tensor:
        """Return complex frequency tensor for the given sequence length.

        Returns:
            Complex tensor of shape [seq_len, head_dim // 2].
        """
        if seq_len > self.freqs_cis.shape[0]:
            self._build_cache(seq_len)
        return self.freqs_cis[:seq_len]


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply rotary embeddings to input tensor.

    Args:
        x: Input tensor. Shape [batch, n_heads, seq_len, head_dim].
        freqs_cis: Complex frequency tensor. Shape [seq_len, head_dim // 2].

    Returns:
        Tensor with rotary embeddings applied. Same shape as input.
    """
    # Reshape x to pairs: [batch, n_heads, seq, head_dim//2, 2]
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    # Convert to complex: [batch, n_heads, seq, head_dim//2]
    x_complex = torch.view_as_complex(x_pairs)

    # Broadcast freqs_cis: [1, 1, seq, head_dim//2]
    freqs = freqs_cis.unsqueeze(0).unsqueeze(0)

    # Apply rotation via complex multiplication
    x_rotated = x_complex * freqs

    # Convert back to real: [batch, n_heads, seq, head_dim//2, 2] -> [batch, n_heads, seq, head_dim]
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with all projections wrapped in DynamicLoRALinear.

    Supports GQA where n_heads > n_kv_heads (key/value heads are shared).
    """

    def __init__(self, config: OuroborosConfig) -> None:
        super().__init__()
        cb = config.core_block
        self.n_heads = cb.n_heads
        self.n_kv_heads = cb.n_kv_heads
        self.head_dim = cb.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # GQA repetition factor

        d_model = cb.d_model
        d_kv = self.n_kv_heads * self.head_dim
        rank = config.lora.rank
        alpha = config.lora.alpha

        # All 4 projections wrapped in DynamicLoRALinear
        qkv_bias = cb.qkv_bias
        self.q_proj = DynamicLoRALinear(nn.Linear(d_model, d_model, bias=qkv_bias), rank, alpha)
        self.k_proj = DynamicLoRALinear(nn.Linear(d_model, d_kv, bias=qkv_bias), rank, alpha)
        self.v_proj = DynamicLoRALinear(nn.Linear(d_model, d_kv, bias=qkv_bias), rank, alpha)
        self.o_proj = DynamicLoRALinear(nn.Linear(d_model, d_model, bias=False), rank, alpha)

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=cb.max_seq_len,
            theta=cb.rope_theta,
        )

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Forward pass with grouped-query attention.

        Args:
            x: Input tensor. Shape [batch, seq_len, d_model].
            attention_mask: Optional causal mask. Shape [batch, 1, seq_len, seq_len].

        Returns:
            Output tensor. Shape [batch, seq_len, d_model].
        """
        batch, seq_len, _ = x.shape

        # Project queries, keys, values
        q = self.q_proj(x)  # [batch, seq, n_heads * head_dim]
        k = self.k_proj(x)  # [batch, seq, n_kv_heads * head_dim]
        v = self.v_proj(x)  # [batch, seq, n_kv_heads * head_dim]

        # Reshape to [batch, n_heads, seq, head_dim]
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings to Q and K
        freqs_cis = self.rotary_emb(seq_len)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # Expand KV heads for GQA: [batch, n_kv_heads, seq, dim] -> [batch, n_heads, seq, dim]
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention (auto-selects Flash Attention on supported hardware)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None),
        )  # [batch, n_heads, seq, head_dim]

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_out)  # [batch, seq, d_model]


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network with all projections wrapped in DynamicLoRALinear.

    SwiGLU(x) = (gate_proj(x) * SiLU(up_proj(x))) down-projected.
    Note: Llama uses gate_proj for the gating and up_proj for the value,
    with SiLU activation on gate_proj.
    """

    def __init__(self, config: OuroborosConfig) -> None:
        super().__init__()
        d_model = config.core_block.d_model
        d_ff = config.core_block.d_ff
        rank = config.lora.rank
        alpha = config.lora.alpha

        self.gate_proj = DynamicLoRALinear(nn.Linear(d_model, d_ff, bias=False), rank, alpha)
        self.up_proj = DynamicLoRALinear(nn.Linear(d_model, d_ff, bias=False), rank, alpha)
        self.down_proj = DynamicLoRALinear(nn.Linear(d_ff, d_model, bias=False), rank, alpha)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Shape [batch, seq_len, d_model].

        Returns:
            Output tensor. Shape [batch, seq_len, d_model].
        """
        # SwiGLU: SiLU(gate) * up, then down-project
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CoreBlock(nn.Module):
    """Shared transformer block for Ouroboros.

    Pre-norm architecture with GQA attention and SwiGLU FFN.
    All 7 linear layers are wrapped in DynamicLoRALinear for runtime modification.
    """

    def __init__(self, config: OuroborosConfig) -> None:
        super().__init__()
        d_model = config.core_block.d_model
        eps = config.core_block.rms_norm_eps

        self.attn_norm = RMSNorm(d_model, eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(d_model, eps)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor. Shape [batch, seq_len, d_model].
            attention_mask: Optional attention mask.

        Returns:
            Output tensor. Shape [batch, seq_len, d_model].
        """
        # Pre-norm attention with residual
        h = x + self.attention(self.attn_norm(x), attention_mask)
        # Pre-norm FFN with residual
        h = h + self.ffn(self.ffn_norm(h))
        return h

    def get_lora_modules(self) -> dict[str, DynamicLoRALinear]:
        """Return all DynamicLoRALinear modules by their target name.

        Returns:
            Dict mapping target name to DynamicLoRALinear module.
            Keys: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
        """
        return {
            "q_proj": self.attention.q_proj,
            "k_proj": self.attention.k_proj,
            "v_proj": self.attention.v_proj,
            "o_proj": self.attention.o_proj,
            "gate_proj": self.ffn.gate_proj,
            "up_proj": self.ffn.up_proj,
            "down_proj": self.ffn.down_proj,
        }

    def set_all_lora(self, deltas: dict[str, tuple[Tensor, Tensor]]) -> None:
        """Set LoRA matrices on all targets from Controller output.

        Args:
            deltas: Dict mapping target name to (lora_A, lora_B) tuple.
        """
        modules = self.get_lora_modules()
        for name, (A, B) in deltas.items():
            modules[name].set_lora(A, B)

    def clear_all_lora(self) -> None:
        """Clear LoRA matrices from all targets."""
        for module in self.get_lora_modules().values():
            module.clear_lora()
