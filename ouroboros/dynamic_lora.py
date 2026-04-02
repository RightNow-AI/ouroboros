"""Dynamic LoRA linear layer for Ouroboros.

Wraps a frozen base nn.Linear and accepts externally-generated LoRA A/B matrices
at runtime. When no LoRA is set, behaves as a standard linear layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DynamicLoRALinear(nn.Module):
    """Linear layer with runtime-injectable LoRA modifications.

    The base linear layer's weights can be frozen while LoRA A/B matrices
    are dynamically set by the Controller at each recursion step.

    Args:
        base_linear: The pretrained linear layer to wrap.
        rank: LoRA rank (bottleneck dimension).
        alpha: LoRA scaling factor. Output is scaled by alpha/rank.
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Runtime-set LoRA matrices (not parameters — set externally)
        self._lora_A: Tensor | None = None  # [batch, rank, in_features] or [rank, in_features]
        self._lora_B: Tensor | None = None  # [batch, out_features, rank] or [out_features, rank]

    @property
    def in_features(self) -> int:
        return self.base_linear.in_features

    @property
    def out_features(self) -> int:
        return self.base_linear.out_features

    @property
    def weight(self) -> Tensor:
        """Access base linear weight for compatibility."""
        return self.base_linear.weight

    @property
    def bias(self) -> Tensor | None:
        """Access base linear bias for compatibility."""
        return self.base_linear.bias

    def set_lora(self, A: Tensor, B: Tensor) -> None:
        """Set LoRA matrices for the next forward pass.

        Args:
            A: LoRA down-projection. Shape [batch, rank, in_features] or [rank, in_features].
            B: LoRA up-projection. Shape [batch, out_features, rank] or [out_features, rank].
        """
        self._lora_A = A
        self._lora_B = B

    def clear_lora(self) -> None:
        """Remove LoRA matrices, reverting to base linear behavior."""
        self._lora_A = None
        self._lora_B = None

    @property
    def has_lora(self) -> bool:
        """Whether LoRA matrices are currently set."""
        return self._lora_A is not None and self._lora_B is not None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optional dynamic LoRA.

        Args:
            x: Input tensor. Shape [batch, seq_len, in_features].

        Returns:
            Output tensor. Shape [batch, seq_len, out_features].
        """
        # Base linear: [batch, seq, in] -> [batch, seq, out]
        out = self.base_linear(x)

        if self.has_lora:
            A = self._lora_A  # [batch, rank, in] or [rank, in]
            B = self._lora_B  # [batch, out, rank] or [out, rank]

            if A.dim() == 2:
                # Unbatched: same LoRA for all batch items
                # x: [batch, seq, in], A: [rank, in] -> [batch, seq, rank]
                lora_out = torch.einsum("bsi,ri->bsr", x, A)
                # lora_out: [batch, seq, rank], B: [out, rank] -> [batch, seq, out]
                lora_out = torch.einsum("bsr,or->bso", lora_out, B)
            else:
                # Batched: different LoRA per batch item
                # x: [batch, seq, in], A: [batch, rank, in] -> [batch, seq, rank]
                lora_out = torch.einsum("bsi,bri->bsr", x, A)
                # lora_out: [batch, seq, rank], B: [batch, out, rank] -> [batch, seq, out]
                lora_out = torch.einsum("bsr,bor->bso", lora_out, B)

            out = out + self.scaling * lora_out

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )
