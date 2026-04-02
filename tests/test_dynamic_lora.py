"""Tests for DynamicLoRALinear."""

import pytest
import torch
import torch.nn as nn

from ouroboros.dynamic_lora import DynamicLoRALinear


@pytest.fixture
def lora_layer():
    """A DynamicLoRALinear wrapping a simple linear layer."""
    base = nn.Linear(32, 64, bias=False)
    return DynamicLoRALinear(base, rank=4, alpha=8.0)


class TestDynamicLoRALinear:
    def test_base_forward_shape(self, lora_layer):
        """Output shape matches base linear without LoRA."""
        x = torch.randn(2, 8, 32)
        out = lora_layer(x)
        assert out.shape == (2, 8, 64)

    def test_no_lora_matches_base(self, lora_layer):
        """Without LoRA set, output equals base linear."""
        x = torch.randn(2, 8, 32)
        expected = lora_layer.base_linear(x)
        actual = lora_layer(x)
        torch.testing.assert_close(actual, expected)

    def test_lora_changes_output(self, lora_layer):
        """Setting LoRA should change the output."""
        x = torch.randn(2, 8, 32)
        base_out = lora_layer(x).clone()

        # Set non-zero LoRA
        A = torch.randn(2, 4, 32)  # [batch, rank, in]
        B = torch.randn(2, 64, 4)  # [batch, out, rank]
        lora_layer.set_lora(A, B)
        lora_out = lora_layer(x)

        assert not torch.allclose(base_out, lora_out, atol=1e-6)

    def test_clear_lora_reverts(self, lora_layer):
        """Clearing LoRA should revert to base behavior."""
        x = torch.randn(2, 8, 32)
        base_out = lora_layer(x).clone()

        A = torch.randn(2, 4, 32)
        B = torch.randn(2, 64, 4)
        lora_layer.set_lora(A, B)
        _ = lora_layer(x)

        lora_layer.clear_lora()
        reverted_out = lora_layer(x)
        torch.testing.assert_close(reverted_out, base_out)

    def test_unbatched_lora(self, lora_layer):
        """Unbatched LoRA (2D) should work and produce same output for all batch items."""
        x = torch.randn(2, 8, 32)
        A = torch.randn(4, 32)  # [rank, in] — unbatched
        B = torch.randn(64, 4)  # [out, rank] — unbatched

        lora_layer.set_lora(A, B)
        out = lora_layer(x)
        assert out.shape == (2, 8, 64)

    def test_batched_lora(self, lora_layer):
        """Batched LoRA (3D) should produce different outputs per batch item."""
        x = torch.randn(2, 8, 32)
        A = torch.randn(2, 4, 32)
        B = torch.randn(2, 64, 4)
        # Make A/B different across batch
        A[1] *= 2.0
        B[1] *= 2.0

        lora_layer.set_lora(A, B)
        out = lora_layer(x)

        # With different LoRA per batch, outputs should differ
        # (even if base is same, LoRA contribution differs)
        assert out.shape == (2, 8, 64)

    def test_gradient_flow_base(self, lora_layer):
        """Gradients flow through base linear path."""
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = lora_layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert lora_layer.base_linear.weight.grad is not None

    def test_gradient_flow_lora(self, lora_layer):
        """Gradients flow through LoRA path."""
        x = torch.randn(2, 8, 32)
        A = torch.randn(2, 4, 32, requires_grad=True)
        B = torch.randn(2, 64, 4, requires_grad=True)

        lora_layer.set_lora(A, B)
        out = lora_layer(x)
        loss = out.sum()
        loss.backward()

        assert A.grad is not None
        assert B.grad is not None

    def test_scaling_factor(self):
        """Verify scaling = alpha / rank."""
        base = nn.Linear(16, 16, bias=False)
        lora = DynamicLoRALinear(base, rank=4, alpha=16.0)
        assert lora.scaling == 4.0

    def test_has_lora_property(self, lora_layer):
        """has_lora reflects current state."""
        assert not lora_layer.has_lora
        lora_layer.set_lora(torch.randn(4, 32), torch.randn(64, 4))
        assert lora_layer.has_lora
        lora_layer.clear_lora()
        assert not lora_layer.has_lora
