"""Tests for CoreBlock components."""

import pytest
import torch

from ouroboros.config import OuroborosConfig
from ouroboros.core_block import (
    CoreBlock,
    GroupedQueryAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUFFN,
    apply_rotary_emb,
)


class TestRMSNorm:
    def test_output_shape(self, tiny_config):
        d = tiny_config.core_block.d_model
        norm = RMSNorm(d)
        x = torch.randn(2, 8, d)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self, tiny_config):
        d = tiny_config.core_block.d_model
        norm = RMSNorm(d)
        x = torch.randn(2, 8, d)
        out = norm(x)
        # RMS of normalized output should be close to 1
        rms = out.pow(2).mean(-1).sqrt()
        assert rms.mean().item() == pytest.approx(1.0, abs=0.3)


class TestRotaryEmbedding:
    def test_freqs_shape(self, tiny_config):
        head_dim = tiny_config.core_block.head_dim
        rope = RotaryEmbedding(head_dim, max_seq_len=64)
        freqs = rope(32)
        assert freqs.shape == (32, head_dim // 2)

    def test_apply_rotary_preserves_shape(self, tiny_config):
        head_dim = tiny_config.core_block.head_dim
        rope = RotaryEmbedding(head_dim, max_seq_len=64)
        freqs = rope(16)
        x = torch.randn(2, 4, 16, head_dim)  # [batch, heads, seq, dim]
        out = apply_rotary_emb(x, freqs)
        assert out.shape == x.shape


class TestGroupedQueryAttention:
    def test_output_shape(self, tiny_config):
        attn = GroupedQueryAttention(tiny_config)
        d = tiny_config.core_block.d_model
        x = torch.randn(2, 16, d)
        out = attn(x)
        assert out.shape == (2, 16, d)

    def test_lora_targets(self, tiny_config):
        attn = GroupedQueryAttention(tiny_config)
        assert hasattr(attn, "q_proj")
        assert hasattr(attn, "k_proj")
        assert hasattr(attn, "v_proj")
        assert hasattr(attn, "o_proj")


class TestSwiGLUFFN:
    def test_output_shape(self, tiny_config):
        ffn = SwiGLUFFN(tiny_config)
        d = tiny_config.core_block.d_model
        x = torch.randn(2, 16, d)
        out = ffn(x)
        assert out.shape == (2, 16, d)


class TestCoreBlock:
    def test_output_shape(self, tiny_config):
        block = CoreBlock(tiny_config)
        d = tiny_config.core_block.d_model
        x = torch.randn(2, 16, d)
        out = block(x)
        assert out.shape == (2, 16, d)

    def test_residual_connection(self, tiny_config):
        """Output should not be identical to input (processing happened)."""
        block = CoreBlock(tiny_config)
        d = tiny_config.core_block.d_model
        x = torch.randn(2, 16, d)
        out = block(x)
        assert not torch.allclose(x, out)

    def test_get_lora_modules(self, tiny_config):
        block = CoreBlock(tiny_config)
        modules = block.get_lora_modules()
        assert set(modules.keys()) == {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        }

    def test_set_clear_all_lora(self, tiny_config):
        block = CoreBlock(tiny_config)
        d = tiny_config.core_block.d_model
        r = tiny_config.lora.rank

        # Build fake deltas
        dims = tiny_config.target_dims()
        deltas = {}
        for name in tiny_config.lora.TARGET_NAMES:
            in_f, out_f = dims[name]
            A = torch.randn(2, r, in_f)
            B = torch.randn(2, out_f, r)
            deltas[name] = (A, B)

        # Set and verify
        block.set_all_lora(deltas)
        for m in block.get_lora_modules().values():
            assert m.has_lora

        # Clear and verify
        block.clear_all_lora()
        for m in block.get_lora_modules().values():
            assert not m.has_lora

    def test_lora_affects_output(self, tiny_config):
        """Setting LoRA should change the block output."""
        block = CoreBlock(tiny_config)
        d = tiny_config.core_block.d_model
        r = tiny_config.lora.rank
        x = torch.randn(2, 16, d)

        out_base = block(x).clone()

        # Set large LoRA deltas
        dims = tiny_config.target_dims()
        deltas = {}
        for name in tiny_config.lora.TARGET_NAMES:
            in_f, out_f = dims[name]
            A = torch.randn(2, r, in_f) * 0.1
            B = torch.randn(2, out_f, r) * 0.1
            deltas[name] = (A, B)

        block.set_all_lora(deltas)
        out_lora = block(x)
        block.clear_all_lora()

        assert not torch.allclose(out_base, out_lora, atol=1e-5)

    def test_gradient_flow(self, tiny_config):
        """Gradients should flow through the block."""
        block = CoreBlock(tiny_config)
        d = tiny_config.core_block.d_model
        x = torch.randn(2, 16, d, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
