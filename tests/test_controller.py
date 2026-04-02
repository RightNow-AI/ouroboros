"""Tests for Controller and ControllerHead."""

import pytest
import torch

from ouroboros.config import OuroborosConfig
from ouroboros.controller import Controller, ControllerHead


class TestControllerHead:
    def test_output_shapes(self):
        head = ControllerHead(
            style_dim=16, rank=4, in_features=32, out_features=64,
            n_basis=4, bottleneck_divisor=2,
        )
        style = torch.randn(2, 16)
        A, B = head(style)
        assert A.shape == (2, 4, 32)  # [batch, rank, in]
        assert B.shape == (2, 64, 4)  # [batch, out, rank]

    def test_zero_init(self):
        """Coefficient projections should be zero-initialized → near-zero output."""
        head = ControllerHead(
            style_dim=16, rank=4, in_features=32, out_features=64,
            n_basis=4, bottleneck_divisor=2,
        )
        style = torch.randn(2, 16)
        A, B = head(style)

        # With zero-initialized coefficients, output should be near zero
        assert A.abs().max().item() < 1e-6
        assert B.abs().max().item() < 1e-6

    def test_gradient_flow(self):
        head = ControllerHead(
            style_dim=16, rank=4, in_features=32, out_features=64,
            n_basis=4, bottleneck_divisor=2,
        )
        style = torch.randn(2, 16, requires_grad=True)
        A, B = head(style)
        loss = A.sum() + B.sum()
        loss.backward()
        assert style.grad is not None

    def test_different_styles_different_outputs(self):
        """Different style vectors should produce different LoRA matrices."""
        head = ControllerHead(
            style_dim=16, rank=4, in_features=32, out_features=64,
            n_basis=4, bottleneck_divisor=2,
        )
        # After training, coefficients would be non-zero.
        # Manually make them non-zero for this test.
        with torch.no_grad():
            head.coeff_proj_A.weight.fill_(0.1)
            head.coeff_proj_B.weight.fill_(0.1)

        style1 = torch.randn(1, 16)
        style2 = torch.randn(1, 16)

        A1, B1 = head(style1)
        A2, B2 = head(style2)

        assert not torch.allclose(A1, A2, atol=1e-6)
        assert not torch.allclose(B1, B2, atol=1e-6)


class TestController:
    def test_output_keys(self, tiny_config):
        controller = Controller(tiny_config)
        d = tiny_config.core_block.d_model
        h = torch.randn(2, 8, d)
        halt_conf = torch.zeros(2, 1)

        deltas = controller(h, step=0, halt_confidence=halt_conf)

        expected_keys = set(tiny_config.lora.TARGET_NAMES)
        assert set(deltas.keys()) == expected_keys

    def test_output_shapes(self, tiny_config):
        controller = Controller(tiny_config)
        d = tiny_config.core_block.d_model
        r = tiny_config.lora.rank
        h = torch.randn(2, 8, d)
        halt_conf = torch.zeros(2, 1)

        deltas = controller(h, step=0, halt_confidence=halt_conf)
        dims = tiny_config.target_dims()

        for name, (A, B) in deltas.items():
            in_f, out_f = dims[name]
            assert A.shape == (2, r, in_f), f"{name}: A shape mismatch"
            assert B.shape == (2, out_f, r), f"{name}: B shape mismatch"

    def test_zero_init_produces_near_zero(self, tiny_config):
        """Fresh Controller should produce near-zero deltas."""
        controller = Controller(tiny_config)
        d = tiny_config.core_block.d_model
        h = torch.randn(2, 8, d)
        halt_conf = torch.zeros(2, 1)

        deltas = controller(h, step=0, halt_confidence=halt_conf)

        for name, (A, B) in deltas.items():
            assert A.abs().max().item() < 1e-5, f"{name}: A not near zero"
            assert B.abs().max().item() < 1e-5, f"{name}: B not near zero"

    def test_different_steps_different_deltas(self, tiny_config):
        """Different recursion steps should produce different deltas (after training)."""
        controller = Controller(tiny_config)
        # Make coefficients non-zero
        for head in controller.heads.values():
            with torch.no_grad():
                head.coeff_proj_A.weight.fill_(0.1)
                head.coeff_proj_B.weight.fill_(0.1)

        d = tiny_config.core_block.d_model
        h = torch.randn(2, 8, d)
        halt_conf = torch.zeros(2, 1)

        deltas_0 = controller(h, step=0, halt_confidence=halt_conf)
        deltas_5 = controller(h, step=5, halt_confidence=halt_conf)

        # At least some targets should differ
        any_different = False
        for name in deltas_0:
            A0, _ = deltas_0[name]
            A5, _ = deltas_5[name]
            if not torch.allclose(A0, A5, atol=1e-6):
                any_different = True
                break
        assert any_different, "Different steps should produce different deltas"

    def test_different_inputs_different_deltas(self, tiny_config):
        """Different hidden states should produce different deltas (after training)."""
        controller = Controller(tiny_config)
        for head in controller.heads.values():
            with torch.no_grad():
                head.coeff_proj_A.weight.fill_(0.1)
                head.coeff_proj_B.weight.fill_(0.1)

        d = tiny_config.core_block.d_model
        h1 = torch.randn(2, 8, d)
        h2 = torch.randn(2, 8, d)
        halt_conf = torch.zeros(2, 1)

        deltas_1 = controller(h1, step=0, halt_confidence=halt_conf)
        deltas_2 = controller(h2, step=0, halt_confidence=halt_conf)

        any_different = False
        for name in deltas_1:
            A1, _ = deltas_1[name]
            A2, _ = deltas_2[name]
            if not torch.allclose(A1, A2, atol=1e-6):
                any_different = True
                break
        assert any_different, "Different inputs should produce different deltas"

    def test_gradient_flow(self, tiny_config):
        controller = Controller(tiny_config)
        d = tiny_config.core_block.d_model
        h = torch.randn(2, 8, d, requires_grad=True)
        halt_conf = torch.zeros(2, 1)

        deltas = controller(h, step=0, halt_confidence=halt_conf)
        loss = sum(A.sum() + B.sum() for A, B in deltas.values())
        loss.backward()
        assert h.grad is not None

    def test_scalar_halt_confidence(self, tiny_config):
        """Should handle scalar halt confidence."""
        controller = Controller(tiny_config)
        d = tiny_config.core_block.d_model
        h = torch.randn(2, 8, d)
        halt_conf = torch.tensor(0.5)

        deltas = controller(h, step=0, halt_confidence=halt_conf)
        assert len(deltas) == 7  # All targets
