"""Tests for Halter and ACTManager."""

import pytest
import torch

from ouroboros.config import OuroborosConfig
from ouroboros.halter import ACTManager, Halter


class TestHalter:
    def test_output_shape(self, tiny_config):
        halter = Halter(tiny_config)
        d = tiny_config.core_block.d_model
        h = torch.randn(2, 8, d)
        prob = halter(h)
        assert prob.shape == (2, 8)

    def test_output_range(self, tiny_config):
        """Halt probabilities should be in (0, 1)."""
        halter = Halter(tiny_config)
        d = tiny_config.core_block.d_model
        h = torch.randn(2, 8, d)
        prob = halter(h)
        assert (prob > 0).all()
        assert (prob < 1).all()

    def test_initial_halt_prob(self, tiny_config):
        """With bias init -2.0, initial halt prob should be ~0.12."""
        halter = Halter(tiny_config)
        d = tiny_config.core_block.d_model
        # Use zeros input to isolate bias effect
        h = torch.zeros(1, 1, d)
        prob = halter(h)
        # sigmoid(-2) ≈ 0.119
        assert prob.item() == pytest.approx(0.119, abs=0.05)

    def test_gradient_flow(self, tiny_config):
        halter = Halter(tiny_config)
        d = tiny_config.core_block.d_model
        h = torch.randn(2, 8, d, requires_grad=True)
        prob = halter(h)
        prob.sum().backward()
        assert h.grad is not None


class TestACTManager:
    def test_initial_state(self):
        act = ACTManager(2, 4, 16, torch.device("cpu"))
        assert act.cumulative_halt.shape == (2, 4)
        assert act.halted.shape == (2, 4)
        assert act.accumulated_output.shape == (2, 4, 16)
        assert not act.halted.any()

    def test_step_accumulates(self):
        act = ACTManager(1, 1, 4, torch.device("cpu"))
        h = torch.ones(1, 1, 4)
        halt_prob = torch.tensor([[0.3]])

        all_halted, running = act.step(h, halt_prob)
        assert not all_halted
        assert act.cumulative_halt[0, 0].item() == pytest.approx(0.3, abs=1e-5)

    def test_halting_at_threshold(self):
        """Token should halt when cumulative >= 1.0."""
        act = ACTManager(1, 1, 4, torch.device("cpu"))
        h = torch.ones(1, 1, 4)

        # Step 1: prob=0.4, cumulative=0.4
        act.step(h, torch.tensor([[0.4]]))
        assert not act.halted[0, 0]

        # Step 2: prob=0.4, cumulative=0.8
        act.step(h, torch.tensor([[0.4]]))
        assert not act.halted[0, 0]

        # Step 3: prob=0.4, cumulative would be 1.2 → halts
        act.step(h, torch.tensor([[0.4]]))
        assert act.halted[0, 0]

    def test_remainder_sums_to_one(self):
        """All ACT weights should sum to 1.0 per token."""
        act = ACTManager(1, 1, 4, torch.device("cpu"))

        # Run several steps with known halt probs
        for i in range(10):
            h = torch.ones(1, 1, 4) * (i + 1)
            prob = torch.tensor([[0.2]])
            all_halted, _ = act.step(h, prob)
            if all_halted:
                break

        result = act.finalize(torch.ones(1, 1, 4) * 99)

        # The accumulated output represents weighted sum
        # We can verify by checking the weight contributions
        # cumulative should be 1.0 after halting
        assert act.cumulative_halt[0, 0].item() == pytest.approx(1.0, abs=1e-5)

    def test_finalize_forces_halt(self):
        """Finalize should handle tokens that never naturally halted."""
        act = ACTManager(1, 1, 4, torch.device("cpu"))
        h = torch.ones(1, 1, 4)

        # Only one step with very low prob
        act.step(h, torch.tensor([[0.01]]))
        assert not act.halted[0, 0]

        # Finalize should force halt
        result = act.finalize(h)
        assert act.halted[0, 0]
        assert result.shape == (1, 1, 4)

    def test_act_loss_scalar(self):
        act = ACTManager(2, 4, 8, torch.device("cpu"))
        h = torch.randn(2, 4, 8)
        act.step(h, torch.full((2, 4), 0.3))
        loss = act.act_loss()
        assert loss.dim() == 0  # scalar

    def test_mean_steps(self):
        act = ACTManager(1, 2, 4, torch.device("cpu"))
        h = torch.ones(1, 2, 4)

        # 3 steps
        for _ in range(3):
            act.step(h, torch.tensor([[0.1, 0.5]]))

        mean = act.mean_steps()
        assert mean > 0

    def test_mixed_halting(self):
        """Different tokens should halt at different steps."""
        act = ACTManager(1, 2, 4, torch.device("cpu"))
        h = torch.ones(1, 2, 4)

        # Token 0: low halt prob, token 1: high halt prob
        for _ in range(10):
            halt_prob = torch.tensor([[0.1, 0.8]])
            all_halted, _ = act.step(h, halt_prob)
            if all_halted:
                break

        # Token 1 should have halted earlier (fewer steps)
        assert act.n_steps[0, 1] < act.n_steps[0, 0]
