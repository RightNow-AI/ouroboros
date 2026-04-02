"""Tests for the full OuroborosModel."""

import pytest
import torch

from ouroboros.config import OuroborosConfig
from ouroboros.model import OuroborosModel


class TestOuroborosModel:
    def test_creation(self, tiny_config):
        model = OuroborosModel(tiny_config)
        assert model.phase == 1

    def test_forward_phase1(self, tiny_config):
        """Phase 1: fixed depth, no Controller."""
        model = OuroborosModel(tiny_config)
        model.set_phase(1)

        batch, seq = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        labels = torch.randint(0, tiny_config.vocab_size, (batch, seq))

        output = model(input_ids, labels=labels)
        assert "logits" in output
        assert "loss" in output
        assert output["logits"].shape == (batch, seq, tiny_config.vocab_size)
        assert output["loss"].dim() == 0

    def test_forward_phase2(self, tiny_config):
        """Phase 2: Controller active, variable depth."""
        model = OuroborosModel(tiny_config)
        model.set_phase(2)

        batch, seq = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        labels = torch.randint(0, tiny_config.vocab_size, (batch, seq))

        output = model(input_ids, labels=labels)
        assert "logits" in output
        assert "loss" in output

    def test_forward_phase3(self, tiny_config):
        """Phase 3: adaptive depth with ACT."""
        model = OuroborosModel(tiny_config)
        model.set_phase(3)

        batch, seq = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        labels = torch.randint(0, tiny_config.vocab_size, (batch, seq))

        output = model(input_ids, labels=labels)
        assert "logits" in output
        assert "loss" in output
        assert "act_loss" in output

    def test_phase_switching_freezes(self, tiny_config):
        """Phase switching should correctly freeze/unfreeze components."""
        model = OuroborosModel(tiny_config)

        # Phase 1: Controller and Halter frozen
        model.set_phase(1)
        for p in model.controller.parameters():
            assert not p.requires_grad
        for p in model.halter.parameters():
            assert not p.requires_grad
        for p in model.core_block.parameters():
            assert p.requires_grad

        # Phase 2: Controller unfrozen, Halter still frozen
        model.set_phase(2)
        for p in model.controller.parameters():
            assert p.requires_grad
        for p in model.halter.parameters():
            assert not p.requires_grad

        # Phase 3: All unfrozen
        model.set_phase(3)
        for p in model.controller.parameters():
            assert p.requires_grad
        for p in model.halter.parameters():
            assert p.requires_grad

    def test_gradient_flow_phase1(self, tiny_config):
        model = OuroborosModel(tiny_config)
        model.set_phase(1)
        model.train()

        batch, seq = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        labels = torch.randint(0, tiny_config.vocab_size, (batch, seq))

        output = model(input_ids, labels=labels)
        output["loss"].backward()

        # CoreBlock should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.core_block.parameters()
            if p.requires_grad
        )
        assert has_grad, "CoreBlock should have gradients in phase 1"

    def test_gradient_flow_phase2(self, tiny_config):
        model = OuroborosModel(tiny_config)
        model.set_phase(2)
        model.train()

        # Make Controller coefficients non-zero so gradients can flow
        # (zero-init means zero output → zero gradients by design)
        with torch.no_grad():
            for head in model.controller.heads.values():
                head.coeff_proj_A.weight.fill_(0.1)
                head.coeff_proj_B.weight.fill_(0.1)

        batch, seq = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        labels = torch.randint(0, tiny_config.vocab_size, (batch, seq))

        output = model(input_ids, labels=labels)
        output["loss"].backward()

        # Controller should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.controller.parameters()
            if p.requires_grad
        )
        assert has_grad, "Controller should have gradients in phase 2"

    def test_gradient_flow_phase3(self, tiny_config):
        model = OuroborosModel(tiny_config)
        model.set_phase(3)
        model.train()

        batch, seq = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        labels = torch.randint(0, tiny_config.vocab_size, (batch, seq))

        output = model(input_ids, labels=labels)
        output["loss"].backward()

        # Halter should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.halter.parameters()
            if p.requires_grad
        )
        assert has_grad, "Halter should have gradients in phase 3"

    def test_generate(self, tiny_config):
        model = OuroborosModel(tiny_config)
        model.set_phase(1)

        prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
        generated = model.generate(prompt, max_new_tokens=8, temperature=0.8)
        assert generated.shape[0] == 1
        assert generated.shape[1] == 4 + 8  # prompt + generated

    def test_param_summary(self, tiny_config):
        model = OuroborosModel(tiny_config)
        summary = model.param_summary()
        assert "total" in summary
        assert "core_block" in summary
        assert "controller" in summary
        assert "halter" in summary

    def test_gradient_checkpointing(self, tiny_config):
        model = OuroborosModel(tiny_config)
        model.set_phase(1)
        model.enable_gradient_checkpointing()
        model.train()

        batch, seq = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        labels = torch.randint(0, tiny_config.vocab_size, (batch, seq))

        output = model(input_ids, labels=labels)
        output["loss"].backward()  # Should not error

    def test_no_lora_leaks_between_steps(self, tiny_config):
        """LoRA should be cleared after each recursion step."""
        model = OuroborosModel(tiny_config)
        model.set_phase(2)

        batch, seq = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (batch, seq))
        model(input_ids)

        # After forward, all LoRA should be cleared
        for m in model.core_block.get_lora_modules().values():
            assert not m.has_lora
