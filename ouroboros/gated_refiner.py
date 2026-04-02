"""OuroborosGatedRefiner — SGRC with gated recurrence on intact pretrained model.

Key fixes from literature:
1. Gated recurrence: h = gate * h_new + (1-gate) * h_old  (bias -2 = 88% retain)
2. Per-step unique LayerNorm (RingFormer insight)
3. Loss only at final iteration (Huginn insight)
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ouroboros.config import OuroborosConfig
from ouroboros.controller import Controller
from ouroboros.dynamic_lora import DynamicLoRALinear
from ouroboros.halter import Halter, ACTManager
from ouroboros.utils import get_logger, format_params

logger = get_logger(__name__)


class RecurrenceGate(nn.Module):
    """Gated recurrence to prevent drift in recursive loops.

    h_out = gate * h_new + (1 - gate) * h_old

    With bias=-2.0, sigmoid(-2)=0.12, so model retains 88% of old state by default.
    This creates a gradient highway across many recurrence steps.
    Source: "Thinking Deeper, Not Longer" (arXiv:2603.21676)
    """

    def __init__(self, d_model: int, bias_init: float = -2.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, bias_init)

    def forward(self, h_new: Tensor, h_old: Tensor) -> Tensor:
        combined = torch.cat([h_new, h_old], dim=-1)  # [batch, seq, 2*d]
        gate = torch.sigmoid(self.gate_proj(combined))  # [batch, seq, d]
        return gate * h_new + (1 - gate) * h_old


class PerStepNorm(nn.Module):
    """Per-step unique RMSNorm. Each recurrence step gets its own normalization.
    Source: RingFormer (arXiv:2502.13181)
    """

    def __init__(self, d_model: int, max_steps: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(max_steps, d_model))
        self.eps = eps

    def forward(self, x: Tensor, step: int) -> Tensor:
        w = self.weight[min(step, self.weight.shape[0] - 1)]
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * w


class OuroborosGatedRefiner(nn.Module):
    """SGRC with gated recurrence on an intact pretrained model.

    Architecture:
        Base model (frozen, all layers) -> last layer output
        -> [GATED RECURSIVE REFINEMENT with Controller LoRA] -> norm -> LM head

    The gate prevents residual drift. Per-step norms differentiate steps.
    Controller generates LoRA deltas. Together this is clean SGRC.
    """

    def __init__(self, config: OuroborosConfig, base_model_name: str | None = None) -> None:
        super().__init__()
        self.config = config
        self._phase: int = 2

        self.base_model: nn.Module | None = None
        self.lm_head: nn.Module | None = None
        self.norm: nn.Module | None = None

        # SGRC components
        self.controller = Controller(config)
        self.halter = Halter(config)

        # Gated recurrence (the key missing piece)
        self.recurrence_gate = RecurrenceGate(config.core_block.d_model)

        # Per-step normalization
        self.step_norm = PerStepNorm(
            config.core_block.d_model,
            max_steps=config.controller.max_steps,
        )

        self._lora_modules: dict[str, DynamicLoRALinear] = {}

        if base_model_name:
            self.load_base_model(base_model_name)

    def load_base_model(self, model_name: str, token: str | None = None) -> None:
        from transformers import AutoModelForCausalLM

        logger.info(f"Loading base model: {model_name}")
        pretrained = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, token=token,
        )

        self.base_model = pretrained.model
        self.lm_head = pretrained.lm_head
        self.norm = self.base_model.norm

        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

        # Wrap last layer
        last_layer = self.base_model.layers[-1]
        rank = self.config.lora.rank
        alpha = self.config.lora.alpha

        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            original = getattr(last_layer.self_attn, proj_name)
            wrapped = DynamicLoRALinear(original, rank, alpha)
            setattr(last_layer.self_attn, proj_name, wrapped)
            self._lora_modules[proj_name] = wrapped

        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            original = getattr(last_layer.mlp, proj_name)
            wrapped = DynamicLoRALinear(original, rank, alpha)
            setattr(last_layer.mlp, proj_name, wrapped)
            self._lora_modules[proj_name] = wrapped

        # Match dtypes
        model_dtype = next(self.base_model.parameters()).dtype
        self.controller = self.controller.to(dtype=model_dtype)
        self.halter = self.halter.to(dtype=model_dtype)
        self.recurrence_gate = self.recurrence_gate.to(dtype=model_dtype)
        self.step_norm = self.step_norm.to(dtype=model_dtype)

        base_params = sum(p.numel() for p in self.base_model.parameters())
        new_params = (sum(p.numel() for p in self.controller.parameters()) +
                      sum(p.numel() for p in self.halter.parameters()) +
                      sum(p.numel() for p in self.recurrence_gate.parameters()) +
                      sum(p.numel() for p in self.step_norm.parameters()))
        logger.info(f"Base model: {format_params(base_params)} (frozen)")
        logger.info(f"SGRC params: {format_params(new_params)} (trainable)")

    def set_all_lora(self, deltas: dict[str, tuple[Tensor, Tensor]]) -> None:
        for name, (A, B) in deltas.items():
            if name in self._lora_modules:
                self._lora_modules[name].set_lora(A, B)

    def clear_all_lora(self) -> None:
        for module in self._lora_modules.values():
            module.clear_lora()

    def set_phase(self, phase: int) -> None:
        assert phase in (2, 3)
        self._phase = phase
        if self.base_model:
            self.base_model.requires_grad_(False)
        if self.lm_head:
            self.lm_head.requires_grad_(False)
        self.controller.requires_grad_(True)
        self.recurrence_gate.requires_grad_(True)
        self.step_norm.requires_grad_(True)
        self.halter.requires_grad_(phase >= 3)
        logger.info(f"Phase {phase}: {'adaptive' if phase == 3 else 'fixed'} depth")

    def _get_position_embeddings(self, seq_len: int, device: torch.device):
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        dummy = torch.zeros(1, seq_len, self.config.core_block.d_model, device=device,
                           dtype=next(self.base_model.parameters()).dtype)
        return self.base_model.rotary_emb(dummy, position_ids)

    def _run_last_layer(self, h: Tensor, pos_emb) -> Tensor:
        last_layer = self.base_model.layers[-1]
        outputs = last_layer(h, position_embeddings=pos_emb)
        return outputs[0] if isinstance(outputs, tuple) else outputs

    def _gated_refinement(self, h: Tensor, depth: int, pos_emb) -> Tensor:
        """Fixed-depth gated refinement."""
        for step in range(depth):
            halt_conf = torch.zeros(h.shape[0], 1, device=h.device)
            deltas = self.controller(h, step, halt_conf)
            self.set_all_lora(deltas)

            h_new = self._run_last_layer(h, pos_emb)
            self.clear_all_lora()

            # Per-step normalization
            h_new = self.step_norm(h_new, step)

            # Gated recurrence: prevents drift
            h = self.recurrence_gate(h_new, h)

        return h

    def _gated_refinement_adaptive(self, h: Tensor, max_steps: int, pos_emb) -> tuple[Tensor, Tensor]:
        """Adaptive-depth gated refinement with ACT."""
        batch, seq_len, d_model = h.shape
        act = ACTManager(batch, seq_len, d_model, h.device, h.dtype)

        for step in range(max_steps):
            halt_conf = act.cumulative_halt.mean(dim=1, keepdim=True)
            deltas = self.controller(h, step, halt_conf)
            self.set_all_lora(deltas)

            h_new = self._run_last_layer(h, pos_emb)
            self.clear_all_lora()

            h_new = self.step_norm(h_new, step)
            h = self.recurrence_gate(h_new, h)

            halt_prob = self.halter(h)
            all_halted, _ = act.step(h, halt_prob)
            if all_halted:
                break

        h = act.finalize(h)
        return h, act.act_loss()

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        # Full base model forward
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        h_before_last = base_outputs.hidden_states[-2]

        seq_len = input_ids.shape[1]
        pos_emb = self._get_position_embeddings(seq_len, input_ids.device)

        # Run last layer normally first
        h = self._run_last_layer(h_before_last, pos_emb)

        # Gated recursive refinement
        act_loss = torch.tensor(0.0, device=h.device)
        if self._phase == 2:
            depth = random.choice(self.config.training.phase2_depths)
            h = self._gated_refinement(h, depth, pos_emb)
        else:
            h, act_loss = self._gated_refinement_adaptive(
                h, self.config.training.max_recursion_steps, pos_emb,
            )

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if self._phase == 3:
                loss = loss + self.config.halter.act_loss_weight * act_loss

        return {"logits": logits, "loss": loss, "act_loss": act_loss}

    @torch.no_grad()
    def forward_base_only(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Any]:
        """Base model only — no refinement. For A/B comparison."""
        self.clear_all_lora()
        outputs = self.base_model(input_ids=input_ids)
        h = outputs.last_hidden_state
        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}
