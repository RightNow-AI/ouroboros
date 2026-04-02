"""OuroborosRefiner — applies SGRC on top of an intact pretrained model.

Instead of destroying the base model by averaging layers, this approach:
1. Keeps the full pretrained model intact (all 24 layers)
2. Wraps the LAST layer's linear projections in DynamicLoRALinear
3. After the normal forward pass, runs recursive refinement steps
   where the Controller generates unique LoRA deltas per step
4. Halter decides when to stop refining (adaptive depth)

This is the cleanest test of SGRC: base model performance + Controller = improvement?
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


class OuroborosRefiner(nn.Module):
    """Ouroboros applied as a refinement stage on top of an intact pretrained model.

    The base model runs its full forward pass normally. Then the last transformer
    layer is reused as a recursive refinement block, with the Controller generating
    unique LoRA deltas at each refinement step.
    """

    def __init__(self, config: OuroborosConfig, base_model_name: str | None = None) -> None:
        super().__init__()
        self.config = config
        self._phase: int = 2

        self.base_model: nn.Module | None = None
        self.lm_head: nn.Module | None = None
        self.norm: nn.Module | None = None

        self.controller = Controller(config)
        self.halter = Halter(config)
        self._lora_modules: dict[str, DynamicLoRALinear] = {}

        if base_model_name:
            self.load_base_model(base_model_name)

    def load_base_model(self, model_name: str, token: str | None = None) -> None:
        """Load a pretrained model and wrap the last layer for SGRC."""
        from transformers import AutoModelForCausalLM

        logger.info(f"Loading base model: {model_name}")
        pretrained = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, token=token,
        )

        self.base_model = pretrained.model
        self.lm_head = pretrained.lm_head
        self.norm = self.base_model.norm

        # Freeze everything
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

        # Wrap last layer projections in DynamicLoRALinear
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

        logger.info(f"Wrapped {len(self._lora_modules)} projections in last layer")
        base_params = sum(p.numel() for p in self.base_model.parameters())
        ctrl_params = sum(p.numel() for p in self.controller.parameters())
        halt_params = sum(p.numel() for p in self.halter.parameters())
        # Match Controller/Halter dtype to base model
        model_dtype = next(self.base_model.parameters()).dtype
        self.controller = self.controller.to(dtype=model_dtype)
        self.halter = self.halter.to(dtype=model_dtype)

        logger.info(f"Base model: {format_params(base_params)} (frozen)")
        logger.info(f"Controller: {format_params(ctrl_params)} (trainable)")
        logger.info(f"Halter: {format_params(halt_params)} (trainable)")

    def set_all_lora(self, deltas: dict[str, tuple[Tensor, Tensor]]) -> None:
        for name, (A, B) in deltas.items():
            if name in self._lora_modules:
                self._lora_modules[name].set_lora(A, B)

    def clear_all_lora(self) -> None:
        for module in self._lora_modules.values():
            module.clear_lora()

    def set_phase(self, phase: int) -> None:
        assert phase in (2, 3), f"Refiner supports phase 2 or 3, got {phase}"
        self._phase = phase
        if self.base_model is not None:
            self.base_model.requires_grad_(False)
        if self.lm_head is not None:
            self.lm_head.requires_grad_(False)
        self.controller.requires_grad_(True)
        self.halter.requires_grad_(phase >= 3)
        desc = {2: "Controller active, fixed depth", 3: "Full SGRC: adaptive depth"}
        logger.info(f"Refiner phase {phase}: {desc[phase]}")

    def _get_position_embeddings(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Get rotary position embeddings from the base model."""
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        # Create a dummy hidden state to get the right dtype
        dummy = torch.zeros(1, seq_len, self.config.core_block.d_model, device=device,
                           dtype=next(self.base_model.parameters()).dtype)
        cos, sin = self.base_model.rotary_emb(dummy, position_ids)
        return cos, sin

    def _run_last_layer(self, hidden_states: Tensor, position_embeddings: tuple[Tensor, Tensor] | None = None) -> Tensor:
        last_layer = self.base_model.layers[-1]
        outputs = last_layer(hidden_states, position_embeddings=position_embeddings)
        return outputs[0] if isinstance(outputs, tuple) else outputs

    def _refinement_fixed(self, h: Tensor, depth: int, pos_emb: tuple[Tensor, Tensor] | None = None) -> Tensor:
        for step in range(depth):
            halt_conf = torch.zeros(h.shape[0], 1, device=h.device)
            deltas = self.controller(h, step, halt_conf)
            self.set_all_lora(deltas)
            h = self._run_last_layer(h, pos_emb)
            self.clear_all_lora()
        return h

    def _refinement_adaptive(self, h: Tensor, max_steps: int, pos_emb: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        batch, seq_len, d_model = h.shape
        act = ACTManager(batch, seq_len, d_model, h.device, h.dtype)

        for step in range(max_steps):
            halt_conf = act.cumulative_halt.mean(dim=1, keepdim=True)
            deltas = self.controller(h, step, halt_conf)
            self.set_all_lora(deltas)
            h = self._run_last_layer(h, pos_emb)
            self.clear_all_lora()

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
        """Forward: base model -> recursive refinement -> LM head."""
        # Step 1: Full base model forward, get hidden states before last layer
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_hidden = base_outputs.hidden_states
        h_before_last = all_hidden[-2]  # before last layer

        seq_len = input_ids.shape[1]
        pos_emb = self._get_position_embeddings(seq_len, input_ids.device)

        # Step 2: Run last layer normally (base behavior)
        h = self._run_last_layer(h_before_last, pos_emb)

        # Step 3: Recursive refinement with SGRC
        act_loss = torch.tensor(0.0, device=h.device)

        if self._phase == 2:
            depth = random.choice(self.config.training.phase2_depths)
            h = self._refinement_fixed(h, depth, pos_emb)
        else:
            max_steps = self.config.training.max_recursion_steps
            h, act_loss = self._refinement_adaptive(h, max_steps, pos_emb)

        # Step 4: Norm + LM head
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
        """Forward through base model ONLY (no refinement). For A/B comparison."""
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
