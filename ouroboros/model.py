"""OuroborosModel — the complete architecture.

Combines CoreBlock, Controller, Halter, embeddings, and LM head into a single
model with 3-phase training support:

    Phase 1: Fixed recursion depth, Controller frozen → recover pretrained knowledge
    Phase 2: Variable depth, Controller unfrozen → learn dynamic computation
    Phase 3: Adaptive depth (ACT), all unfrozen → full Ouroboros with reasoning
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ouroboros.config import OuroborosConfig
from ouroboros.core_block import CoreBlock, RMSNorm
from ouroboros.controller import Controller
from ouroboros.halter import Halter, ACTManager
from ouroboros.utils import count_parameters, format_params, get_logger

logger = get_logger(__name__)


class OuroborosModel(nn.Module):
    """Ouroboros: Self-Generating Recursive Computation.

    A model that dynamically generates its own computation at each recursion step
    via a Controller hypernetwork, achieving deep reasoning with minimal parameters.

    Args:
        config: Complete Ouroboros configuration.
    """

    def __init__(self, config: OuroborosConfig) -> None:
        super().__init__()
        self.config = config

        # Embeddings (from pretrained)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.core_block.d_model)

        # Core components
        self.core_block = CoreBlock(config)
        self.controller = Controller(config)
        self.halter = Halter(config)

        # Output head (from pretrained)
        self.norm = RMSNorm(config.core_block.d_model, config.core_block.rms_norm_eps)
        self.lm_head = nn.Linear(config.core_block.d_model, config.vocab_size, bias=False)

        # Training state
        self._phase: int = 1
        self._use_gradient_checkpointing: bool = False

    @property
    def phase(self) -> int:
        return self._phase

    def set_phase(self, phase: int) -> None:
        """Set training phase, adjusting which components are frozen/unfrozen.

        Phase 1: CoreBlock trainable, Controller frozen, Halter frozen.
                 Fixed recursion depth. Recover pretrained knowledge.
        Phase 2: CoreBlock trainable, Controller unfrozen, Halter frozen.
                 Variable depth. Controller learns dynamic computation.
        Phase 3: All unfrozen. Adaptive depth via ACT.
        """
        assert phase in (1, 2, 3), f"Phase must be 1, 2, or 3, got {phase}"
        self._phase = phase

        # Embeddings and LM head: always trainable
        self.embed_tokens.requires_grad_(True)
        self.norm.requires_grad_(True)
        self.lm_head.requires_grad_(True)

        # CoreBlock: always trainable (recovering / adapting shared weights)
        self.core_block.requires_grad_(True)

        # Controller: frozen in phase 1, unfrozen in phase 2+
        self.controller.requires_grad_(phase >= 2)

        # Halter: frozen in phase 1-2, unfrozen in phase 3
        self.halter.requires_grad_(phase >= 3)

        phase_desc = {
            1: "Recover Knowledge (fixed depth, Controller frozen)",
            2: "Awaken Controller (variable depth, Controller active)",
            3: "Full Ouroboros (adaptive depth, all active)",
        }
        logger.info(f"Set training phase {phase}: {phase_desc[phase]}")

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for the recursion loop."""
        self._use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._use_gradient_checkpointing = False

    def _get_recursion_depth(self) -> int:
        """Get recursion depth based on current phase."""
        tc = self.config.training
        if self._phase == 1:
            return tc.phase1_fixed_depth
        elif self._phase == 2:
            return random.choice(tc.phase2_depths)
        else:
            return tc.max_recursion_steps

    def _apply_core_block_with_lora(
        self,
        h: Tensor,
        attention_mask: Tensor | None,
        *lora_flat: Tensor,
    ) -> Tensor:
        """Checkpointable function: set LoRA → core_block → clear LoRA.

        All LoRA tensors are passed as explicit args so gradient checkpointing
        can properly replay the forward pass during recomputation.
        """
        if lora_flat:
            # Reconstruct deltas dict from flat tensor args
            target_names = self.config.lora.TARGET_NAMES
            deltas = {}
            for i, name in enumerate(target_names):
                deltas[name] = (lora_flat[2 * i], lora_flat[2 * i + 1])
            self.core_block.set_all_lora(deltas)

        h = self.core_block(h, attention_mask)
        self.core_block.clear_all_lora()
        return h

    def _step_forward(
        self,
        h: Tensor,
        attention_mask: Tensor | None,
        deltas: dict[str, tuple[Tensor, Tensor]] | None,
    ) -> Tensor:
        """Single recursion step with optional gradient checkpointing."""
        if self._use_gradient_checkpointing and self.training:
            # Flatten deltas into positional tensor args for checkpoint compatibility
            lora_flat: tuple[Tensor, ...] = ()
            if deltas:
                flat_list = []
                for name in self.config.lora.TARGET_NAMES:
                    A, B = deltas[name]
                    flat_list.extend([A, B])
                lora_flat = tuple(flat_list)

            h = checkpoint(
                self._apply_core_block_with_lora,
                h, attention_mask, *lora_flat,
                use_reentrant=False,
            )
        else:
            if deltas:
                self.core_block.set_all_lora(deltas)
            h = self.core_block(h, attention_mask)
            self.core_block.clear_all_lora()
        return h

    def _forward_fixed_depth(
        self,
        h: Tensor,
        depth: int,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with fixed recursion depth (phases 1 and 2).

        Args:
            h: Hidden states after embedding. Shape [batch, seq, d_model].
            depth: Number of recursion steps.
            attention_mask: Optional attention mask.

        Returns:
            Hidden states after all recursion steps. Shape [batch, seq, d_model].
        """
        for step in range(depth):
            deltas = None
            if self._phase >= 2:
                halt_conf = torch.zeros(h.shape[0], 1, device=h.device)
                deltas = self.controller(h, step, halt_conf)

            h = self._step_forward(h, attention_mask, deltas)

        return h

    def _forward_adaptive(
        self,
        h: Tensor,
        max_steps: int,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass with adaptive recursion depth via ACT (phase 3).

        Args:
            h: Hidden states after embedding. Shape [batch, seq, d_model].
            max_steps: Maximum recursion steps allowed.
            attention_mask: Optional attention mask.

        Returns:
            Tuple of:
                h: ACT-weighted hidden states. Shape [batch, seq, d_model].
                act_loss: Scalar ACT loss tensor.
        """
        batch, seq_len, d_model = h.shape
        act = ACTManager(batch, seq_len, d_model, h.device, h.dtype)

        for step in range(max_steps):
            halt_conf = act.cumulative_halt.mean(dim=1, keepdim=True)  # [batch, 1]
            deltas = self.controller(h, step, halt_conf)

            h = self._step_forward(h, attention_mask, deltas)

            halt_prob = self.halter(h)  # [batch, seq]
            all_halted, running_mask = act.step(h, halt_prob)
            if all_halted:
                break

        h = act.finalize(h)
        return h, act.act_loss()

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        return_dict: bool = True,
    ) -> dict[str, Tensor] | tuple[Tensor, ...]:
        """Full forward pass.

        Args:
            input_ids: Token IDs. Shape [batch, seq_len].
            labels: Target token IDs for loss. Shape [batch, seq_len].
            attention_mask: Optional attention mask.
            return_dict: Whether to return a dict (default) or tuple.

        Returns:
            Dict with keys: logits, loss (if labels), act_loss (phase 3), mean_depth (phase 3).
        """
        # Embed tokens
        h = self.embed_tokens(input_ids)  # [batch, seq, d_model]

        # Recursive computation
        act_loss = torch.tensor(0.0, device=h.device)
        mean_depth = 0.0

        if self._phase <= 2:
            depth = self._get_recursion_depth()
            h = self._forward_fixed_depth(h, depth, attention_mask)
            mean_depth = float(depth)
        else:
            max_steps = self.config.training.max_recursion_steps
            h, act_loss = self._forward_adaptive(h, max_steps, attention_mask)

        # Output head
        h = self.norm(h)
        logits = self.lm_head(h)  # [batch, seq, vocab]

        # Compute loss
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Add ACT loss in phase 3
            if self._phase == 3:
                loss = loss + self.config.halter.act_loss_weight * act_loss

        result = {"logits": logits}
        if loss is not None:
            result["loss"] = loss
        result["act_loss"] = act_loss
        result["mean_depth"] = mean_depth

        if return_dict:
            return result
        return tuple(result.values())

    def param_summary(self) -> dict[str, str]:
        """Return human-readable parameter count by component."""
        counts = count_parameters(self, requires_grad_only=False)
        return {k: format_params(v) for k, v in counts.items()}

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> Tensor:
        """Generate tokens with top-p (nucleus) sampling.

        Args:
            input_ids: Prompt token IDs. Shape [batch, prompt_len].
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability threshold.
            top_k: Top-k filtering.

        Returns:
            Generated token IDs including prompt. Shape [batch, prompt_len + generated].
        """
        was_training = self.training
        self.train(False)
        generated = input_ids

        for _ in range(max_new_tokens):
            # Truncate to max sequence length
            max_len = self.config.core_block.max_seq_len
            context = generated[:, -max_len:]

            # Forward pass
            outputs = self.forward(context)
            logits = outputs["logits"][:, -1, :]  # [batch, vocab]

            # Temperature scaling
            if temperature > 0:
                logits = logits / temperature
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                continue

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove,
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            generated = torch.cat([generated, next_token], dim=-1)

        self.train(was_training)
        return generated
