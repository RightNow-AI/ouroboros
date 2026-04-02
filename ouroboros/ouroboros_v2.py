"""Ouroboros v2 — Paper-ready architecture.

Combines proven techniques from literature with our novel SGRC Controller:

From Huginn (2025): Prelude/Recurrent/Coda split
From "Thinking Deeper" (2026): Gated recurrence with identity bias
From RingFormer (2025): Per-step unique LayerNorm
From Relaxed Recursive Transformers (2024): SVD-initialized per-step LoRA
NOVEL: Controller hypernetwork generates LoRA dynamically from hidden state

Architecture:
    Prelude (8 frozen layers) → [GATED RECURSIVE BLOCK with SGRC] → Coda (8 frozen layers)
    The recursive block loops 1 pretrained layer N times.
    Each step: Controller sees hidden state → generates unique LoRA → applies via gated residual.
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ouroboros.dynamic_lora import DynamicLoRALinear
from ouroboros.halter import Halter, ACTManager
from ouroboros.utils import get_logger, format_params

logger = get_logger(__name__)


# ============================================================
# Stability Components
# ============================================================

class RecurrenceGate(nn.Module):
    """h_out = gate * h_new + (1-gate) * h_old. Bias -2 → 88% retain."""

    def __init__(self, d_model: int, bias_init: float = -2.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, bias_init)

    def forward(self, h_new: Tensor, h_old: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate_proj(torch.cat([h_new, h_old], dim=-1)))
        return gate * h_new + (1 - gate) * h_old


class PerStepLayerNorm(nn.Module):
    """Each recurrence step gets unique RMSNorm parameters."""

    def __init__(self, d_model: int, max_steps: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(max_steps, d_model))
        self.eps = eps
        self.max_steps = max_steps

    def forward(self, x: Tensor, step: int) -> Tensor:
        idx = min(step, self.max_steps - 1)
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weights[idx]


class LayerScale(nn.Module):
    """Per-channel scaling initialized to small value."""

    def __init__(self, d_model: int, init_value: float = 1e-4) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model) * init_value)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


# ============================================================
# Compact Controller (Zhyper-style diagonal modulation)
# ============================================================

class CompactController(nn.Module):
    """Generates per-step LoRA modulation from hidden state.

    Instead of generating full LoRA A/B matrices, generates a diagonal
    scaling vector per target. This is more parameter-efficient and
    trains faster than the full Controller.

    For each target: delta_W = A_fixed @ diag(controller_output) @ B_fixed
    Where A_fixed and B_fixed are initialized from SVD of removed layers.
    """

    def __init__(
        self,
        d_model: int,
        lora_rank: int,
        target_names: list[str],
        max_steps: int = 64,
        style_dim: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.lora_rank = lora_rank
        self.target_names = target_names

        # Input: mean-pooled hidden state + step embedding
        self.state_proj = nn.Sequential(
            nn.Linear(d_model, style_dim * 2),
            nn.SiLU(),
        )
        self.step_embed = nn.Embedding(max_steps, style_dim)

        # Style network → per-target diagonal scaling
        self.style_net = nn.Sequential(
            nn.Linear(style_dim * 3, style_dim * 2),
            nn.SiLU(),
            nn.Linear(style_dim * 2, style_dim),
            nn.SiLU(),
        )

        # Per-target diagonal projection (outputs rank-dimensional vector)
        self.target_heads = nn.ModuleDict({
            name: nn.Linear(style_dim, lora_rank, bias=False)
            for name in target_names
        })

        # Zero-init the target heads so Controller starts neutral
        for head in self.target_heads.values():
            nn.init.zeros_(head.weight)

    def forward(
        self,
        hidden_states: Tensor,
        step: int,
        halt_confidence: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Generate per-target diagonal scaling vectors.

        Returns: dict mapping target_name → diagonal vector [batch, rank]
        """
        batch = hidden_states.shape[0]
        state = hidden_states.mean(dim=1)  # [batch, d_model]
        state_proj = self.state_proj(state)  # [batch, style_dim*2]

        max_steps = self.step_embed.num_embeddings
        step_idx = torch.tensor(min(step, max_steps - 1), device=hidden_states.device)
        step_emb = self.step_embed(step_idx).unsqueeze(0).expand(batch, -1)

        combined = torch.cat([state_proj, step_emb], dim=-1)
        style = self.style_net(combined)  # [batch, style_dim]

        diags = {}
        for name, head in self.target_heads.items():
            diags[name] = head(style)  # [batch, rank]

        return diags


# ============================================================
# SVD-Initialized LoRA Module
# ============================================================

class SVDLoRALinear(nn.Module):
    """Linear layer with SVD-initialized fixed LoRA bases + Controller diagonal modulation.

    base_output + scaling * (x @ A_fixed.T @ diag(controller_diag) @ B_fixed.T)
    """

    def __init__(self, base_linear: nn.Module, rank: int, alpha: float = 1.0) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Fixed LoRA bases (initialized from SVD, frozen after init)
        self.lora_A = nn.Parameter(torch.zeros(rank, base_linear.in_features), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, rank), requires_grad=False)

        # Runtime diagonal from Controller
        self._diag: Tensor | None = None

    def init_from_svd(self, residual_weight: Tensor) -> None:
        """Initialize A and B from SVD of a weight residual."""
        U, S, Vh = torch.linalg.svd(residual_weight.float(), full_matrices=False)
        # A = Vh[:rank, :] (right singular vectors)
        # B = U[:, :rank] @ diag(S[:rank])
        r = min(self.rank, S.shape[0])
        self.lora_A.data[:r] = Vh[:r].to(self.lora_A.dtype)
        self.lora_B.data[:, :r] = (U[:, :r] * S[:r].unsqueeze(0)).to(self.lora_B.dtype)

    def set_diag(self, diag: Tensor) -> None:
        """Set the Controller-generated diagonal scaling. Shape [batch, rank]."""
        self._diag = diag

    def clear_diag(self) -> None:
        self._diag = None

    def forward(self, x: Tensor) -> Tensor:
        out = self.base_linear(x)  # [batch, seq, out]

        if self._diag is not None:
            # Cast LoRA bases to match input dtype
            lora_A = self.lora_A.to(dtype=x.dtype)
            lora_B = self.lora_B.to(dtype=x.dtype)
            # x @ A^T → [batch, seq, rank]
            lora_down = F.linear(x, lora_A)
            # Apply diagonal modulation: [batch, seq, rank] * [batch, 1, rank]
            diag = self._diag.unsqueeze(1)  # [batch, 1, rank]
            lora_mod = lora_down * diag
            # @ B^T → [batch, seq, out]
            lora_up = F.linear(lora_mod, lora_B)
            out = out + self.scaling * lora_up

        return out


# ============================================================
# Main Model
# ============================================================

class OuroborosV2(nn.Module):
    """Ouroboros v2: Prelude/Recurrent/Coda with SGRC Controller.

    Takes a pretrained Qwen model and surgically converts it:
    - Prelude (first N layers): frozen, maps tokens to latent space
    - Recurrent block (1 middle layer): looped with SGRC
    - Coda (last N layers): frozen, maps latent to predictions
    """

    def __init__(
        self,
        n_prelude: int = 8,
        n_coda: int = 8,
        recurrent_layer_idx: int = 18,
        lora_rank: int = 32,
        lora_alpha: float = 16.0,
        max_recurrence: int = 64,
        default_depth: int = 8,
        variable_depths: tuple[int, ...] = (4, 8, 16, 32),
        act_loss_weight: float = 0.01,
        halter_hidden: int = 256,
        halter_bias: float = -2.0,
        controller_style_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_prelude = n_prelude
        self.n_coda = n_coda
        self.recurrent_layer_idx = recurrent_layer_idx
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.max_recurrence = max_recurrence
        self.default_depth = default_depth
        self.variable_depths = variable_depths
        self.act_loss_weight = act_loss_weight
        self._phase = 2

        # These get populated by load_base_model()
        self.embed_tokens: nn.Module | None = None
        self.prelude_layers: nn.ModuleList | None = None
        self.recurrent_layer: nn.Module | None = None
        self.coda_layers: nn.ModuleList | None = None
        self.final_norm: nn.Module | None = None
        self.lm_head: nn.Module | None = None
        self.rotary_emb: nn.Module | None = None

        # SGRC components (created after loading base model)
        self.controller: CompactController | None = None
        self.gate: RecurrenceGate | None = None
        self.step_norm: PerStepLayerNorm | None = None
        self.attn_scale: LayerScale | None = None
        self.ffn_scale: LayerScale | None = None
        self.halter: Halter | None = None

        # SVD LoRA modules on recurrent layer
        self._svd_lora_modules: dict[str, SVDLoRALinear] = {}

        # Store config for Halter creation
        self._halter_hidden = halter_hidden
        self._halter_bias = halter_bias
        self._controller_style_dim = controller_style_dim

    def load_base_model(
        self,
        model_name: str = "Qwen/Qwen2.5-3B",
        token: str | None = None,
    ) -> None:
        """Load pretrained model and set up Prelude/Recurrent/Coda split."""
        from transformers import AutoModelForCausalLM

        logger.info(f"Loading {model_name}...")
        pretrained = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, token=token,
        )
        base = pretrained.model
        total_layers = len(base.layers)
        logger.info(f"Total layers: {total_layers}")

        # Extract components
        self.embed_tokens = base.embed_tokens
        self.rotary_emb = base.rotary_emb
        self.final_norm = base.norm
        self.lm_head = pretrained.lm_head

        # Split layers
        self.prelude_layers = nn.ModuleList(
            [base.layers[i] for i in range(self.n_prelude)]
        )
        self.recurrent_layer = base.layers[self.recurrent_layer_idx]
        self.coda_layers = nn.ModuleList(
            [base.layers[i] for i in range(total_layers - self.n_coda, total_layers)]
        )

        # Freeze everything
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.prelude_layers.parameters():
            param.requires_grad = False
        for param in self.recurrent_layer.parameters():
            param.requires_grad = False
        for param in self.coda_layers.parameters():
            param.requires_grad = False
        for param in self.final_norm.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False
        if self.rotary_emb is not None:
            for param in self.rotary_emb.parameters():
                param.requires_grad = False

        d_model = base.config.hidden_size

        # Wrap recurrent layer projections with SVDLoRALinear
        target_names = []
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            original = getattr(self.recurrent_layer.self_attn, proj_name)
            wrapped = SVDLoRALinear(original, self.lora_rank, self.lora_alpha)
            setattr(self.recurrent_layer.self_attn, proj_name, wrapped)
            self._svd_lora_modules[proj_name] = wrapped
            target_names.append(proj_name)

        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            original = getattr(self.recurrent_layer.mlp, proj_name)
            wrapped = SVDLoRALinear(original, self.lora_rank, self.lora_alpha)
            setattr(self.recurrent_layer.mlp, proj_name, wrapped)
            self._svd_lora_modules[proj_name] = wrapped
            target_names.append(proj_name)

        # Initialize SVD LoRA from removed layer residuals
        logger.info("Computing SVD initialization from removed layers...")
        removed_indices = list(range(self.n_prelude, total_layers - self.n_coda))
        removed_indices.remove(self.recurrent_layer_idx)

        if removed_indices:
            self._init_svd_from_residuals(base.layers, removed_indices)

        # Create SGRC components
        model_dtype = next(self.embed_tokens.parameters()).dtype

        self.controller = CompactController(
            d_model=d_model,
            lora_rank=self.lora_rank,
            target_names=target_names,
            max_steps=self.max_recurrence,
            style_dim=self._controller_style_dim,
        ).to(dtype=model_dtype)

        self.gate = RecurrenceGate(d_model).to(dtype=model_dtype)
        self.step_norm = PerStepLayerNorm(d_model, self.max_recurrence).to(dtype=model_dtype)
        self.attn_scale = LayerScale(d_model).to(dtype=model_dtype)
        self.ffn_scale = LayerScale(d_model).to(dtype=model_dtype)

        # Create Halter using a simple config-like approach
        self.halter = nn.Sequential(
            nn.Linear(d_model, self._halter_hidden),
            nn.SiLU(),
            nn.Linear(self._halter_hidden, self._halter_hidden),
            nn.SiLU(),
            nn.Linear(self._halter_hidden, 1),
        ).to(dtype=model_dtype)
        # Bias init for halter
        nn.init.zeros_(self.halter[-1].weight)
        nn.init.constant_(self.halter[-1].bias, self._halter_bias)

        # Free the unused layers
        del pretrained
        unused_indices = set(range(total_layers)) - set(range(self.n_prelude)) - {self.recurrent_layer_idx} - set(range(total_layers - self.n_coda, total_layers))
        # These are already freed since we only kept references to the ones we need

        # Log stats
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Prelude: {self.n_prelude} layers, Recurrent: layer {self.recurrent_layer_idx}, Coda: {self.n_coda} layers")
        logger.info(f"Frozen: {format_params(frozen_params)}, Trainable: {format_params(trainable_params)}")
        logger.info(f"SVD LoRA targets: {list(self._svd_lora_modules.keys())}")

    def _init_svd_from_residuals(self, all_layers: nn.ModuleList, removed_indices: list[int]) -> None:
        """Initialize SVD LoRA bases from average residual of removed layers."""
        recurrent_sd = {}
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            recurrent_sd[name] = getattr(self.recurrent_layer.self_attn, name).base_linear.weight.data
        for name in ["gate_proj", "up_proj", "down_proj"]:
            recurrent_sd[name] = getattr(self.recurrent_layer.mlp, name).base_linear.weight.data

        for proj_name, svd_module in self._svd_lora_modules.items():
            # Average residual: mean(removed_layer_weight - recurrent_weight)
            residuals = []
            for idx in removed_indices:
                layer = all_layers[idx]
                if proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    w = getattr(layer.self_attn, proj_name).weight.data
                else:
                    w = getattr(layer.mlp, proj_name).weight.data
                residuals.append(w - recurrent_sd[proj_name])

            avg_residual = torch.stack(residuals).mean(dim=0)
            svd_module.init_from_svd(avg_residual)
            logger.info(f"  SVD init {proj_name}: residual norm={avg_residual.norm():.2f}")

    def set_phase(self, phase: int) -> None:
        assert phase in (2, 3)
        self._phase = phase
        if self.controller:
            self.controller.requires_grad_(True)
        if self.gate:
            self.gate.requires_grad_(True)
        if self.step_norm:
            self.step_norm.requires_grad_(True)
        if self.attn_scale:
            self.attn_scale.requires_grad_(True)
        if self.ffn_scale:
            self.ffn_scale.requires_grad_(True)
        if self.halter:
            self.halter.requires_grad_(phase >= 3)
        logger.info(f"Phase {phase}: {'adaptive depth' if phase == 3 else 'fixed depth'}")

    def _get_pos_emb(self, h: Tensor, seq_len: int) -> tuple[Tensor, Tensor]:
        pos_ids = torch.arange(seq_len, device=h.device).unsqueeze(0)
        return self.rotary_emb(h, pos_ids)

    def _run_recurrent_step(self, h: Tensor, pos_emb: tuple, step: int) -> Tensor:
        """One recurrence step: Controller → set diag → layer forward → gate."""
        # Controller generates diagonal modulation
        diags = self.controller(h, step)
        for name, svd_mod in self._svd_lora_modules.items():
            svd_mod.set_diag(diags[name])

        # Run layer
        outputs = self.recurrent_layer(h, position_embeddings=pos_emb)
        h_new = outputs[0] if isinstance(outputs, tuple) else outputs

        # Clear diags
        for svd_mod in self._svd_lora_modules.values():
            svd_mod.clear_diag()

        # Per-step norm
        h_new = self.step_norm(h_new, step)

        # Gated recurrence
        h = self.gate(h_new, h)
        return h

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        batch, seq_len = input_ids.shape

        # Embed
        h = self.embed_tokens(input_ids)
        pos_emb = self._get_pos_emb(h, seq_len)

        # Prelude (frozen)
        for layer in self.prelude_layers:
            out = layer(h, position_embeddings=pos_emb)
            h = out[0] if isinstance(out, tuple) else out

        # Recurrent block with SGRC
        act_loss = torch.tensor(0.0, device=h.device)

        if self._phase == 2:
            depth = random.choice(self.variable_depths)
            for step in range(depth):
                h = self._run_recurrent_step(h, pos_emb, step)
        else:
            # Adaptive depth with ACT
            d_model = h.shape[-1]
            act = ACTManager(batch, seq_len, d_model, h.device, h.dtype)

            for step in range(self.max_recurrence):
                h = self._run_recurrent_step(h, pos_emb, step)

                halt_logits = self.halter(h).squeeze(-1)
                halt_prob = torch.sigmoid(halt_logits)
                all_halted, _ = act.step(h, halt_prob)
                if all_halted:
                    break

            h = act.finalize(h)
            act_loss = act.act_loss()

        # Coda (frozen)
        for layer in self.coda_layers:
            out = layer(h, position_embeddings=pos_emb)
            h = out[0] if isinstance(out, tuple) else out

        # Output
        h = self.final_norm(h)
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
                loss = loss + self.act_loss_weight * act_loss

        return {"logits": logits, "loss": loss, "act_loss": act_loss}

    @torch.no_grad()
    def forward_base_only(self, input_ids: Tensor, labels: Tensor | None = None) -> dict[str, Any]:
        """Run only Prelude + 1x Recurrent (no SGRC) + Coda. For baseline comparison."""
        batch, seq_len = input_ids.shape
        h = self.embed_tokens(input_ids)
        pos_emb = self._get_pos_emb(h, seq_len)

        for layer in self.prelude_layers:
            out = layer(h, position_embeddings=pos_emb)
            h = out[0] if isinstance(out, tuple) else out

        # Single pass through recurrent layer (no Controller)
        for mod in self._svd_lora_modules.values():
            mod.clear_diag()
        out = self.recurrent_layer(h, position_embeddings=pos_emb)
        h = out[0] if isinstance(out, tuple) else out

        for layer in self.coda_layers:
            out = layer(h, position_embeddings=pos_emb)
            h = out[0] if isinstance(out, tuple) else out

        h = self.final_norm(h)
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
