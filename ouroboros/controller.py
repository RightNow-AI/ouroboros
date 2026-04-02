"""Controller — the novel hypernetwork that generates per-step LoRA deltas.

This is the core innovation of Ouroboros. The Controller observes the model's
current hidden state and dynamically generates unique weight modifications for
the next recursion step. No two forward passes through Ouroboros are the same
computation, even on the same input at different recursion depths.

Architecture:
    1. State summary (mean-pooled hidden states) is projected to 512-dim
    2. Step embedding (learned, 256-dim) encodes recursion depth
    3. Halt confidence (32-dim) provides feedback from adaptive computation
    4. These are concatenated and fed through a style network → 64-dim style vector
    5. Per-target ControllerHeads use factored basis matrices to expand the style
       vector into full LoRA A/B matrices for each of the 7 linear layers

Key design choices:
    - Factored basis: basis_A = basis_A1 @ basis_A2 (compressed)
    - Zero initialization on coefficient projections so model starts as plain
      recursive transformer and Controller influence emerges gradually
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from ouroboros.config import OuroborosConfig


class ControllerHead(nn.Module):
    """Generates LoRA A/B matrices for a single linear layer target.

    Uses factored basis matrices with learned coefficients from the style vector.
    The coefficient projections are zero-initialized so the Controller starts
    with zero influence.

    Args:
        style_dim: Dimension of the input style vector.
        rank: LoRA rank.
        in_features: Input dimension of the target linear layer.
        out_features: Output dimension of the target linear layer.
        n_basis: Number of basis vectors for the linear combination.
        bottleneck_divisor: Compression factor for basis factorization.
    """

    def __init__(
        self,
        style_dim: int,
        rank: int,
        in_features: int,
        out_features: int,
        n_basis: int,
        bottleneck_divisor: int = 4,
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.n_basis = n_basis

        # Coefficient projections: style_vector → basis coefficients
        # CRITICAL: Zero-initialized so initial LoRA deltas = 0
        self.coeff_proj_A = nn.Linear(style_dim, n_basis, bias=False)
        self.coeff_proj_B = nn.Linear(style_dim, n_basis, bias=False)

        # Factored basis matrices for A: basis_A = basis_A1 @ basis_A2
        # Full basis_A shape would be [n_basis, rank, in_features]
        # Factored: [n_basis, rank, bottleneck] @ [n_basis, bottleneck, in_features]
        bottleneck_A = max(1, in_features // bottleneck_divisor)
        self.basis_A1 = nn.Parameter(torch.empty(n_basis, rank, bottleneck_A))
        self.basis_A2 = nn.Parameter(torch.empty(n_basis, bottleneck_A, in_features))

        # Factored basis matrices for B: basis_B = basis_B1 @ basis_B2
        # Full basis_B shape would be [n_basis, out_features, rank]
        # Factored: [n_basis, out_features, bottleneck] @ [n_basis, bottleneck, rank]
        bottleneck_B = max(1, out_features // bottleneck_divisor)
        self.basis_B1 = nn.Parameter(torch.empty(n_basis, out_features, bottleneck_B))
        self.basis_B2 = nn.Parameter(torch.empty(n_basis, bottleneck_B, rank))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights.

        Coefficient projections are zero-initialized (CRITICAL for Ouroboros).
        Basis matrices use Kaiming uniform for stable starting point.
        """
        # Zero-init coefficient projections → initial LoRA deltas = 0
        nn.init.zeros_(self.coeff_proj_A.weight)
        nn.init.zeros_(self.coeff_proj_B.weight)

        # Kaiming init for basis matrices
        for basis in [self.basis_A1, self.basis_A2, self.basis_B1, self.basis_B2]:
            fan_in = basis.shape[-1]
            std = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(basis, -std, std)

    def forward(self, style: Tensor) -> tuple[Tensor, Tensor]:
        """Generate LoRA A and B matrices from a style vector.

        Args:
            style: Style vector from Controller. Shape [batch, style_dim].

        Returns:
            Tuple of:
                lora_A: Shape [batch, rank, in_features].
                lora_B: Shape [batch, out_features, rank].
        """
        # Compute basis coefficients from style vector
        coeff_A = self.coeff_proj_A(style)  # [batch, n_basis]
        coeff_B = self.coeff_proj_B(style)  # [batch, n_basis]

        # Compute full basis matrices via factored product
        # basis_A1: [n_basis, rank, bottleneck], basis_A2: [n_basis, bottleneck, in]
        basis_A = torch.bmm(self.basis_A1, self.basis_A2)  # [n_basis, rank, in_features]
        # basis_B1: [n_basis, out, bottleneck], basis_B2: [n_basis, bottleneck, rank]
        basis_B = torch.bmm(self.basis_B1, self.basis_B2)  # [n_basis, out_features, rank]

        # Linear combination: coefficients × basis matrices
        # coeff_A: [batch, n_basis], basis_A: [n_basis, rank, in]
        lora_A = torch.einsum("bn,nri->bri", coeff_A, basis_A)  # [batch, rank, in_features]
        # coeff_B: [batch, n_basis], basis_B: [n_basis, out, rank]
        lora_B = torch.einsum("bn,nor->bor", coeff_B, basis_B)  # [batch, out_features, rank]

        return lora_A, lora_B


class Controller(nn.Module):
    """The Controller hypernetwork — Ouroboros's novel component.

    Takes the model's current hidden state, recursion step, and halt confidence,
    and generates unique LoRA weight modifications for all 7 linear layers in
    the CoreBlock.

    The Controller enables compositional reasoning by making each recursion step
    perform a dynamically different computation based on what has been computed so far.
    """

    def __init__(self, config: OuroborosConfig) -> None:
        super().__init__()
        cc = config.controller
        d_model = config.core_block.d_model

        # Input projections
        self.state_proj = nn.Sequential(
            nn.Linear(d_model, cc.state_proj_dim),
            nn.SiLU(),
        )
        self.step_embed = nn.Embedding(cc.max_steps, cc.step_embed_dim)
        self.halt_proj = nn.Sequential(
            nn.Linear(1, cc.halt_proj_dim),
        )

        # Style network: projects concatenated inputs to compact style vector
        # Input: state_proj_dim + step_embed_dim + halt_proj_dim
        self.style_net = nn.Sequential(
            nn.Linear(cc.input_dim, cc.style_hidden_dim),
            nn.SiLU(),
            nn.Linear(cc.style_hidden_dim, cc.style_hidden_dim),
            nn.SiLU(),
            nn.Linear(cc.style_hidden_dim, cc.style_dim),
        )

        # One ControllerHead per LoRA target (7 total)
        target_dims = config.target_dims()
        self.heads = nn.ModuleDict()
        for name in config.lora.TARGET_NAMES:
            in_features, out_features = target_dims[name]
            self.heads[name] = ControllerHead(
                style_dim=cc.style_dim,
                rank=config.lora.rank,
                in_features=in_features,
                out_features=out_features,
                n_basis=cc.n_basis,
                bottleneck_divisor=cc.bottleneck_divisor,
            )

    def forward(
        self,
        hidden_states: Tensor,
        step: int,
        halt_confidence: Tensor,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        """Generate LoRA deltas for all CoreBlock linear layers.

        Args:
            hidden_states: Current hidden states. Shape [batch, seq_len, d_model].
            step: Current recursion step index (0-indexed).
            halt_confidence: Mean halt confidence across tokens.
                Shape [batch, 1] or scalar tensor.

        Returns:
            Dict mapping target name to (lora_A, lora_B) tuple.
        """
        batch = hidden_states.shape[0]

        # 1. State summary: mean-pool across sequence dimension
        state_summary = hidden_states.mean(dim=1)  # [batch, d_model]
        state_proj = self.state_proj(state_summary)  # [batch, state_proj_dim]

        # 2. Step embedding (clamped to max_steps range for safety)
        max_steps = self.step_embed.num_embeddings
        step_clamped = min(step, max_steps - 1)
        step_idx = torch.tensor(step_clamped, device=hidden_states.device, dtype=torch.long)
        step_emb = self.step_embed(step_idx).unsqueeze(0).expand(batch, -1)  # [batch, step_embed_dim]

        # 3. Halt confidence projection
        if halt_confidence.dim() == 0:
            halt_confidence = halt_confidence.unsqueeze(0).unsqueeze(0).expand(batch, 1)
        elif halt_confidence.dim() == 1:
            halt_confidence = halt_confidence.unsqueeze(1)
        halt_proj = self.halt_proj(halt_confidence.to(dtype=hidden_states.dtype))  # [batch, halt_proj_dim]

        # 4. Concatenate and compute style vector
        combined = torch.cat([state_proj, step_emb, halt_proj], dim=-1)  # [batch, input_dim]
        style = self.style_net(combined)  # [batch, style_dim]

        # 5. Generate LoRA deltas for each target
        deltas = {}
        for name, head in self.heads.items():
            lora_A, lora_B = head(style)
            deltas[name] = (lora_A, lora_B)

        return deltas
