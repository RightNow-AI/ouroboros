"""Halter and ACTManager — adaptive computation time for Ouroboros.

Implements the Adaptive Computation Time (ACT) mechanism from Graves (2016)
to allow per-token dynamic recursion depth. Easy tokens halt early, hard
tokens use more recursion steps.

Key mechanism:
    - Each token accumulates halt probabilities across steps
    - When cumulative probability >= 1.0, the token stops updating
    - The "remainder" mechanism ensures smooth gradients: the final step's
      weight is 1 - previous_cumulative, making all weights sum to exactly 1.0
    - ACT loss = mean ponder cost, penalizing unnecessary computation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ouroboros.config import OuroborosConfig


class Halter(nn.Module):
    """Per-token halt probability predictor.

    3-layer MLP: d_model → hidden → hidden → 1 → sigmoid.
    Bias initialized to -2.0 so initial halt prob ≈ 0.12 (tokens halt ~step 8).
    """

    def __init__(self, config: OuroborosConfig) -> None:
        super().__init__()
        d_model = config.core_block.d_model
        hidden = config.halter.hidden_dim
        bias_init = config.halter.halt_bias_init

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        # Initialize final layer bias to halt_bias_init
        # sigmoid(-2.0) ≈ 0.12 → tokens accumulate slowly, halting around step 8
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, bias_init)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute per-token halt probability.

        Args:
            hidden_states: Shape [batch, seq_len, d_model].

        Returns:
            Halt probabilities. Shape [batch, seq_len], values in (0, 1).
        """
        logits = self.net(hidden_states).squeeze(-1)  # [batch, seq_len]
        return torch.sigmoid(logits)


class ACTManager:
    """Stateful manager for Adaptive Computation Time across recursion steps.

    Tracks per-token halt state, accumulates weighted outputs, and computes
    the ACT loss (ponder cost) for training.

    Uses Graves (2016) remainder mechanism:
        - At each step, if cumulative_halt + halt_prob >= 1.0, the token halts
        - The halting step's weight is remainder = 1.0 - previous_cumulative
        - Non-halting steps use halt_prob as their weight
        - This ensures all weights sum to exactly 1.0 per token

    Args:
        batch_size: Batch dimension.
        seq_len: Sequence length.
        d_model: Hidden state dimension.
        device: Torch device.
        dtype: Torch dtype for accumulated output.
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model

        # Per-token state
        self.cumulative_halt = torch.zeros(
            batch_size, seq_len, device=device, dtype=torch.float32,
        )  # [batch, seq]
        self.halted = torch.zeros(
            batch_size, seq_len, device=device, dtype=torch.bool,
        )  # [batch, seq]
        self.accumulated_output = torch.zeros(
            batch_size, seq_len, d_model, device=device, dtype=dtype,
        )  # [batch, seq, d_model]

        # For ACT loss
        self.ponder_cost = torch.zeros(
            batch_size, seq_len, device=device, dtype=torch.float32,
        )  # [batch, seq]
        self.n_steps = torch.zeros(
            batch_size, seq_len, device=device, dtype=torch.long,
        )  # [batch, seq]

    def step(
        self,
        hidden_states: Tensor,
        halt_prob: Tensor,
    ) -> tuple[bool, Tensor]:
        """Process one recursion step of ACT.

        Args:
            hidden_states: Current hidden states. Shape [batch, seq, d_model].
            halt_prob: Per-token halt probability. Shape [batch, seq].

        Returns:
            Tuple of:
                all_halted: Whether all tokens have halted.
                running_mask: Boolean mask of still-running tokens. Shape [batch, seq].
        """
        # Tokens that are still running (not yet halted)
        running = ~self.halted  # [batch, seq]

        # Check which running tokens will halt this step
        will_halt = running & ((self.cumulative_halt + halt_prob) >= 1.0)

        # Remainder for halting tokens: weight = 1 - previous_cumulative
        remainder = 1.0 - self.cumulative_halt  # [batch, seq]

        # Compute weight for this step
        # Halting tokens: use remainder. Running tokens: use halt_prob. Already halted: 0.
        weight = torch.where(
            will_halt,
            remainder,
            torch.where(running, halt_prob, torch.zeros_like(halt_prob)),
        )  # [batch, seq]

        # Accumulate weighted output
        self.accumulated_output += weight.unsqueeze(-1) * hidden_states.float()

        # Update cumulative halt (only for running, non-halting tokens)
        self.cumulative_halt = torch.where(
            will_halt,
            torch.ones_like(self.cumulative_halt),
            torch.where(running, self.cumulative_halt + halt_prob, self.cumulative_halt),
        )

        # Update halted mask
        self.halted = self.halted | will_halt

        # Update step counter and ponder cost for running tokens
        self.n_steps += running.long()
        self.ponder_cost += torch.where(running, halt_prob, torch.zeros_like(halt_prob))

        return self.halted.all().item(), running

    def finalize(self, hidden_states: Tensor) -> Tensor:
        """Finalize ACT: handle tokens that never halted within max steps.

        For tokens still running, treat the final step as forced halt
        (remainder = 1 - cumulative_halt).

        Args:
            hidden_states: Final hidden states. Shape [batch, seq, d_model].

        Returns:
            ACT-weighted output. Shape [batch, seq, d_model].
        """
        # Force-halt any remaining tokens
        still_running = ~self.halted
        if still_running.any():
            remainder = 1.0 - self.cumulative_halt
            weight = torch.where(
                still_running, remainder, torch.zeros_like(remainder),
            )
            self.accumulated_output += weight.unsqueeze(-1) * hidden_states.float()
            self.halted.fill_(True)

        return self.accumulated_output

    def act_loss(self) -> Tensor:
        """Compute ACT loss (mean ponder cost).

        This penalizes the model for using more computation than necessary.
        Minimizing this encourages the model to halt early when possible.

        Returns:
            Scalar loss tensor.
        """
        return self.ponder_cost.mean()

    def mean_steps(self) -> float:
        """Return the mean number of recursion steps used across all tokens."""
        return self.n_steps.float().mean().item()
