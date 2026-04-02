"""Reinforcement learning training for Ouroboros Phase 3+.

GRPO (Group Relative Policy Optimization) via TRL for:
    - Math reasoning (GSM8K/MATH format extraction + correctness)
    - Code generation (test case passing)
    - Combined with ACT loss for efficient computation
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final numerical answer from GSM8K format.

    Expected format: ... #### <number>
    """
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    return None


def extract_boxed_answer(text: str) -> str | None:
    r"""Extract answer from \boxed{...} format (MATH dataset)."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


def math_reward_fn(
    completions: list[str],
    references: list[str],
) -> list[float]:
    """Compute rewards for math completions.

    Args:
        completions: Model-generated solutions.
        references: Ground truth answers.

    Returns:
        List of reward scores (0.0 or 1.0).
    """
    rewards = []
    for completion, reference in zip(completions, references):
        # Try GSM8K format first, then MATH format
        pred = extract_gsm8k_answer(completion) or extract_boxed_answer(completion)
        ref = extract_gsm8k_answer(reference) or extract_boxed_answer(reference) or reference

        if pred is not None and ref is not None:
            try:
                rewards.append(1.0 if float(pred) == float(ref) else 0.0)
            except ValueError:
                rewards.append(1.0 if pred.strip() == ref.strip() else 0.0)
        else:
            rewards.append(0.0)

    return rewards


def create_grpo_trainer(
    model: Any,
    tokenizer: Any,
    config: Any,
    reward_fn: Any = math_reward_fn,
) -> Any:
    """Create a GRPO trainer for Ouroboros.

    Args:
        model: The Ouroboros model.
        tokenizer: The tokenizer.
        config: Ouroboros config.
        reward_fn: Reward function for evaluating completions.

    Returns:
        Configured TRL trainer.
    """
    from trl import GRPOConfig, GRPOTrainer

    tc = config.training

    grpo_config = GRPOConfig(
        output_dir=f"{tc.checkpoint_dir}/grpo",
        per_device_train_batch_size=tc.batch_size_per_gpu,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.phase3_lr,
        max_steps=5000,
        logging_steps=tc.log_every,
        save_steps=tc.save_every,
        bf16=tc.bf16,
        max_grad_norm=tc.max_grad_norm,
        num_generations=4,  # Group size for GRPO
        report_to="wandb" if tc.wandb_project else "none",
    )

    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
    )

    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GRPO RL training for Ouroboros")
    parser.add_argument("--checkpoint", required=True, help="Phase 3 checkpoint path")
    parser.add_argument("--config", default="configs/ouroboros_config.yaml")
    args = parser.parse_args()

    from ouroboros.config import OuroborosConfig
    from ouroboros.convert import load_ouroboros
    from transformers import AutoTokenizer

    config = OuroborosConfig.from_yaml(args.config)
    model, _ = load_ouroboros(args.checkpoint)
    model.set_phase(3)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = create_grpo_trainer(model, tokenizer, config)
    trainer.train()
