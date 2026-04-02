"""Recursion depth analysis for Ouroboros.

Analyzes how the model allocates computation (recursion depth) across
different types of inputs. Easy text should use fewer steps, hard
reasoning should use more.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ouroboros.config import OuroborosConfig
from ouroboros.model import OuroborosModel
from ouroboros.halter import ACTManager
from ouroboros.utils import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def analyze_depth(
    model: OuroborosModel,
    tokenizer: Any,
    prompts: list[str],
    max_steps: int = 64,
) -> list[dict[str, Any]]:
    """Analyze per-token recursion depth for a list of prompts.

    Args:
        model: Ouroboros model in phase 3 (adaptive depth).
        tokenizer: Tokenizer.
        prompts: List of text prompts.
        max_steps: Maximum recursion steps.

    Returns:
        List of analysis dicts per prompt.
    """
    model_was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    results = []

    for prompt in prompts:
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device)
        batch, seq_len = input_ids.shape

        h = model.embed_tokens(input_ids)
        d_model = h.shape[-1]

        act = ACTManager(batch, seq_len, d_model, device, h.dtype)
        per_step_depth = []

        for step in range(max_steps):
            halt_conf = act.cumulative_halt.mean(dim=1, keepdim=True)
            deltas = model.controller(h, step, halt_conf)
            model.core_block.set_all_lora(deltas)
            h = model.core_block(h)
            model.core_block.clear_all_lora()

            halt_prob = model.halter(h)
            all_halted, _ = act.step(h, halt_prob)

            pct_halted = act.halted.float().mean().item() * 100
            per_step_depth.append({
                "step": step,
                "pct_halted": pct_halted,
                "mean_halt_prob": halt_prob.mean().item(),
            })

            if all_halted:
                break

        n_steps_per_token = act.n_steps.squeeze(0).cpu().tolist()
        token_texts = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu())

        results.append({
            "prompt": prompt,
            "mean_depth": act.mean_steps(),
            "min_depth": min(n_steps_per_token),
            "max_depth": max(n_steps_per_token),
            "per_token": list(zip(token_texts, n_steps_per_token)),
            "per_step": per_step_depth,
        })

        logger.info(
            f"Prompt: {prompt[:50]}... | "
            f"mean_depth={results[-1]['mean_depth']:.1f} | "
            f"range=[{results[-1]['min_depth']}, {results[-1]['max_depth']}]"
        )

    model.train(model_was_training)
    return results


def print_depth_report(results: list[dict]) -> None:
    """Pretty-print depth analysis results."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    for r in results:
        console.print(f"\n[bold cyan]Prompt:[/bold cyan] {r['prompt'][:80]}")
        console.print(
            f"  Mean depth: [green]{r['mean_depth']:.1f}[/green] | "
            f"Range: [{r['min_depth']}, {r['max_depth']}]"
        )

        # Token-level depth table
        table = Table(title="Per-Token Depth", show_lines=False)
        table.add_column("Token", style="cyan")
        table.add_column("Steps", justify="right")
        table.add_column("Bar", min_width=30)

        for token, steps in r["per_token"][:20]:  # Show first 20 tokens
            bar_len = min(steps, 40)
            bar = "#" * bar_len
            table.add_row(token, str(steps), bar)

        if len(r["per_token"]) > 20:
            table.add_row("...", "...", "...")

        console.print(table)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", nargs="+", default=[
        "The cat sat on the mat.",
        "What is the derivative of x^3 + 2x^2 - 5x + 3?",
        "Solve: If 3x + 7 = 22, what is x?",
        "Write a function to find the longest common subsequence of two strings.",
    ])
    args = parser.parse_args()

    from ouroboros.convert import load_ouroboros
    from transformers import AutoTokenizer

    model, config = load_ouroboros(args.checkpoint, device="cuda")
    model.set_phase(3)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    results = analyze_depth(model, tokenizer, args.prompts)
    print_depth_report(results)
