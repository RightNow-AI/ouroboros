"""Evaluation suite for Ouroboros.

Measures:
    - WikiText-2 perplexity
    - lm-eval-harness benchmarks (GSM8K, MATH, ARC, HellaSwag)
    - Recursion depth analysis
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from ouroboros.config import OuroborosConfig
from ouroboros.model import OuroborosModel
from ouroboros.utils import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def compute_perplexity(
    model: OuroborosModel,
    tokenizer: any,
    dataset_name: str = "wikitext",
    subset: str = "wikitext-2-v1",
    split: str = "test",
    max_seq_len: int = 2048,
    batch_size: int = 4,
) -> float:
    """Compute perplexity on a text dataset.

    Args:
        model: The Ouroboros model.
        tokenizer: Tokenizer for encoding text.
        dataset_name: HuggingFace dataset name.
        subset: Dataset subset.
        split: Dataset split.
        max_seq_len: Maximum sequence length.
        batch_size: Evaluation batch size.

    Returns:
        Perplexity score.
    """
    from datasets import load_dataset

    model_was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    dataset = load_dataset(dataset_name, subset, split=split)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    total_loss = 0.0
    total_tokens = 0

    # Process in chunks of max_seq_len
    for i in range(0, len(input_ids) - 1, max_seq_len):
        chunk = input_ids[i : i + max_seq_len + 1].unsqueeze(0).to(device)
        if chunk.shape[1] < 2:
            continue

        ids = chunk[:, :-1]
        labels = chunk[:, 1:]

        outputs = model(ids)
        logits = outputs["logits"]

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    logger.info(f"Perplexity on {dataset_name}/{subset}: {perplexity:.2f}")

    model.train(model_was_training)
    return perplexity


def run_lm_benchmarks(
    model: OuroborosModel,
    tokenizer: any,
    tasks: list[str] | None = None,
    num_fewshot: int = 0,
    batch_size: int = 4,
) -> dict:
    """Run lm-eval-harness benchmarks.

    Args:
        model: The Ouroboros model.
        tokenizer: Tokenizer.
        tasks: List of task names. Defaults to standard reasoning suite.
        num_fewshot: Number of few-shot examples.
        batch_size: Evaluation batch size.

    Returns:
        Dict of task results.
    """
    if tasks is None:
        tasks = ["gsm8k", "arc_challenge", "hellaswag", "mmlu"]

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
        )

        logger.info("Benchmark Results:")
        for task_name, task_result in results["results"].items():
            acc = task_result.get("acc,none", task_result.get("exact_match,none", "N/A"))
            logger.info(f"  {task_name}: {acc}")

        return results
    except ImportError:
        logger.warning("lm-eval not installed. Install with: pip install lm-eval")
        return {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Ouroboros")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--perplexity", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--phase", type=int, default=3)
    args = parser.parse_args()

    from ouroboros.convert import load_ouroboros
    from transformers import AutoTokenizer

    model, config = load_ouroboros(args.checkpoint, device="cuda")
    model.set_phase(args.phase)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    if args.perplexity:
        ppl = compute_perplexity(model, tokenizer, max_seq_len=config.training.max_seq_len)
        print(f"Perplexity: {ppl:.2f}")

    if args.benchmark:
        results = run_lm_benchmarks(model, tokenizer, tasks=args.tasks)
