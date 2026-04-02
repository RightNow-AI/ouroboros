"""Data pipeline for Ouroboros training.

Phase 1: General text (SlimPajama/RedPajama) — recover pretrained knowledge
Phase 2: High-quality mix (textbooks, code, math) — awaken Controller
Phase 3: Reasoning datasets (GSM8K-train, MATH, code) — full Ouroboros
"""

from __future__ import annotations

from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

from ouroboros.utils import get_logger

logger = get_logger(__name__)


class StreamingTextDataset(IterableDataset):
    """Streaming dataset from HuggingFace datasets with tokenization.

    Tokenizes text on-the-fly and packs into fixed-length sequences.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        max_seq_len: int = 2048,
        split: str = "train",
        subset: str | None = None,
        streaming: bool = True,
        text_field: str = "text",
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.subset = subset
        self.streaming = streaming
        self.text_field = text_field

    def _load_dataset(self) -> Any:
        from datasets import load_dataset

        kwargs: dict[str, Any] = {
            "split": self.split,
            "streaming": self.streaming,
        }
        if self.subset:
            kwargs["name"] = self.subset

        return load_dataset(self.dataset_name, **kwargs)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        dataset = self._load_dataset()
        buffer: list[int] = []

        for example in dataset:
            text = example.get(self.text_field, "")
            if not text:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            # Yield packed sequences of max_seq_len
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}


class InterleavedDataset(IterableDataset):
    """Interleave multiple streaming datasets with given weights."""

    def __init__(
        self,
        datasets: list[StreamingTextDataset],
        weights: list[float] | None = None,
    ) -> None:
        self.datasets = datasets
        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        import random

        iterators = [iter(d) for d in self.datasets]
        active = list(range(len(iterators)))

        while active:
            # Weighted random choice
            idx = random.choices(active, weights=[self.weights[i] for i in active])[0]
            try:
                yield next(iterators[idx])
            except StopIteration:
                active.remove(idx)
                if not active:
                    break


def get_phase1_dataset(
    tokenizer: Any,
    max_seq_len: int = 2048,
) -> StreamingTextDataset:
    """Phase 1: General text for knowledge recovery."""
    return StreamingTextDataset(
        dataset_name="HuggingFaceFW/fineweb",
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        subset="sample-10BT",
        streaming=True,
    )


def get_phase2_dataset(
    tokenizer: Any,
    max_seq_len: int = 2048,
) -> StreamingTextDataset:
    """Phase 2: High-quality educational text for Controller awakening."""
    return StreamingTextDataset(
        dataset_name="HuggingFaceFW/fineweb-edu",
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        subset="sample-10BT",
        streaming=True,
    )


def get_phase3_dataset(
    tokenizer: Any,
    max_seq_len: int = 2048,
) -> InterleavedDataset:
    """Phase 3: Reasoning datasets for full Ouroboros training."""
    datasets = [
        StreamingTextDataset(
            dataset_name="openai/gsm8k",
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            text_field="question",
            subset="main",
            split="train",
            streaming=False,
        ),
        StreamingTextDataset(
            dataset_name="HuggingFaceFW/fineweb-edu",
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            subset="sample-10BT",
            streaming=True,
        ),
    ]
    return InterleavedDataset(datasets, weights=[0.6, 0.4])


def create_dataloader(
    dataset: IterableDataset,
    batch_size: int = 4,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for streaming datasets."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
