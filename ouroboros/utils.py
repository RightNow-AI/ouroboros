"""Utility functions for Ouroboros."""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module, requires_grad_only: bool = True) -> dict[str, int]:
    """Count parameters by top-level module.

    Args:
        model: The model to count parameters for.
        requires_grad_only: If True, only count trainable parameters.

    Returns:
        Dict mapping module name to parameter count.
    """
    counts: dict[str, int] = {}
    for name, module in model.named_children():
        n = sum(
            p.numel() for p in module.parameters()
            if not requires_grad_only or p.requires_grad
        )
        counts[name] = n
    counts["total"] = sum(counts.values())
    return counts


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger with rich-compatible formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def format_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
