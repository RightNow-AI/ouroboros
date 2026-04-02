"""Shared test fixtures for Ouroboros tests."""

import pytest
import torch

from ouroboros.config import OuroborosConfig


@pytest.fixture
def tiny_config() -> OuroborosConfig:
    """Tiny config for fast CPU testing."""
    return OuroborosConfig.tiny()


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def seq_len(tiny_config: OuroborosConfig) -> int:
    return tiny_config.training.max_seq_len  # 64 for tiny


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")
