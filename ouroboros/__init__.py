"""Ouroboros: Dynamic Weight Generation for Recursive Transformers.

A Controller hypernetwork generates input-conditioned LoRA modulation
for recursive transformer blocks, making each recurrence step depend
on both the current hidden state and the step index.
"""

from ouroboros.config import OuroborosConfig
from ouroboros.dynamic_lora import DynamicLoRALinear
from ouroboros.core_block import CoreBlock
from ouroboros.controller import Controller, ControllerHead
from ouroboros.halter import Halter, ACTManager
from ouroboros.model import OuroborosModel
from ouroboros.ouroboros_v2 import (
    OuroborosV2,
    CompactController,
    SVDLoRALinear,
    RecurrenceGate,
    PerStepLayerNorm,
    LayerScale,
)
from ouroboros.gated_refiner import OuroborosGatedRefiner

__version__ = "0.1.0"

__all__ = [
    "OuroborosConfig",
    "DynamicLoRALinear",
    "CoreBlock",
    "Controller",
    "ControllerHead",
    "Halter",
    "ACTManager",
    "OuroborosModel",
    "OuroborosV2",
    "CompactController",
    "SVDLoRALinear",
    "RecurrenceGate",
    "PerStepLayerNorm",
    "LayerScale",
    "OuroborosGatedRefiner",
]
