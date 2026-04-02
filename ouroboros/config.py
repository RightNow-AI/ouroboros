"""Ouroboros configuration system.

All hyperparameters are defined as nested dataclasses with preset configurations
for tiny (testing) and default (Llama-3.2-3B based) model variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# -- Llama-3.2-3B dimensions --
LLAMA_3B_D_MODEL = 3072
LLAMA_3B_N_HEADS = 24
LLAMA_3B_N_KV_HEADS = 8
LLAMA_3B_D_FF = 8192
LLAMA_3B_HEAD_DIM = 128
LLAMA_3B_VOCAB = 128256
LLAMA_3B_N_LAYERS = 28
LLAMA_3B_ROPE_THETA = 500000.0


@dataclass
class CoreBlockConfig:
    """Shared transformer block configuration."""

    d_model: int = LLAMA_3B_D_MODEL
    n_heads: int = LLAMA_3B_N_HEADS
    n_kv_heads: int = LLAMA_3B_N_KV_HEADS
    d_ff: int = LLAMA_3B_D_FF
    head_dim: int = LLAMA_3B_HEAD_DIM
    rms_norm_eps: float = 1e-5
    rope_theta: float = LLAMA_3B_ROPE_THETA
    max_seq_len: int = 2048
    dropout: float = 0.0
    qkv_bias: bool = False  # Qwen uses bias on q/k/v, Llama doesn't

    def __post_init__(self) -> None:
        assert self.d_model == self.n_heads * self.head_dim, (
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) * "
            f"head_dim ({self.head_dim})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )


@dataclass
class LoRAConfig:
    """Dynamic LoRA configuration."""

    rank: int = 16
    alpha: float = 32.0

    @property
    def scaling(self) -> float:
        """LoRA scaling factor: alpha / rank."""
        return self.alpha / self.rank

    # Target names for all 7 linear layers in CoreBlock
    TARGET_NAMES: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass
class ControllerConfig:
    """Controller hypernetwork configuration."""

    state_proj_dim: int = 512
    step_embed_dim: int = 256
    halt_proj_dim: int = 32
    style_dim: int = 64
    style_hidden_dim: int = 1024
    n_basis: int = 8
    max_steps: int = 128
    bottleneck_divisor: int = 48  # basis factorization compression (tuned for ~50M Controller)

    @property
    def input_dim(self) -> int:
        """Total input dimension to style network."""
        return self.state_proj_dim + self.step_embed_dim + self.halt_proj_dim


@dataclass
class HalterConfig:
    """Halter (adaptive computation time) configuration."""

    hidden_dim: int = 256
    halt_bias_init: float = -2.0  # sigmoid(-2) ≈ 0.12 → halt ~step 8
    act_loss_weight: float = 0.01


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # General
    max_recursion_steps: int = 64
    max_seq_len: int = 2048
    batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 16  # 4 GPUs × 4 batch × 16 accum = 256 effective
    num_gpus: int = 4
    seed: int = 42

    # Phase 1: Recover Knowledge
    phase1_steps: int = 20000
    phase1_fixed_depth: int = 8
    phase1_lr: float = 1e-4
    phase1_min_lr: float = 1e-6
    phase1_warmup_steps: int = 1000

    # Phase 2: Awaken Controller
    phase2_steps: int = 15000  # 20k-35k
    phase2_depths: tuple[int, ...] = (4, 8, 16, 32)
    phase2_lr: float = 5e-5
    phase2_min_lr: float = 1e-6
    phase2_warmup_steps: int = 500
    phase2_distill_alpha: float = 0.5
    phase2_distill_temp: float = 2.0
    phase2_teacher_model: str = "Qwen/Qwen2.5-72B-Instruct"

    # Phase 3: Full Ouroboros
    phase3_steps: int = 15000  # 35k-50k
    phase3_lr: float = 1e-5
    phase3_min_lr: float = 1e-6
    phase3_warmup_steps: int = 500
    phase3_act_loss_weight: float = 0.01

    # Optimization
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    bf16: bool = True

    # Logging
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 2000
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "ouroboros"

    @property
    def effective_batch_size(self) -> int:
        """Total effective batch size across all GPUs and accumulation steps."""
        return self.batch_size_per_gpu * self.num_gpus * self.gradient_accumulation_steps


@dataclass
class OuroborosConfig:
    """Top-level Ouroboros configuration.

    Contains all hyperparameters for the complete architecture.
    Use class methods tiny() and default() for preset configs.
    """

    vocab_size: int = LLAMA_3B_VOCAB
    core_block: CoreBlockConfig = field(default_factory=CoreBlockConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    halter: HalterConfig = field(default_factory=HalterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Model identity
    base_model: str = "meta-llama/Llama-3.2-3B"
    n_base_layers: int = LLAMA_3B_N_LAYERS

    def __post_init__(self) -> None:
        self.core_block.__post_init__()

    # -- Helper methods --

    def lora_param_shapes(self) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
        """Return (A_shape, B_shape) for each LoRA target in CoreBlock.

        Returns:
            Dict mapping target name to ((rank, in_features), (out_features, rank)).
        """
        d = self.core_block.d_model
        d_kv = self.core_block.n_kv_heads * self.core_block.head_dim
        d_ff = self.core_block.d_ff
        r = self.lora.rank

        shapes = {
            "q_proj": ((r, d), (d, r)),
            "k_proj": ((r, d), (d_kv, r)),
            "v_proj": ((r, d), (d_kv, r)),
            "o_proj": ((r, d), (d, r)),
            "gate_proj": ((r, d), (d_ff, r)),
            "up_proj": ((r, d), (d_ff, r)),
            "down_proj": ((r, d_ff), (d, r)),
        }
        return shapes

    def total_lora_params_per_step(self) -> int:
        """Total number of LoRA parameters generated per recursion step."""
        total = 0
        for (a_shape, b_shape) in self.lora_param_shapes().values():
            total += a_shape[0] * a_shape[1]  # rank * in_features
            total += b_shape[0] * b_shape[1]  # out_features * rank
        return total

    def target_dims(self) -> dict[str, tuple[int, int]]:
        """Return (in_features, out_features) for each LoRA target.

        These are the dimensions of the linear layers being modified.
        """
        d = self.core_block.d_model
        d_kv = self.core_block.n_kv_heads * self.core_block.head_dim
        d_ff = self.core_block.d_ff

        return {
            "q_proj": (d, d),
            "k_proj": (d, d_kv),
            "v_proj": (d, d_kv),
            "o_proj": (d, d),
            "gate_proj": (d, d_ff),
            "up_proj": (d, d_ff),
            "down_proj": (d_ff, d),
        }

    # -- Presets --

    @classmethod
    def tiny(cls) -> OuroborosConfig:
        """Tiny config for unit tests. ~200K params, runs on CPU."""
        return cls(
            vocab_size=256,
            core_block=CoreBlockConfig(
                d_model=128,
                n_heads=4,
                n_kv_heads=2,
                d_ff=256,
                head_dim=32,
                max_seq_len=64,
            ),
            lora=LoRAConfig(rank=4, alpha=8.0),
            controller=ControllerConfig(
                state_proj_dim=64,
                step_embed_dim=32,
                halt_proj_dim=16,
                style_dim=16,
                style_hidden_dim=64,
                n_basis=4,
                max_steps=16,
                bottleneck_divisor=2,
            ),
            halter=HalterConfig(hidden_dim=64),
            training=TrainingConfig(
                max_recursion_steps=8,
                max_seq_len=64,
                batch_size_per_gpu=2,
                gradient_accumulation_steps=1,
                num_gpus=1,
                phase2_depths=(4, 8),
            ),
            base_model="tiny-test",
            n_base_layers=4,
        )

    @classmethod
    def default(cls) -> OuroborosConfig:
        """Default config based on Llama-3.2-3B. ~555M total params."""
        return cls()

    # -- Serialization --

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dictionary (tuples converted to lists for YAML compat)."""
        d = asdict(self)
        # Convert tuples to lists recursively for YAML safe_load compatibility
        def _convert(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj
        return _convert(d)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> OuroborosConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OuroborosConfig:
        """Construct config from a nested dictionary."""
        core_block = CoreBlockConfig(**data.pop("core_block", {}))
        lora = LoRAConfig(**{k: v for k, v in data.pop("lora", {}).items() if k != "TARGET_NAMES"})
        controller = ControllerConfig(**data.pop("controller", {}))
        halter = HalterConfig(**data.pop("halter", {}))
        # Handle tuple fields in training config
        training_data = data.pop("training", {})
        if "phase2_depths" in training_data and isinstance(training_data["phase2_depths"], list):
            training_data["phase2_depths"] = tuple(training_data["phase2_depths"])
        training = TrainingConfig(**training_data)
        return cls(
            core_block=core_block,
            lora=lora,
            controller=controller,
            halter=halter,
            training=training,
            **data,
        )
