"""Main training script for Ouroboros.

Supports 3-phase training with DeepSpeed ZeRO-2:
    Phase 1: Recover pretrained knowledge (fixed depth, Controller frozen)
    Phase 2: Awaken Controller (variable depth, with distillation)
    Phase 3: Full Ouroboros (adaptive depth, ACT)

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed --num_gpus=4 training/train.py \
        --phase 1 --config configs/ouroboros_config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ouroboros.config import OuroborosConfig
from ouroboros.model import OuroborosModel
from ouroboros.utils import get_logger, seed_everything, format_params

logger = get_logger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Cosine annealing with linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def setup_deepspeed(
    model: OuroborosModel,
    args: argparse.Namespace,
    config: OuroborosConfig,
) -> tuple:
    """Initialize DeepSpeed engine."""
    import deepspeed

    tc = config.training
    phase = args.phase

    # Phase-specific LR
    lr_map = {1: tc.phase1_lr, 2: tc.phase2_lr, 3: tc.phase3_lr}
    lr = lr_map[phase]

    # Optimizer params: only trainable parameters
    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}
    ]

    # Load DeepSpeed config
    ds_config_path = args.ds_config or "configs/ds_config.json"
    with open(ds_config_path) as f:
        ds_config = json.load(f)

    # Override with our settings
    ds_config["train_micro_batch_size_per_gpu"] = tc.batch_size_per_gpu
    ds_config["gradient_accumulation_steps"] = tc.gradient_accumulation_steps
    ds_config["optimizer"]["params"]["lr"] = lr

    # Phase-specific scheduler config
    warmup_map = {1: tc.phase1_warmup_steps, 2: tc.phase2_warmup_steps, 3: tc.phase3_warmup_steps}
    steps_map = {1: tc.phase1_steps, 2: tc.phase2_steps, 3: tc.phase3_steps}
    min_lr_map = {1: tc.phase1_min_lr, 2: tc.phase2_min_lr, 3: tc.phase3_min_lr}

    ds_config["scheduler"] = {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": min_lr_map[phase],
            "warmup_max_lr": lr,
            "warmup_num_steps": warmup_map[phase],
            "total_num_steps": steps_map[phase],
        },
    }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        config=ds_config,
    )

    return model_engine, optimizer, scheduler


def train_phase(
    model_engine: any,
    dataloader: any,
    config: OuroborosConfig,
    phase: int,
    args: argparse.Namespace,
) -> None:
    """Main training loop for a single phase."""
    tc = config.training
    steps_map = {1: tc.phase1_steps, 2: tc.phase2_steps, 3: tc.phase3_steps}
    total_steps = steps_map[phase]

    # Wandb logging
    if args.wandb and model_engine.local_rank == 0:
        import wandb
        wandb.init(
            project=tc.wandb_project,
            name=f"ouroboros-phase{phase}",
            config=config.to_dict(),
        )

    # Training loop
    model_engine.train()
    data_iter = iter(dataloader)
    global_step = args.resume_step
    start_time = time.time()
    running_loss = 0.0

    logger.info(f"Phase {phase}: Training for {total_steps} steps "
                f"(effective batch={tc.effective_batch_size})")

    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)

        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]

        model_engine.backward(loss)
        model_engine.step()  # DeepSpeed handles optimizer + scheduler step

        # Logging
        running_loss += loss.item()
        global_step += 1

        if global_step % tc.log_every == 0 and model_engine.local_rank == 0:
            avg_loss = running_loss / tc.log_every
            elapsed = time.time() - start_time
            tokens_per_sec = (
                tc.log_every * tc.effective_batch_size * tc.max_seq_len / elapsed
            )
            current_lr = model_engine.get_lr()[0] if hasattr(model_engine, 'get_lr') else 0.0

            log_dict = {
                "loss": avg_loss,
                "lr": current_lr,
                "tokens_per_sec": tokens_per_sec,
                "step": global_step,
            }

            if phase == 3:
                log_dict["act_loss"] = outputs["act_loss"].item()
                log_dict["mean_depth"] = outputs["mean_depth"]

            logger.info(
                f"Step {global_step}/{total_steps} | "
                f"loss={avg_loss:.4f} | lr={current_lr:.2e} | "
                f"tok/s={tokens_per_sec:.0f}"
            )

            if args.wandb and model_engine.local_rank == 0:
                import wandb
                wandb.log(log_dict, step=global_step)

            running_loss = 0.0
            start_time = time.time()

        # Checkpointing
        if global_step % tc.save_every == 0:
            save_dir = Path(tc.checkpoint_dir) / f"phase{phase}" / f"step_{global_step}"
            model_engine.save_checkpoint(str(save_dir))
            if model_engine.local_rank == 0:
                logger.info(f"Saved checkpoint at step {global_step}")

    # Final save
    save_dir = Path(tc.checkpoint_dir) / f"phase{phase}" / "final"
    model_engine.save_checkpoint(str(save_dir))
    if model_engine.local_rank == 0:
        logger.info(f"Phase {phase} training complete. Final checkpoint saved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Ouroboros")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--config", type=str, default="configs/ouroboros_config.yaml")
    parser.add_argument("--ds_config", type=str, default="configs/ds_config.json")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)  # DeepSpeed adds this
    args = parser.parse_args()

    # Load config
    config = OuroborosConfig.from_yaml(args.config)
    seed_everything(config.training.seed)

    # Create model
    ds_checkpoint_dir = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        # Check if this is a DeepSpeed checkpoint (has global_step* subdirs)
        ds_subdirs = list(ckpt_path.glob("global_step*"))
        if ds_subdirs:
            # DeepSpeed checkpoint — create fresh model, load weights after DS init
            logger.info(f"DeepSpeed checkpoint detected at {args.checkpoint}")
            model = OuroborosModel(config)
            ds_checkpoint_dir = args.checkpoint
        elif (ckpt_path / "model.pt").exists():
            # Our custom format (from convert.py)
            logger.info(f"Loading converted model from {args.checkpoint}")
            from ouroboros.convert import load_ouroboros
            model, config = load_ouroboros(args.checkpoint)
        else:
            logger.error(f"Unknown checkpoint format at {args.checkpoint}")
            sys.exit(1)
    else:
        model = OuroborosModel(config)

    # Set training phase
    model.set_phase(args.phase)
    model.enable_gradient_checkpointing()

    # Log parameter counts
    param_summary = model.param_summary()
    logger.info(f"Parameter summary: {param_summary}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {format_params(trainable)} / {format_params(total)}")

    # Setup DeepSpeed
    model_engine, optimizer, scheduler = setup_deepspeed(model, args, config)

    # Load DeepSpeed checkpoint weights if applicable
    if ds_checkpoint_dir:
        logger.info(f"Loading DeepSpeed weights from {ds_checkpoint_dir}")
        # Only load model weights, not optimizer/scheduler state
        # (phase transitions change which params are trainable)
        model_engine.load_checkpoint(
            ds_checkpoint_dir,
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
        )

    # Setup data
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from training.data import (
        get_phase1_dataset, get_phase2_dataset, get_phase3_dataset, create_dataloader,
    )

    dataset_map = {
        1: get_phase1_dataset,
        2: get_phase2_dataset,
        3: get_phase3_dataset,
    }
    dataset = dataset_map[args.phase](tokenizer, config.training.max_seq_len)
    dataloader = create_dataloader(dataset, config.training.batch_size_per_gpu)

    # Train
    train_phase(model_engine, dataloader, config, args.phase, args)


if __name__ == "__main__":
    main()
