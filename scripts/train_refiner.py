"""Train the OuroborosRefiner — SGRC on top of intact Qwen 0.5B.

Only trains Controller (~40M params). Base model frozen.
Cleanest test: does the Controller improve on base Qwen?
"""

import argparse
import math
import time
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))

from ouroboros.config import OuroborosConfig
from ouroboros.gated_refiner import OuroborosGatedRefiner as OuroborosRefiner
from ouroboros.utils import get_logger, seed_everything, format_params
from training.data import get_phase2_dataset, create_dataloader

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ouroboros_qwen05.yaml")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4, help="Fixed refinement depth")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--save_dir", default="checkpoints_refiner")
    args = parser.parse_args()

    seed_everything(42)
    config = OuroborosConfig.from_yaml(args.config)
    config.training.phase2_depths = (args.depth,)

    refiner = OuroborosRefiner(config, base_model_name="Qwen/Qwen2.5-0.5B")
    refiner = refiner.cuda()
    refiner.set_phase(2)
    refiner.train()

    trainable = [p for p in refiner.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable)
    logger.info(f"Trainable params: {format_params(total_trainable)}")

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = get_phase2_dataset(tokenizer, max_seq_len=config.training.max_seq_len)
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, num_workers=2)

    # Baseline loss
    logger.info("Computing baseline loss (no refinement)...")
    baseline_losses = []
    data_iter = iter(dataloader)
    for i in range(10):
        batch = next(data_iter)
        ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        with torch.no_grad():
            base_out = refiner.forward_base_only(ids, labels=labels)
        baseline_losses.append(base_out["loss"].item())
    baseline = sum(baseline_losses) / len(baseline_losses)
    logger.info(f"BASELINE LOSS (Qwen alone): {baseline:.4f}")

    # Training
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_iter = iter(dataloader)
    running_loss = 0.0
    start_time = time.time()
    logger.info(f"Training for {args.steps} steps, depth={args.depth}, lr={args.lr}")

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()

        output = refiner(ids, labels=labels)
        loss = output["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            delta = avg_loss - baseline
            marker = "BETTER" if delta < 0 else "worse"

            logger.info(
                f"Step {step}/{args.steps} | loss={avg_loss:.4f} | "
                f"baseline={baseline:.4f} | delta={delta:+.4f} ({marker}) | "
                f"lr={lr:.2e}"
            )
            running_loss = 0.0
            start_time = time.time()

        if step % args.save_every == 0:
            ckpt_path = save_dir / f"step_{step}.pt"
            torch.save({
                "controller": refiner.controller.state_dict(),
                "halter": refiner.halter.state_dict(),
                "step": step,
                "baseline_loss": baseline,
            }, str(ckpt_path))
            logger.info(f"Saved: {ckpt_path}")

    # Final save
    torch.save({
        "controller": refiner.controller.state_dict(),
        "halter": refiner.halter.state_dict(),
        "step": args.steps,
        "baseline_loss": baseline,
    }, str(save_dir / "final.pt"))

    # Final comparison
    logger.info("\n=== FINAL COMPARISON ===")
    refiner.eval()
    final_losses = []
    data_iter = iter(dataloader)
    for i in range(50):
        batch = next(data_iter)
        ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        with torch.no_grad():
            out = refiner(ids, labels=labels)
        final_losses.append(out["loss"].item())
    final_avg = sum(final_losses) / len(final_losses)

    logger.info(f"BASELINE (Qwen alone):  {baseline:.4f}")
    logger.info(f"FINAL (Qwen + SGRC):    {final_avg:.4f}")
    logger.info(f"DELTA:                  {final_avg - baseline:+.4f}")
    if final_avg < baseline:
        logger.info("SGRC IMPROVES THE MODEL!")
    else:
        logger.info("SGRC did not improve (needs more training or tuning)")


if __name__ == "__main__":
    main()
