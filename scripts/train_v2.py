"""Train Ouroboros v2 on 8x H100. The paper run."""

import argparse, math, time, sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))

from ouroboros.ouroboros_v2 import OuroborosV2
from ouroboros.utils import get_logger, seed_everything, format_params
from training.data import get_phase2_dataset, create_dataloader

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--save_dir", default="checkpoints_v2")
    args = parser.parse_args()

    seed_everything(42)

    model = OuroborosV2(
        n_prelude=8, n_coda=8, recurrent_layer_idx=18,
        lora_rank=args.lora_rank, lora_alpha=16.0,
        max_recurrence=64, default_depth=args.depth,
        variable_depths=(4, 8, 16), controller_style_dim=128,
    )
    model.load_base_model("Qwen/Qwen2.5-3B")
    model = model.cuda()
    model.set_phase(2)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable: {format_params(sum(p.numel() for p in trainable))}")

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)
    warmup_steps = 2000

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = get_phase2_dataset(tokenizer, max_seq_len=2048)
    dataloader = create_dataloader(dataset, batch_size=args.batch_size, num_workers=2)

    # Baseline
    logger.info("Computing baseline...")
    data_iter = iter(dataloader)
    base_losses = []
    for i in range(20):
        batch = next(data_iter)
        ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        with torch.no_grad():
            out = model.forward_base_only(ids, labels=labels)
        base_losses.append(out["loss"].item())
    baseline = sum(base_losses) / len(base_losses)
    logger.info(f"BASELINE: {baseline:.4f}")

    # Train
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data_iter = iter(dataloader)
    running_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad()

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        output = model(ids, labels=labels)
        loss = output["loss"] / args.grad_accum
        loss.backward()

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * (step / warmup_steps)
            optimizer.step()
            if step >= warmup_steps:
                scheduler.step()
            optimizer.zero_grad()

        running_loss += output["loss"].item()

        if step % args.log_every == 0:
            avg = running_loss / args.log_every
            lr = optimizer.param_groups[0]["lr"]
            d = avg - baseline
            tag = "BETTER" if d < 0 else "worse"
            logger.info(f"Step {step}/{args.steps} | loss={avg:.4f} | base={baseline:.4f} | delta={d:+.4f} ({tag}) | lr={lr:.2e}")
            running_loss = 0.0
            start_time = time.time()

        if step % args.save_every == 0:
            p = save_dir / f"step_{step}.pt"
            torch.save({"controller": model.controller.state_dict(), "gate": model.gate.state_dict(),
                         "step_norm": model.step_norm.state_dict(), "halter": model.halter.state_dict(),
                         "step": step, "baseline": baseline}, str(p))
            logger.info(f"Saved: {p}")

    torch.save({"controller": model.controller.state_dict(), "gate": model.gate.state_dict(),
                 "step_norm": model.step_norm.state_dict(), "halter": model.halter.state_dict(),
                 "step": args.steps, "baseline": baseline}, str(save_dir / "final.pt"))

    model.eval()
    final_losses = []
    data_iter = iter(dataloader)
    for i in range(50):
        batch = next(data_iter)
        with torch.no_grad():
            out = model(batch["input_ids"].cuda(), labels=batch["labels"].cuda())
        final_losses.append(out["loss"].item())
    final = sum(final_losses) / len(final_losses)
    logger.info(f"BASELINE: {baseline:.4f} | FINAL: {final:.4f} | DELTA: {final-baseline:+.4f}")


if __name__ == "__main__":
    main()
