"""Train V2 with STATIC per-step LoRA (no Controller). Critical ablation."""

import argparse, time, sys
from pathlib import Path
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))
from ouroboros.ouroboros_v2 import OuroborosV2
from ouroboros.utils import get_logger, seed_everything, format_params
from training.data import get_phase2_dataset, create_dataloader

logger = get_logger(__name__)


class StaticPerStepDiag(nn.Module):
    """Static learnable diagonal per step. No Controller, no input dependence."""
    def __init__(self, n_steps, rank, target_names):
        super().__init__()
        self.target_names = target_names
        self.diags = nn.Parameter(torch.zeros(n_steps, len(target_names), rank))
        self.n_steps = n_steps

    def get_diags(self, step):
        idx = min(step, self.n_steps - 1)
        return {name: self.diags[idx, i].unsqueeze(0) for i, name in enumerate(self.target_names)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=50000)
    parser.add_argument("--save_dir", default="checkpoints_static")
    args = parser.parse_args()

    seed_everything(42)

    model = OuroborosV2(
        n_prelude=8, n_coda=8, recurrent_layer_idx=18,
        lora_rank=args.lora_rank, lora_alpha=16.0, max_recurrence=64,
        default_depth=args.depth, variable_depths=(args.depth,),
    )
    model.load_base_model("Qwen/Qwen2.5-3B")
    model = model.cuda()

    target_names = list(model._svd_lora_modules.keys())
    static_diag = StaticPerStepDiag(64, args.lora_rank, target_names)
    static_diag = static_diag.to(dtype=torch.bfloat16, device="cuda")

    model.controller.requires_grad_(False)
    model.gate.requires_grad_(True)
    model.step_norm.requires_grad_(True)
    model.train()

    # Monkey-patch recurrent step to use static diags
    def static_step(h, pos_emb, step):
        diags = static_diag.get_diags(step)
        for name, mod in model._svd_lora_modules.items():
            mod.set_diag(diags[name].expand(h.shape[0], -1))
        out = model.recurrent_layer(h, position_embeddings=pos_emb)
        h_new = out[0] if isinstance(out, tuple) else out
        for mod in model._svd_lora_modules.values():
            mod.clear_diag()
        h_new = model.step_norm(h_new, step)
        return model.gate(h_new, h)

    model._run_recurrent_step = static_step

    trainable = list(static_diag.parameters()) + [p for p in model.gate.parameters()] + [p for p in model.step_norm.parameters()]
    logger.info(f"[STATIC] Trainable: {format_params(sum(p.numel() for p in trainable))}")

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    ds = get_phase2_dataset(tok, max_seq_len=2048)
    dl = create_dataloader(ds, batch_size=args.batch_size, num_workers=2)

    it = iter(dl)
    bl = []
    for i in range(20):
        b = next(it)
        with torch.no_grad():
            bl.append(model.forward_base_only(b["input_ids"].cuda(), labels=b["labels"].cuda())["loss"].item())
    baseline = sum(bl) / len(bl)
    logger.info(f"BASELINE: {baseline:.4f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    it = iter(dl)
    rl = 0.0
    optimizer.zero_grad()

    for step in range(1, args.steps + 1):
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        o = model(b["input_ids"].cuda(), labels=b["labels"].cuda())
        (o["loss"] / args.grad_accum).backward()
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if step < 2000:
                for pg in optimizer.param_groups: pg["lr"] = args.lr * step / 2000
            optimizer.step()
            if step >= 2000: scheduler.step()
            optimizer.zero_grad()
        rl += o["loss"].item()
        if step % args.log_every == 0:
            avg = rl / args.log_every
            logger.info(f"[STATIC] Step {step}/{args.steps} | loss={avg:.4f} | base={baseline:.4f} | delta={avg-baseline:+.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")
            rl = 0.0
        if step % args.save_every == 0:
            torch.save({"static_diag": static_diag.state_dict(), "step": step, "baseline": baseline}, str(save_dir / f"step_{step}.pt"))

    torch.save({"static_diag": static_diag.state_dict(), "step": args.steps, "baseline": baseline}, str(save_dir / "final.pt"))
    model.eval()
    fl = []
    it = iter(dl)
    for i in range(50):
        b = next(it)
        with torch.no_grad():
            fl.append(model(b["input_ids"].cuda(), labels=b["labels"].cuda())["loss"].item())
    final = sum(fl) / len(fl)
    logger.info(f"[STATIC] BASELINE: {baseline:.4f} | FINAL: {final:.4f} | DELTA: {final-baseline:+.4f}")

if __name__ == "__main__":
    main()
