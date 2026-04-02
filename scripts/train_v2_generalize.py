"""V2 with generalization fixes: diverse data, dropout, held-out validation."""

import argparse, time, sys, math
from pathlib import Path
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
sys.path.insert(0, str(Path(__file__).parent.parent))
from ouroboros.ouroboros_v2 import OuroborosV2
from ouroboros.utils import get_logger, seed_everything, format_params

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=50000)
    parser.add_argument("--save_dir", default="checkpoints_v2_gen")
    args = parser.parse_args()
    seed_everything(42)

    model = OuroborosV2(n_prelude=8, n_coda=8, recurrent_layer_idx=18,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, max_recurrence=64,
        default_depth=args.depth, variable_depths=(args.depth,))
    model.load_base_model("Qwen/Qwen2.5-3B")
    model = model.cuda()
    model.set_phase(2)

    # Add dropout to Controller
    if args.dropout > 0 and model.controller is not None:
        old = model.controller.style_net
        new = []
        for layer in old:
            new.append(layer)
            if isinstance(layer, nn.SiLU):
                new.append(nn.Dropout(args.dropout).to(dtype=torch.bfloat16, device="cuda"))
        model.controller.style_net = nn.Sequential(*new)

    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable: {format_params(sum(p.numel() for p in trainable))}")

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    from training.data import StreamingTextDataset, InterleavedDataset, create_dataloader
    train_ds = InterleavedDataset([
        StreamingTextDataset("HuggingFaceFW/fineweb", tok, max_seq_len=2048, subset="sample-10BT", streaming=True),
        StreamingTextDataset("HuggingFaceFW/fineweb-edu", tok, max_seq_len=2048, subset="sample-10BT", streaming=True),
    ], weights=[0.5, 0.5])
    dl = create_dataloader(train_ds, batch_size=args.batch_size, num_workers=2)

    val_texts = [
        "The mitochondria is the powerhouse of the cell responsible for producing adenosine triphosphate.",
        "In 1969 Neil Armstrong became the first human to walk on the Moon during Apollo 11.",
        "The Fibonacci sequence appears throughout nature from spiral leaf arrangements to rabbit breeding.",
        "Climate change refers to long-term shifts in temperatures primarily driven by human activities.",
        "Shakespeare wrote 37 plays including Hamlet Macbeth and Romeo and Juliet.",
        "Convolutional neural networks use learned filters to detect local patterns in input data.",
        "The human genome contains approximately three billion base pairs of DNA.",
        "Quantum entanglement is a phenomenon where measuring one particle determines another state.",
        "The Standard Model describes three of four fundamental forces and all known particles.",
        "Economic indicators such as GDP and unemployment provide insights into national economic health.",
    ]
    val_ids = [tok(t, return_tensors="pt").input_ids.cuda() for t in val_texts]

    def val_loss():
        model.eval()
        ls = []
        for ids in val_ids:
            with torch.no_grad(): ls.append(model(ids, labels=ids)["loss"].item())
        model.train()
        return sum(ls)/len(ls)

    def val_baseline():
        model.eval()
        ls = []
        for ids in val_ids:
            with torch.no_grad(): ls.append(model.forward_base_only(ids, labels=ids)["loss"].item())
        model.train()
        return sum(ls)/len(ls)

    vb = val_baseline()
    logger.info(f"HELD-OUT BASELINE: {vb:.4f}")

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    it = iter(dl); rl = 0.0; optimizer.zero_grad(); best = float("inf")

    for step in range(1, args.steps+1):
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        o = model(b["input_ids"].cuda(), labels=b["labels"].cuda())
        (o["loss"]/args.grad_accum).backward()
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if step < 2000:
                for pg in optimizer.param_groups: pg["lr"] = args.lr * step/2000
            optimizer.step()
            if step >= 2000: scheduler.step()
            optimizer.zero_grad()
        rl += o["loss"].item()
        if step % args.log_every == 0:
            logger.info(f"Step {step}/{args.steps} | train={rl/args.log_every:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")
            rl = 0.0
        if step % args.val_every == 0:
            vl = val_loss()
            d = vl - vb
            tag = "BETTER" if d < 0 else "worse"
            logger.info(f"  VAL: {vl:.4f} | base={vb:.4f} | delta={d:+.4f} ({tag})")
            if vl < best:
                best = vl
                torch.save({"controller": model.controller.state_dict(), "gate": model.gate.state_dict(),
                    "step_norm": model.step_norm.state_dict(), "step": step, "val_loss": vl, "val_base": vb},
                    str(save_dir/"best.pt"))
                logger.info(f"  NEW BEST (val={vl:.4f})")
        if step % args.save_every == 0:
            torch.save({"controller": model.controller.state_dict(), "gate": model.gate.state_dict(),
                "step_norm": model.step_norm.state_dict(), "step": step}, str(save_dir/f"step_{step}.pt"))

    vf = val_loss()
    logger.info(f"\nFINAL: val={vf:.4f} | base={vb:.4f} | delta={vf-vb:+.4f}")
    logger.info(f"BEST:  val={best:.4f} | delta={best-vb:+.4f}")
    logger.info("GENERALIZES!" if best < vb else "Did not generalize")

if __name__ == "__main__":
    main()
