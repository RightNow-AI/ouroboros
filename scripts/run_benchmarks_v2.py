"""Experiment B: Held-out benchmarks for V2. Compare against full Qwen 3B."""

import math, torch, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ouroboros.ouroboros_v2 import OuroborosV2
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 70)
print("EXPERIMENT B: HELD-OUT BENCHMARKS")
print("=" * 70)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

model = OuroborosV2(n_prelude=8, n_coda=8, recurrent_layer_idx=18, lora_rank=32, variable_depths=(8,))
model.load_base_model("Qwen/Qwen2.5-3B")
model = model.cuda()
for p in ["checkpoints_v2/lr_high/final.pt", "checkpoints_v2/depth_8/final.pt"]:
    if Path(p).exists():
        ckpt = torch.load(p, map_location="cuda", weights_only=False)
        model.controller.load_state_dict(ckpt["controller"])
        model.gate.load_state_dict(ckpt["gate"])
        model.step_norm.load_state_dict(ckpt["step_norm"])
        print(f"Loaded: {p}")
        break
model.set_phase(2)
model.eval()

full_3b = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", torch_dtype=torch.bfloat16).cuda().eval()

held_out = [
    "The mitochondria is the powerhouse of the cell responsible for producing adenosine triphosphate through oxidative phosphorylation.",
    "In 1969 Neil Armstrong became the first human to walk on the Moon during the Apollo 11 mission.",
    "SELECT users.name orders.total FROM users INNER JOIN orders ON users.id equals orders.user_id WHERE orders.total greater than 100",
    "The Fibonacci sequence appears throughout nature from the spiral arrangement of leaves to the breeding patterns of rabbits.",
    "Climate change refers to long-term shifts in global temperatures primarily driven by human activities since the Industrial Revolution.",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])",
    "Shakespeare wrote 37 plays and 154 sonnets during his career including Hamlet Macbeth and Romeo and Juliet.",
    "The Standard Model of particle physics describes three of the four fundamental forces and classifies all known elementary particles.",
    "Convolutional neural networks use learned filters to detect local patterns in input data making them effective for image recognition.",
    "The human genome contains approximately three billion base pairs of DNA organized into 23 pairs of chromosomes.",
    "Economic indicators such as GDP unemployment rate and inflation provide insights into the health of a national economy.",
    "Quantum entanglement is a phenomenon where two particles become correlated such that measuring one determines the state of the other.",
]

print(f"\n{'#':<4} {'Full 3B':>10} {'17-layer':>10} {'V2+SGRC':>10}")
print("-" * 38)

tf = tb = ts = 0
wb = wf = 0

for i, text in enumerate(held_out):
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        lf = full_3b(ids, labels=ids).loss.item()
        lb = model.forward_base_only(ids, labels=ids)["loss"].item()
        ls = model.forward(ids, labels=ids)["loss"].item()
    tf += lf; tb += lb; ts += ls
    if ls < lb: wb += 1
    if ls < lf: wf += 1
    print(f"{i+1:<4} {lf:>10.3f} {lb:>10.3f} {ls:>10.3f}")

n = len(held_out)
print(f"\n{'AVG':<4} {tf/n:>10.3f} {tb/n:>10.3f} {ts/n:>10.3f}")
print(f"\nV2+SGRC beats 17-layer: {wb}/{n}")
print(f"V2+SGRC beats full 3B:  {wf}/{n}")
rec = (tb/n - ts/n) / (tb/n - tf/n) * 100 if tb/n != tf/n else 0
print(f"Gap recovered: {rec:.1f}%")

print(f"\nPeak GPU: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print("EXPERIMENT B COMPLETE")
