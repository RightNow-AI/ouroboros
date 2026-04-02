"""Hard benchmark: Qwen 0.5B + SGRC vs Qwen 1.5B vs Qwen 3B.
Can a 0.5B model with 42M extra SGRC params compete with 3-6x larger models?
"""

import math, time, torch
from pathlib import Path
from ouroboros.config import OuroborosConfig
from ouroboros.gated_refiner import OuroborosGatedRefiner
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 70)
print("HARD BENCHMARK: Qwen 0.5B+SGRC vs 1.5B vs 3B")
print("=" * 70)

# Load all models
print("\nLoading models...")
config = OuroborosConfig.from_yaml("configs/ouroboros_qwen05.yaml")
config.training.phase2_depths = (4,)
refiner = OuroborosGatedRefiner(config, base_model_name="Qwen/Qwen2.5-0.5B")
refiner = refiner.cuda()
refiner.set_phase(2)
ckpt = torch.load("checkpoints_gated_refiner/final.pt", map_location="cuda", weights_only=False)
refiner.controller.load_state_dict(ckpt["controller"])
refiner.eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

qwen05 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).cuda().eval()
print("  0.5B loaded")
qwen15 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.bfloat16).cuda().eval()
print("  1.5B loaded")
qwen3b = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", torch_dtype=torch.bfloat16).cuda().eval()
print("  3B loaded")

def hf_loss(model, ids):
    with torch.no_grad():
        return model(ids, labels=ids).loss.item()

def sgrc_loss(ids):
    with torch.no_grad():
        return refiner.forward(ids, labels=ids)["loss"].item()

# === TEST 1: Loss across categories ===
print("\n" + "=" * 70)
print("TEST 1: LOSS COMPARISON")
print("=" * 70)

texts = [
    ("facts", "The capital of France is Paris. It is the largest city in France and is located on the Seine River."),
    ("math", "The sum of the first n natural numbers is n times n plus one divided by two. For n equals 10, the sum is 55."),
    ("code", "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid"),
    ("reasoning", "If all roses are flowers and all flowers need water, then all roses need water. This is transitive reasoning."),
    ("science", "Photosynthesis is the process by which green plants convert carbon dioxide and water into glucose and oxygen using sunlight."),
    ("history", "The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing from hand to machine production."),
    ("quantum", "In quantum mechanics, the Heisenberg uncertainty principle states that certain pairs of physical properties cannot both be known precisely."),
    ("story", "The old lighthouse keeper climbed the spiral staircase every evening at dusk, carrying his worn lantern and a thermos of black coffee."),
]

print(f"\n{'Category':<12} {'0.5B':>8} {'0.5B+SGRC':>10} {'1.5B':>8} {'3B':>8} {'SGRC rank':>10}")
print("-" * 60)

tot = {"05": 0, "sg": 0, "15": 0, "3b": 0}
wins15 = 0
wins3b = 0

for name, text in texts:
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    l05 = hf_loss(qwen05, ids)
    lsg = sgrc_loss(ids)
    l15 = hf_loss(qwen15, ids)
    l3b = hf_loss(qwen3b, ids)

    tot["05"] += l05; tot["sg"] += lsg; tot["15"] += l15; tot["3b"] += l3b
    if lsg < l15: wins15 += 1
    if lsg < l3b: wins3b += 1

    scores = sorted([(l3b, "3B"), (l15, "1.5B"), (lsg, "SGRC"), (l05, "0.5B")])
    rank = [s[1] for s in scores].index("SGRC") + 1
    print(f"{name:<12} {l05:>8.3f} {lsg:>10.3f} {l15:>8.3f} {l3b:>8.3f} {'#'+str(rank):>10}")

n = len(texts)
print(f"\n{'AVERAGE':<12} {tot['05']/n:>8.3f} {tot['sg']/n:>10.3f} {tot['15']/n:>8.3f} {tot['3b']/n:>8.3f}")
print(f"\nSGRC beats 1.5B in {wins15}/{n} categories")
print(f"SGRC beats 3B   in {wins3b}/{n} categories")

# === TEST 2: Perplexity on longer passages ===
print("\n" + "=" * 70)
print("TEST 2: PERPLEXITY ON LONGER PASSAGES")
print("=" * 70)

passages = [
    ("ML text", "Machine learning is a branch of artificial intelligence that enables computers to learn from data and improve their performance without being explicitly programmed. Supervised learning involves training models on labeled datasets where the correct output is known. The model learns to map inputs to outputs by minimizing a loss function."),
    ("History", "The French Revolution of 1789 was a period of radical political and societal change that began with the Estates General and ended with the formation of a new republic. The revolution was driven by widespread discontent with the monarchy, economic hardship, and Enlightenment ideals of liberty and equality."),
    ("Code doc", "A hash table is a data structure that implements an associative array abstract data type mapping keys to values. It uses a hash function to compute an index into an array of buckets from which the desired value can be found. Most hash table designs employ collision resolution strategies."),
]

print(f"\n{'Passage':<12} {'0.5B':>10} {'0.5B+SGRC':>10} {'1.5B':>10} {'3B':>10}")
print("-" * 55)
for name, text in passages:
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    p05 = math.exp(min(hf_loss(qwen05, ids), 20))
    psg = math.exp(min(sgrc_loss(ids), 20))
    p15 = math.exp(min(hf_loss(qwen15, ids), 20))
    p3b = math.exp(min(hf_loss(qwen3b, ids), 20))
    print(f"{name:<12} {p05:>10.1f} {psg:>10.1f} {p15:>10.1f} {p3b:>10.1f}")

# === TEST 3: Efficiency ===
print("\n" + "=" * 70)
print("TEST 3: PARAMETER EFFICIENCY")
print("=" * 70)
info = [
    ("Qwen 0.5B", 494, tot["05"]/n),
    ("0.5B + SGRC", 536, tot["sg"]/n),
    ("Qwen 1.5B", 1544, tot["15"]/n),
    ("Qwen 3B", 3090, tot["3b"]/n),
]
print(f"\n{'Model':<16} {'Params(M)':>10} {'Avg Loss':>10} {'Loss/B params':>14}")
print("-" * 52)
for nm, params, loss in info:
    print(f"{nm:<16} {params:>10} {loss:>10.3f} {loss/(params/1000):>14.3f}")

# === TEST 4: Speed ===
print("\n" + "=" * 70)
print("TEST 4: INFERENCE SPEED (ms/forward)")
print("=" * 70)
test_ids = tok("The quick brown fox jumps over the lazy dog", return_tensors="pt").input_ids.cuda()
for _ in range(3): hf_loss(qwen05, test_ids)  # warmup

for nm, fn in [("Qwen 0.5B", lambda: hf_loss(qwen05, test_ids)),
               ("0.5B+SGRC", lambda: sgrc_loss(test_ids)),
               ("Qwen 1.5B", lambda: hf_loss(qwen15, test_ids)),
               ("Qwen 3B", lambda: hf_loss(qwen3b, test_ids))]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20): fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"  {nm:<16} {ms:>8.1f} ms")

# === SUMMARY ===
print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)
avg_sg = tot["sg"]/n
avg_15 = tot["15"]/n
avg_3b = tot["3b"]/n
print(f"\n0.5B+SGRC (536M) vs 1.5B (1544M):")
print(f"  Avg loss: {avg_sg:.3f} vs {avg_15:.3f}")
print(f"  Result: SGRC {'WINS' if avg_sg < avg_15 else 'LOSES'} with {1544/536:.1f}x fewer params")
print(f"\n0.5B+SGRC (536M) vs 3B (3090M):")
print(f"  Avg loss: {avg_sg:.3f} vs {avg_3b:.3f}")
print(f"  Result: SGRC {'WINS' if avg_sg < avg_3b else 'LOSES'} against {3090/536:.1f}x larger model")

del qwen05, qwen15, qwen3b
print(f"\nPeak GPU: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print("=" * 70)
