"""Benchmark: Qwen 0.5B base vs Qwen + Gated SGRC Refiner."""

import math
import torch
from pathlib import Path
from ouroboros.config import OuroborosConfig
from ouroboros.gated_refiner import OuroborosGatedRefiner
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 70)
print("BENCHMARK: Qwen 0.5B BASE vs Qwen + GATED SGRC")
print("=" * 70)

config = OuroborosConfig.from_yaml("configs/ouroboros_qwen05.yaml")
config.training.phase2_depths = (4,)

refiner = OuroborosGatedRefiner(config, base_model_name="Qwen/Qwen2.5-0.5B")
refiner = refiner.cuda()
refiner.set_phase(2)

ckpt = torch.load("checkpoints_gated_refiner/final.pt", map_location="cuda", weights_only=False)
refiner.controller.load_state_dict(ckpt["controller"])
print(f"Loaded Controller from step {ckpt['step']}")

refiner.eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

base_qwen = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16
).cuda().eval()

# === TEST 1: Loss ===
print("\n" + "=" * 70)
print("TEST 1: LOSS COMPARISON")
print("=" * 70)

test_texts = [
    ("simple", "The capital of France is Paris. It is located in Europe."),
    ("math", "If x = 5 and y = 3, then x + y = 8 and x * y = 15."),
    ("code", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
    ("reasoning", "All mammals are warm-blooded. Dogs are mammals. Therefore, dogs are warm-blooded."),
    ("story", "Once upon a time, there was a brave knight who lived in a castle on a mountain."),
    ("science", "Water molecules consist of two hydrogen atoms and one oxygen atom bonded together."),
    ("history", "The Roman Empire fell in 476 AD when the last emperor was deposed by Germanic tribes."),
]

print(f"\n{'Category':<12} {'Base':>10} {'+SGRC':>10} {'Delta':>10} {'Result':>10}")
print("-" * 55)

total_base = 0.0
total_sgrc = 0.0
for name, text in test_texts:
    tokens = tokenizer(text, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        base_out = refiner.forward_base_only(tokens, labels=tokens)
        sgrc_out = refiner.forward(tokens, labels=tokens)
    b = base_out["loss"].item()
    s = sgrc_out["loss"].item()
    d = s - b
    total_base += b
    total_sgrc += s
    result = "BETTER" if d < 0 else "worse"
    print(f"{name:<12} {b:>10.3f} {s:>10.3f} {d:>+10.3f} {result:>10}")

avg_base = total_base / len(test_texts)
avg_sgrc = total_sgrc / len(test_texts)
print(f"\n{'AVERAGE':<12} {avg_base:>10.3f} {avg_sgrc:>10.3f} {avg_sgrc-avg_base:>+10.3f}")
pct = (avg_base - avg_sgrc) / avg_base * 100
print(f"Improvement: {pct:.1f}%")

# === TEST 2: Generation ===
print("\n" + "=" * 70)
print("TEST 2: GENERATION (Base Qwen)")
print("=" * 70)

prompts = [
    "The quick brown fox",
    "1 + 1 = 2, 2 + 2 = 4, 3 + 3 =",
    "Paris is the capital of",
    "The theory of relativity",
    "def fibonacci(n):",
]

for prompt in prompts:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        gen = base_qwen.generate(
            ids, max_new_tokens=40, temperature=0.7, top_p=0.9,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    print(f"\n  Prompt: {prompt}")
    print(f"  Output: {tokenizer.decode(gen[0], skip_special_tokens=True)[:130]}")

# === TEST 3: Perplexity ===
print("\n" + "=" * 70)
print("TEST 3: PERPLEXITY ON LONGER TEXT")
print("=" * 70)

long_texts = [
    ("ML intro", "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Instead of being explicitly programmed to perform a task, these systems use algorithms to identify patterns in data and make decisions with minimal human intervention."),
    ("Biology", "The process of photosynthesis converts light energy into chemical energy that can be stored and later released to fuel the organisms activities. This process occurs in plants algae and some bacteria using chlorophyll to absorb light energy."),
]

for name, text in long_texts:
    tokens = tokenizer(text, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        base_out = refiner.forward_base_only(tokens, labels=tokens)
        sgrc_out = refiner.forward(tokens, labels=tokens)
    b = base_out["loss"].item()
    s = sgrc_out["loss"].item()
    ppl_b = math.exp(min(b, 20))
    ppl_s = math.exp(min(s, 20))
    print(f"  {name}: Base PPL={ppl_b:.1f}, SGRC PPL={ppl_s:.1f}, delta={s-b:+.3f}")

# === TEST 4: Depth analysis ===
print("\n" + "=" * 70)
print("TEST 4: VARYING REFINEMENT DEPTH")
print("=" * 70)

test_text = "The theory of relativity was developed by Albert Einstein in the early twentieth century."
tokens = tokenizer(test_text, return_tensors="pt").input_ids.cuda()

print(f"\n{'Depth':>6} {'Loss':>10} {'vs Base':>10}")
print("-" * 28)

with torch.no_grad():
    base_out = refiner.forward_base_only(tokens, labels=tokens)
base_loss = base_out["loss"].item()
print(f"{'0':>6} {base_loss:>10.3f} {'baseline':>10}")

for depth in [1, 2, 4, 8, 16]:
    config.training.phase2_depths = (depth,)
    with torch.no_grad():
        out = refiner.forward(tokens, labels=tokens)
    loss = out["loss"].item()
    d = loss - base_loss
    print(f"{depth:>6} {loss:>10.3f} {d:>+10.3f}")

del base_qwen
torch.cuda.empty_cache()
mem = torch.cuda.max_memory_allocated() / 1e9
print(f"\nPeak GPU memory: {mem:.2f} GB")
print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
