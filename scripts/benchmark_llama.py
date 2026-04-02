"""Benchmark the Llama Ouroboros model after all 3 phases."""

import torch
from pathlib import Path
from ouroboros.config import OuroborosConfig
from ouroboros.model import OuroborosModel
from ouroboros.halter import ACTManager
from transformers import AutoTokenizer

print("=" * 60)
print("OUROBOROS LLAMA-3.2-3B BENCHMARK")
print("=" * 60)

# Load config and model
config = OuroborosConfig.from_yaml("configs/ouroboros_config.yaml")
model = OuroborosModel(config).cuda().bfloat16()

# Load Phase 3 final checkpoint
ckpt_dir = Path("checkpoints/phase3/final")
if not ckpt_dir.exists():
    phase3_dir = Path("checkpoints/phase3")
    subdirs = sorted([d for d in phase3_dir.iterdir() if d.is_dir() and d.name.startswith("step_")])
    ckpt_dir = subdirs[-1] if subdirs else phase3_dir

subdirs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir()])
mp_file = (subdirs[0] if subdirs else ckpt_dir) / "mp_rank_00_model_states.pt"

if mp_file.exists():
    state = torch.load(str(mp_file), map_location="cuda", weights_only=False)
    model.load_state_dict(state["module"], strict=False)
    print(f"Loaded: {mp_file}")
else:
    print(f"ERROR: No checkpoint at {mp_file}")
    raise SystemExit(1)

tokenizer = AutoTokenizer.from_pretrained("ouroboros-600m")

# === TEST 1: Loss on diverse text ===
print("\n" + "=" * 60)
print("TEST 1: LOSS ON DIVERSE TEXT")
print("=" * 60)

test_texts = {
    "simple_fact": "The capital of France is Paris. It is located in Europe.",
    "math": "If x = 5 and y = 3, then x + y = 8 and x * y = 15.",
    "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    "reasoning": "All mammals are warm-blooded. Dogs are mammals. Therefore, dogs are warm-blooded.",
    "story": "Once upon a time, there was a brave knight who lived in a castle on top of a mountain.",
}

for phase in [1, 2, 3]:
    model.set_phase(phase)
    model.eval()
    print(f"\n--- Phase {phase} ---")
    for name, text in test_texts.items():
        tokens = tokenizer(text, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model(tokens, labels=tokens)
        loss_val = out["loss"].item()
        print(f"  {name:15s}: loss={loss_val:.3f}")

# === TEST 2: Generation quality ===
print("\n" + "=" * 60)
print("TEST 2: GENERATION QUALITY")
print("=" * 60)

model.set_phase(1)
model.eval()

prompts = [
    "The quick brown fox",
    "In the year 2024,",
    "def fibonacci(n):",
    "The meaning of life is",
    "1 + 1 = 2, 2 + 2 = 4, 3 + 3 =",
    "Paris is the capital of",
]

for prompt in prompts:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=40, temperature=0.7, top_p=0.9)
    output = tokenizer.decode(gen[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {output[:150]}")

# === TEST 3: Controller ablation ===
print("\n" + "=" * 60)
print("TEST 3: CONTROLLER ABLATION (Phase 1 vs Phase 2)")
print("=" * 60)

test_text = "The theory of relativity was developed by Albert Einstein in the early twentieth century."
tokens = tokenizer(test_text, return_tensors="pt").input_ids.cuda()

for phase in [1, 2]:
    model.set_phase(phase)
    model.eval()
    with torch.no_grad():
        out = model(tokens, labels=tokens)
    status = "OFF" if phase == 1 else "ON"
    loss_val = out["loss"].item()
    print(f"  Controller {status} (Phase {phase}): loss={loss_val:.3f}")

# === TEST 4: Adaptive depth ===
print("\n" + "=" * 60)
print("TEST 4: ADAPTIVE DEPTH (Phase 3)")
print("=" * 60)

model.set_phase(3)
model.eval()

depth_prompts = [
    ("easy", "The cat sat on the mat."),
    ("medium", "Explain why the sky appears blue during the day."),
    ("hard", "Prove that the square root of 2 is irrational."),
]

for difficulty, prompt in depth_prompts:
    tokens_in = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    batch, seq_len = tokens_in.shape
    h = model.embed_tokens(tokens_in)
    d_model = h.shape[-1]
    act = ACTManager(batch, seq_len, d_model, h.device, h.dtype)
    steps_used = 0

    with torch.no_grad():
        for step in range(64):
            halt_conf = act.cumulative_halt.mean(dim=1, keepdim=True)
            deltas = model.controller(h, step, halt_conf)
            model.core_block.set_all_lora(deltas)
            h = model.core_block(h)
            model.core_block.clear_all_lora()
            halt_prob = model.halter(h)
            all_halted, _ = act.step(h, halt_prob)
            steps_used = step + 1
            if all_halted:
                break

    mean_d = act.mean_steps()
    print(f"  [{difficulty:6s}] \"{prompt[:50]}\" -> mean_depth={mean_d:.1f}, max_step={steps_used}")

mem = torch.cuda.max_memory_allocated() / 1e9
print(f"\nPeak GPU memory: {mem:.2f} GB")
print("\n" + "=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
