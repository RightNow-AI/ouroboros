"""Experiment G: Visualize Controller diagonal outputs across steps and inputs."""

import torch, sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ouroboros.ouroboros_v2 import OuroborosV2
from transformers import AutoTokenizer

print("=" * 70)
print("EXPERIMENT G: CONTROLLER VISUALIZATION")
print("=" * 70)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
model = OuroborosV2(n_prelude=8, n_coda=8, recurrent_layer_idx=18, lora_rank=32, variable_depths=(8,))
model.load_base_model("Qwen/Qwen2.5-3B")
model = model.cuda()

for p in ["checkpoints_v2/lr_high/final.pt", "checkpoints_v2/depth_8/final.pt"]:
    if Path(p).exists():
        ckpt = torch.load(p, map_location="cuda", weights_only=False)
        model.controller.load_state_dict(ckpt["controller"])
        print(f"Loaded: {p}")
        break

model.set_phase(2)

prompts = {
    "easy_fact": "The sky is blue.",
    "easy_math": "Two plus two equals four.",
    "medium_sci": "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy.",
    "medium_code": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]",
    "hard_reason": "If all mammals are warm-blooded and all whales are mammals then all whales must be warm-blooded.",
    "hard_math": "The integral of e to the power of negative x squared from negative infinity to infinity equals the square root of pi.",
    "hard_quantum": "Quantum entanglement violates Bell inequalities demonstrating that local hidden variable theories cannot explain correlations.",
    "narrative": "The old professor closed the dusty leather-bound book and gazed out the rain-streaked window.",
}

all_diags = {}
for name, text in prompts.items():
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    h = model.embed_tokens(ids)
    pos_emb = model._get_pos_emb(h, ids.shape[1])
    for layer in model.prelude_layers:
        out = layer(h, position_embeddings=pos_emb)
        h = out[0] if isinstance(out, tuple) else out

    step_diags = {}
    with torch.no_grad():
        for step in range(16):
            diags = model.controller(h, step)
            step_diags[step] = {k: v.cpu().numpy().tolist() for k, v in diags.items()}
            for tname, mod in model._svd_lora_modules.items():
                mod.set_diag(diags[tname])
            out = model.recurrent_layer(h, position_embeddings=pos_emb)
            h_new = out[0] if isinstance(out, tuple) else out
            for mod in model._svd_lora_modules.values():
                mod.clear_diag()
            h_new = model.step_norm(h_new, step)
            h = model.gate(h_new, h)
    all_diags[name] = step_diags

print("\n=== STEP VARIATION (do diags change across steps?) ===")
for pn in list(prompts.keys())[:4]:
    d0 = np.array(all_diags[pn][0]["q_proj"][0])
    d7 = np.array(all_diags[pn][7]["q_proj"][0])
    d15 = np.array(all_diags[pn][15]["q_proj"][0])
    print(f"  {pn}: step0vs7 L2={np.linalg.norm(d0-d7):.4f}, step0vs15 L2={np.linalg.norm(d0-d15):.4f}")

print("\n=== INPUT VARIATION (do diags change across inputs?) ===")
for target in ["q_proj", "gate_proj"]:
    print(f"\n  {target} at step 4:")
    vecs = [(pn, np.array(all_diags[pn][4][target][0])) for pn in prompts]
    for i in range(0, len(vecs)-1, 2):
        d = np.linalg.norm(vecs[i][1] - vecs[i+1][1])
        print(f"    {vecs[i][0]} vs {vecs[i+1][0]}: L2={d:.4f}")

print("\n=== MAGNITUDE BY STEP ===")
for pn in ["easy_fact", "hard_quantum"]:
    print(f"\n  {pn}:")
    for s in [0, 4, 8, 15]:
        norms = {t: np.linalg.norm(np.array(all_diags[pn][s][t][0])) for t in ["q_proj", "gate_proj"]}
        print(f"    step {s:2d}: " + " | ".join(f"{t}={v:.4f}" for t, v in norms.items()))

with open("results_visualization.json", "w") as f:
    json.dump(all_diags, f)
print("\nSaved to results_visualization.json")
print("EXPERIMENT G COMPLETE")
