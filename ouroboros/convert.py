"""Pretrained model conversion for Ouroboros.

Surgically converts a pretrained Llama-3.2-3B into Ouroboros:
1. Average all 28 layer weights into one CoreBlock
2. Keep embeddings and LM head from pretrained
3. Add Controller (zero-initialized) and Halter (bias-initialized)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import track

from ouroboros.config import OuroborosConfig
from ouroboros.model import OuroborosModel
from ouroboros.utils import format_params, get_logger

logger = get_logger(__name__)
console = Console()

# Mapping from Llama layer keys to CoreBlock DynamicLoRALinear target names
# Each DynamicLoRALinear wraps a base_linear, so we need the .base_linear prefix
LLAMA_TO_CORE_MAP = {
    "self_attn.q_proj": "attention.q_proj.base_linear",
    "self_attn.k_proj": "attention.k_proj.base_linear",
    "self_attn.v_proj": "attention.v_proj.base_linear",
    "self_attn.o_proj": "attention.o_proj.base_linear",
    "mlp.gate_proj": "ffn.gate_proj.base_linear",
    "mlp.up_proj": "ffn.up_proj.base_linear",
    "mlp.down_proj": "ffn.down_proj.base_linear",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
}


def _average_layer_weights(
    state_dict: dict[str, torch.Tensor],
    n_layers: int,
    layer_prefix: str = "model.layers",
) -> dict[str, torch.Tensor]:
    """Average weights across all transformer layers into a single set.

    Args:
        state_dict: Full pretrained model state dict.
        n_layers: Number of layers in the pretrained model.
        layer_prefix: Prefix for layer keys in the state dict.

    Returns:
        Averaged weights mapped to CoreBlock key names.
    """
    averaged: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    for layer_idx in range(n_layers):
        prefix = f"{layer_prefix}.{layer_idx}."
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            # Strip layer prefix to get the component name
            component_key = key[len(prefix):]

            # Map to CoreBlock key
            core_key = None
            for llama_key, ouro_key in LLAMA_TO_CORE_MAP.items():
                if component_key.startswith(llama_key):
                    suffix = component_key[len(llama_key):]  # e.g., ".weight"
                    core_key = f"core_block.{ouro_key}{suffix}"
                    break

            if core_key is None:
                logger.warning(f"Unmapped key: {key}")
                continue

            if core_key not in averaged:
                averaged[core_key] = torch.zeros_like(value, dtype=torch.float32)
                counts[core_key] = 0

            averaged[core_key] += value.float()
            counts[core_key] += 1

    # Divide by count to get average
    for key in averaged:
        if counts[key] > 0:
            averaged[key] /= counts[key]
        logger.info(f"Averaged {key}: {counts[key]} layers, shape {averaged[key].shape}")

    return averaged


def convert_pretrained_to_ouroboros(
    model_name: str = "meta-llama/Llama-3.2-3B",
    output_dir: str = "ouroboros-600m",
    config_path: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    token: str | None = None,
) -> OuroborosModel:
    """Convert a pretrained Llama model into Ouroboros.

    Steps:
        1. Load pretrained model from HuggingFace
        2. Average all layer weights into single CoreBlock
        3. Copy embeddings and LM head
        4. Initialize Controller (zero-init) and Halter (bias-init)
        5. Save complete Ouroboros model

    Args:
        model_name: HuggingFace model name or path.
        output_dir: Directory to save the converted model.
        dtype: Data type for saved weights.
        token: HuggingFace auth token for gated models.

    Returns:
        The initialized OuroborosModel.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Converting {model_name} to Ouroboros[/bold cyan]\n")

    # Step 1: Load pretrained model
    console.print("[yellow]Loading pretrained model...[/yellow]")
    pretrained = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        token=token,
    )
    pretrained_sd = pretrained.state_dict()

    # Also save tokenizer
    console.print("[yellow]Saving tokenizer...[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.save_pretrained(str(output_path))

    # Step 2: Create Ouroboros config
    if config_path:
        config = OuroborosConfig.from_yaml(config_path)
    else:
        config = OuroborosConfig.default()
    console.print(f"[green]Config: vocab={config.vocab_size}, d_model={config.core_block.d_model}, "
                  f"n_layers_averaged={config.n_base_layers}[/green]")

    # Step 3: Create Ouroboros model (random init)
    console.print("[yellow]Creating Ouroboros model...[/yellow]")
    model = OuroborosModel(config)

    # Step 4: Average pretrained layer weights into CoreBlock
    console.print("[yellow]Averaging layer weights...[/yellow]")
    averaged_weights = _average_layer_weights(pretrained_sd, config.n_base_layers)

    # Step 5: Copy averaged weights into CoreBlock
    core_sd = model.core_block.state_dict()
    loaded_keys = []
    for key, value in averaged_weights.items():
        # Strip "core_block." prefix for state_dict loading
        cb_key = key.replace("core_block.", "")
        if cb_key in core_sd:
            core_sd[cb_key] = value.to(core_sd[cb_key].dtype)
            loaded_keys.append(cb_key)
        else:
            logger.warning(f"Key {cb_key} not found in CoreBlock state dict")

    model.core_block.load_state_dict(core_sd)
    console.print(f"[green]Loaded {len(loaded_keys)} weight tensors into CoreBlock[/green]")

    # Step 6: Copy embeddings
    embed_key = "model.embed_tokens.weight"
    if embed_key in pretrained_sd:
        model.embed_tokens.weight.data.copy_(pretrained_sd[embed_key])
        console.print("[green]Copied embedding weights[/green]")

    # Step 7: Copy LM head
    lm_head_key = "lm_head.weight"
    if lm_head_key in pretrained_sd:
        model.lm_head.weight.data.copy_(pretrained_sd[lm_head_key])
        console.print("[green]Copied LM head weights[/green]")

    # Step 8: Copy final norm
    norm_key = "model.norm.weight"
    if norm_key in pretrained_sd:
        model.norm.weight.data.copy_(pretrained_sd[norm_key])
        console.print("[green]Copied final norm weights[/green]")

    # Step 9: Controller and Halter are already initialized correctly
    # Controller: zero-init coefficient projections (done in ControllerHead.__init__)
    # Halter: bias-init to -2.0 (done in Halter.__init__)
    console.print("[green]Controller zero-initialized, Halter bias-initialized[/green]")

    # Step 10: Convert to target dtype and save
    model = model.to(dtype)

    # Save model state dict
    console.print("[yellow]Saving model...[/yellow]")
    torch.save(model.state_dict(), str(output_path / "model.pt"))

    # Save config
    config.to_yaml(str(output_path / "config.yaml"))

    # Save config as JSON too
    with open(output_path / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Print summary
    param_summary = model.param_summary()
    console.print("\n[bold green]Conversion complete![/bold green]")
    console.print(f"Output directory: {output_path}")
    console.print("\n[bold]Parameter Summary:[/bold]")
    for name, count in param_summary.items():
        console.print(f"  {name}: {count}")

    # Cleanup pretrained model
    del pretrained
    del pretrained_sd
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return model


def load_ouroboros(
    model_dir: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[OuroborosModel, OuroborosConfig]:
    """Load an Ouroboros model from disk.

    Args:
        model_dir: Directory containing model.pt and config.yaml.
        device: Device to load model onto.
        dtype: Data type for model weights.

    Returns:
        Tuple of (model, config).
    """
    model_path = Path(model_dir)

    # Load config
    config = OuroborosConfig.from_yaml(str(model_path / "config.yaml"))

    # Create model and load weights
    model = OuroborosModel(config)
    state_dict = torch.load(
        str(model_path / "model.pt"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=dtype)

    console.print(f"[green]Loaded Ouroboros from {model_dir}[/green]")
    return model, config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert pretrained model to Ouroboros")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B", help="HuggingFace model name")
    parser.add_argument("--output", default="ouroboros-600m", help="Output directory")
    parser.add_argument("--config", default=None, help="Ouroboros config YAML path")
    parser.add_argument("--token", default=None, help="HuggingFace auth token")
    args = parser.parse_args()

    convert_pretrained_to_ouroboros(
        model_name=args.model,
        output_dir=args.output,
        config_path=args.config,
        token=args.token,
    )
