"""Tests for model conversion (using tiny config, no HuggingFace download)."""

import pytest
import torch

from ouroboros.config import OuroborosConfig
from ouroboros.model import OuroborosModel


class TestConvertLogic:
    """Test conversion logic without requiring pretrained model download."""

    def test_model_from_config(self, tiny_config):
        """Model should be constructible from config."""
        model = OuroborosModel(tiny_config)
        assert model is not None

    def test_controller_zero_init_after_creation(self, tiny_config):
        """Controller coefficient projections should be zero after model creation."""
        model = OuroborosModel(tiny_config)
        for name, head in model.controller.heads.items():
            assert head.coeff_proj_A.weight.abs().max().item() == 0.0, (
                f"{name}: coeff_proj_A not zero"
            )
            assert head.coeff_proj_B.weight.abs().max().item() == 0.0, (
                f"{name}: coeff_proj_B not zero"
            )

    def test_halter_bias_init(self, tiny_config):
        """Halter final layer bias should be initialized to halt_bias_init."""
        model = OuroborosModel(tiny_config)
        final_bias = model.halter.net[-1].bias.item()
        expected = tiny_config.halter.halt_bias_init
        assert final_bias == pytest.approx(expected, abs=1e-6)

    def test_save_load_roundtrip(self, tiny_config, tmp_path):
        """Model should survive save/load roundtrip."""
        model = OuroborosModel(tiny_config)

        # Save
        torch.save(model.state_dict(), str(tmp_path / "model.pt"))
        tiny_config.to_yaml(str(tmp_path / "config.yaml"))

        # Load
        config2 = OuroborosConfig.from_yaml(str(tmp_path / "config.yaml"))
        model2 = OuroborosModel(config2)
        sd = torch.load(str(tmp_path / "model.pt"), weights_only=True)
        model2.load_state_dict(sd)

        # Verify outputs match
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        with torch.no_grad():
            out1 = model(input_ids)
            out2 = model2(input_ids)
        torch.testing.assert_close(out1["logits"], out2["logits"])

    def test_config_yaml_roundtrip(self, tiny_config, tmp_path):
        """Config should survive YAML save/load."""
        path = str(tmp_path / "config.yaml")
        tiny_config.to_yaml(path)
        loaded = OuroborosConfig.from_yaml(path)
        assert loaded.vocab_size == tiny_config.vocab_size
        assert loaded.core_block.d_model == tiny_config.core_block.d_model
        assert loaded.lora.rank == tiny_config.lora.rank
        assert loaded.controller.style_dim == tiny_config.controller.style_dim

    def test_lora_param_shapes(self, tiny_config):
        shapes = tiny_config.lora_param_shapes()
        assert len(shapes) == 7
        for name, (a_shape, b_shape) in shapes.items():
            assert len(a_shape) == 2
            assert len(b_shape) == 2
            assert a_shape[0] == tiny_config.lora.rank  # rank is first dim of A

    def test_total_lora_params(self, tiny_config):
        total = tiny_config.total_lora_params_per_step()
        assert total > 0
        assert isinstance(total, int)

    def test_weight_mapping_covers_all_targets(self):
        """LLAMA_TO_CORE_MAP should cover all 7 LoRA targets + 2 norms."""
        from ouroboros.convert import LLAMA_TO_CORE_MAP
        expected_targets = {
            "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "input_layernorm", "post_attention_layernorm",
        }
        assert set(LLAMA_TO_CORE_MAP.keys()) == expected_targets
