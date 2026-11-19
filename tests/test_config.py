import json
import pytest

from qmini.config import QMiniConfig


def test_default_and_light_profile_context_are_512():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert QMiniConfig().max_context == 512
    light = QMiniConfig.from_json(root / "configs" / "qmini_light_0.7m.json")
    assert light.max_context == 512


def test_new_config_defaults_and_legacy_config_fallback_preserve_archive_semantics():
    current = QMiniConfig()
    assert current.quantization_version == "q2-v3"
    assert current.quantize_embeddings is False

    legacy_values = current.to_dict()
    legacy_values.pop("quantization_version")
    legacy_values.pop("quantize_embeddings")
    legacy = QMiniConfig.from_dict(legacy_values)
    assert legacy.quantization_version == "q2-v2"
    assert legacy.quantize_embeddings is True


def test_config_rejects_incompatible_q2_v2_fp16_embedding_boundary():
    with pytest.raises(ValueError, match="quantization_version"):
        QMiniConfig(quantization_version="q2-v2", quantize_embeddings=False).validate()


def test_q2_v3_fp16_embedding_estimate_matches_actual_layout():
    values = QMiniConfig(
        vocab_size=300, d_model=32, d_hidden=48, n_layers=1,
        n_experts=1, kernel_size=3, dilations=(1,), max_context=32,
        quantization_version="q2-v3", quantize_embeddings=False,
    )
    legacy = QMiniConfig.from_dict({
        **values.to_dict(),
        "quantization_version": "q2-v2",
        "quantize_embeddings": True,
    })
    assert values.estimated_packed_export_bytes() < legacy.estimated_packed_export_bytes()


def test_config_round_trip(tmp_path):
    original = QMiniConfig(
        vocab_size=512, d_model=32, d_hidden=48, n_layers=2,
        n_experts=2, kernel_size=5, dilations=(1, 2),
        max_context=64, max_thinking_steps=2, thinking_steps=1,
    )
    path = original.save(tmp_path / "config.json")
    restored = QMiniConfig.from_json(path)
    assert restored.to_dict() == original.to_dict()


def test_unknown_config_keys_fail_loudly():
    with pytest.raises(ValueError, match="Unknown Q-Mini config keys"):
        QMiniConfig.from_dict({"vocab_size": 128, "hidden_szie": 256})


def test_dilation_count_must_match_layers():
    with pytest.raises(ValueError, match="dilations"):
        QMiniConfig.from_dict({"n_layers": 3, "dilations": [1, 2]})

@pytest.mark.parametrize(
    "values",
    [
        {"d_model": 32.5},
        {"dilations": [1.5, 2, 4, 8, 16, 32]},
        {"rms_norm_eps": float("nan")},
        {"seed": -1},
        {"max_context": 1},
    ],
)
def test_invalid_config_values_are_rejected_without_coercion(values):
    with pytest.raises((TypeError, ValueError)):
        QMiniConfig.from_dict(values)

from qmini.model import QMiniMoE


@pytest.mark.parametrize("use_state_space", [False, True])
def test_estimated_parameter_count_matches_live_model(use_state_space):
    config = QMiniConfig(
        vocab_size=128, d_model=32, d_hidden=48, n_layers=2,
        n_experts=2, kernel_size=5, dilations=(1, 2),
        max_context=64, max_thinking_steps=2, thinking_steps=1,
        use_state_space=use_state_space,
    )
    model = QMiniMoE(config)
    assert config.estimated_parameter_count() == model.parameter_report()["total"]


def test_capacity_profiles_have_exact_audited_parameter_targets():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    expected = {
        "qmini_5m.json": 5_245_996,
        "qmini_research_152m.json": 152_483_800,
        "qmini_5b.json": 5_132_630_288,
    }
    for filename, parameter_count in expected.items():
        config = QMiniConfig.from_json(root / filename)
        assert config.estimated_parameter_count() == parameter_count
        assert config.use_state_space
    qmini_5b = QMiniConfig.from_json(root / "qmini_5b.json")
    assert qmini_5b.gradient_checkpointing
    assert qmini_5b.estimated_unique_active_parameter_count() == 852_873_488
    assert qmini_5b.estimated_packed_export_bytes() == 1_294_831_200
    assert qmini_5b.estimated_packed_export_bytes() < 1_500_000_000
    assert qmini_5b.estimated_streaming_state_bytes(batch_size=1) == 80_527_360
    assert qmini_5b.estimated_cpu_inference_working_set_bytes(batch_size=1) == 1_409_085_024

    dense_reference = QMiniConfig.from_json(root / "qmini_dense_853m.json")
    assert dense_reference.estimated_parameter_count() == 852_281_344
    assert dense_reference.estimated_unique_active_parameter_count() == 852_281_344
    assert dense_reference.estimated_packed_export_bytes() < qmini_5b.estimated_packed_export_bytes()
