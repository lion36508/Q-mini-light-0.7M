from __future__ import annotations

import json
from pathlib import Path

import pytest

from qmini_addsub.config import AddSubConfig, MAX_PARAMETER_BUDGET
from qmini_addsub.model import AddSubLanguageModel
from qmini_addsub.preflight import run_training_preflight


ROOT = Path(__file__).resolve().parents[1]


def test_parameter_estimator_matches_legacy_compact_graph():
    config = AddSubConfig()
    model = AddSubLanguageModel(config, seed=7)
    assert config.estimated_parameter_count() == 760_340
    assert model.parameter_count() == config.estimated_parameter_count()


def test_active_two_million_profile_is_under_the_hard_cap():
    path = ROOT / "configs" / "qmini_addsub_2m.json"
    config = AddSubConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    model = AddSubLanguageModel(config, seed=7)
    assert config.d_model == 128
    assert config.d_hidden == 448
    assert config.n_layers == 4
    assert config.max_context == 512
    assert config.estimated_parameter_count() == 1_985_684
    assert model.parameter_count() == 1_985_684
    assert model.parameter_count() <= MAX_PARAMETER_BUDGET


def test_config_rejects_profiles_that_exceed_two_million_parameters():
    with pytest.raises(ValueError, match="exceeding its"):
        AddSubConfig(d_model=128, d_hidden=500, n_layers=4).validate()
    with pytest.raises(ValueError, match="parameter_budget"):
        AddSubConfig(parameter_budget=MAX_PARAMETER_BUDGET + 1).validate()


def test_training_preflight_runs_a_real_optimizer_step_without_writing_artifacts(tmp_path):
    config = AddSubConfig(
        vocab_size=300,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
    )
    config_path = tmp_path / "profile.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    report = run_training_preflight(
        config_path,
        device_name="cpu",
        num_threads=1,
        batch_size=1,
        sequence_length=8,
    )
    assert report["status"] == "training_engine_preflight_passed"
    assert report["optimizer_step_completed"] is True
    assert report["parameter_count"] == config.estimated_parameter_count()
    assert report["synthetic_preflight_tokens_are_not_training_data"] is True
    assert {path.name for path in tmp_path.iterdir()} == {"profile.json"}
