from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from qmini_addsub.config import AddSubConfig
from qmini_addsub.initialize import (
    INITIAL_MODEL_FORMAT,
    build_bootstrap_tokenizer,
    initialize_model_artifacts,
    load_initial_model_checkpoint,
    verify_model_bundle,
)
from qmini_addsub.model import AddSubLanguageModel


PROFILE = Path(__file__).resolve().parents[1] / "configs" / "qmini_addsub_2m.json"


def test_bootstrap_tokenizer_is_dataset_independent_and_round_trips_utf8() -> None:
    tokenizer = build_bootstrap_tokenizer()
    assert tokenizer.get_vocab_size(with_added_tokens=True) == 264
    for token in ("<unk>", "<pad>", "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"):
        assert tokenizer.token_to_id(token) is not None

    sample = "English only: café, punctuation, and UTF-8 ✓"
    encoded = tokenizer.encode(sample, add_special_tokens=False)
    assert tokenizer.decode(encoded.ids, skip_special_tokens=False) == sample


def test_model_first_initialization_creates_complete_untrained_bundle(tmp_path: Path) -> None:
    root = tmp_path / "drive" / "trainability_addsub_2m_v1"
    result = initialize_model_artifacts(root, PROFILE, threads=2)

    artifacts = root / "artifacts"
    assert result["stage"] == "initialized_untrained"
    assert result["parameter_count"] == 1_887_644
    assert result["training"]["status"] == "not_started"
    assert result["dataset"]["downloaded_by_initializer"] is False

    for name in (
        "tokenizer.json",
        "model_config.json",
        "initial_model.pt",
        "model_manifest.json",
        "initialization_report.json",
        "MODEL_CARD.md",
    ):
        assert (artifacts / name).is_file()

    assert (root / "data").is_dir()
    assert not any((root / "data").iterdir())
    assert (root / "runs" / "pretrain").is_dir()
    assert (root / "runs" / "sft").is_dir()
    assert (root / "evaluation").is_dir()

    config = AddSubConfig.from_dict(
        json.loads((artifacts / "model_config.json").read_text(encoding="utf-8"))
    )
    tokenizer = Tokenizer.from_file(str(artifacts / "tokenizer.json"))
    assert config.vocab_size == tokenizer.get_vocab_size(with_added_tokens=True) == 264
    assert config.max_context == 512

    payload = torch.load(artifacts / "initial_model.pt", map_location="cpu", weights_only=True)
    assert payload["format"] == INITIAL_MODEL_FORMAT
    assert payload["stage"] == "initialized_untrained"
    assert payload["step"] == 0

    reloaded = AddSubLanguageModel(config, seed=config.seed)
    loaded_sha = load_initial_model_checkpoint(
        reloaded,
        artifacts / "initial_model.pt",
        payload["tokenizer_sha256"],
    )
    assert loaded_sha == result["initial_model_sha256"]
    assert verify_model_bundle(root)["parameter_count"] == result["parameter_count"]


def test_model_first_initialization_is_idempotent_and_preserves_existing_bundle(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    first = initialize_model_artifacts(root, PROFILE, threads=2)
    checkpoint = root / "artifacts" / "initial_model.pt"
    original_bytes = checkpoint.read_bytes()

    second = initialize_model_artifacts(root, PROFILE, threads=2)

    assert second["initial_model_sha256"] == first["initial_model_sha256"]
    assert checkpoint.read_bytes() == original_bytes


def test_partial_model_bundle_is_never_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(parents=True)
    marker = artifact_dir / "tokenizer.json"
    marker.write_text("user-data-preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        initialize_model_artifacts(root, PROFILE, threads=2)

    assert marker.read_text(encoding="utf-8") == "user-data-preserve"
