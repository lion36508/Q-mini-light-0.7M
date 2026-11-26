from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from qmini_addsub.config import AddSubConfig
from qmini_addsub.train import train_from_scratch
from qmini_addsub.initialize import (
    INITIAL_MODEL_FORMAT,
    build_bootstrap_tokenizer,
    initialize_model_artifacts,
    load_initial_model_checkpoint,
    verify_model_bundle,
    verify_self_contained_model_source,
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
        "parameter_layout.json",
        "model_manifest.json",
        "initialization_report.json",
        "MODEL_CARD.md",
    ):
        assert (artifacts / name).is_file()

    source_root = artifacts / "source_code"
    assert (source_root / "qmini_addsub" / "model.py").is_file()
    assert (source_root / "qmini_addsub" / "train.py").is_file()
    assert (source_root / "qmini_addsub" / "generate.py").is_file()
    assert (source_root / "qmini_addsub" / "runtime.py").is_file()
    assert (source_root / "qmini" / "__init__.py").is_file()
    assert (source_root / "scripts" / "prepare_qmini_addsub_data.py").is_file()
    assert (source_root / "configs" / "qmini_addsub_2m.json").is_file()
    source_manifest = json.loads(
        (source_root / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert source_manifest["python_file_count"] > 20
    assert result["source_verification"]["isolated_source_import"] is True
    assert result["source_verification"]["model_bundle_verified"] is True
    assert result["source_verification"]["source_package"].startswith(str(source_root))
    persisted_manifest = json.loads(
        (artifacts / "model_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted_manifest["source_verification"]["isolated_source_import"] is True
    assert len(list((source_root / "tests").glob("*.py"))) > 0

    layout = json.loads(
        (artifacts / "parameter_layout.json").read_text(encoding="utf-8")
    )
    assert layout["parameter_count"] == result["parameter_count"]
    assert layout["parameters"]
    assert sum(entry["numel"] for entry in layout["parameters"]) == result["parameter_count"]
    assert layout["parameters"][0]["flat_offset_start"] == 0
    assert layout["parameters"][-1]["flat_offset_end"] == result["parameter_count"]

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
    independent = verify_self_contained_model_source(root)
    assert independent["isolated_source_import"] is True
    assert independent["parameter_count"] == result["parameter_count"]


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



def test_pretraining_loads_the_exact_saved_model_first_weights(tmp_path: Path) -> None:
    from qmini_addsub.initialize import INITIAL_MODEL_FORMAT

    config = AddSubConfig(
        vocab_size=264,
        d_model=8,
        d_hidden=16,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=16,
        dropout=0.0,
        seed=91,
    )
    special = ["<unk>", "<pad>", "<bos>", "<eos>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    vocab = {token: index for index, token in enumerate(special)}
    vocab.update({f"byte-token-{index:03d}": index for index in range(8, 264)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    tokenizer_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()

    initial_model = AddSubLanguageModel(config, seed=config.seed)
    initial_path = tmp_path / "initial_model.pt"
    torch.save(
        {
            "format": INITIAL_MODEL_FORMAT,
            "stage": "initialized_untrained",
            "step": 0,
            "config": config.to_dict(),
            "parameter_count": initial_model.parameter_count(),
            "tokenizer_sha256": tokenizer_sha,
            "wavebit_options": None,
            "model": initial_model.state_dict(),
        },
        initial_path,
    )
    initial_sha = hashlib.sha256(initial_path.read_bytes()).hexdigest()

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    stream_entries = {}
    source_splits = {}
    for split in ("train", "validation", "test"):
        text_path = data_dir / f"pretrain_{split}.txt"
        text_path.write_text(f"English {split} data for Q-Mini.\n", encoding="utf-8")
        source_sha = hashlib.sha256(text_path.read_bytes()).hexdigest()
        source_splits[split] = {"sha256": source_sha}
        stream_path = data_dir / f"pretrain_{split}.u16"
        values = (np.arange(64, dtype=np.uint16) % 12) + 8
        values.tofile(stream_path)
        stream_entries[split] = {
            "path": stream_path.name,
            "tokens": int(values.size),
            "sha256": hashlib.sha256(stream_path.read_bytes()).hexdigest(),
            "source_sha256": source_sha,
            "source_bytes": text_path.stat().st_size,
        }

    manifest = {
        "format": "qmini-addsub-data-v1",
        "language": "English",
        "synthetic_training_records_included": 0,
        "pretraining": {"splits": source_splits},
        "token_streams": stream_entries,
        "tokenizer": {"path": tokenizer_path.name, "sha256": tokenizer_sha},
    }
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    output_dir = tmp_path / "pretrain-run"
    result = train_from_scratch(
        train_bin=data_dir / "pretrain_train.u16",
        valid_bin=data_dir / "pretrain_validation.u16",
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        config=config,
        steps=1,
        batch_size=1,
        learning_rate=1e-3,
        sequence_length=8,
        save_every=1,
        validate_every=1,
        seed=42,
        device_name="cpu",
        num_threads=1,
        initial_model=initial_path,
    )

    checkpoint = torch.load(
        output_dir / "checkpoint-step-000000001.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert result["initial_model_sha256"] == initial_sha
    assert checkpoint["run_contract"]["initial_model_sha256"] == initial_sha
    assert checkpoint["step"] == 1


def test_model_first_and_training_colab_use_the_saved_source_bundle() -> None:
    model_first_path = (
        Path(__file__).resolve().parents[1]
        / "colab" / "QMini_AddSub_2M_Model_First.ipynb"
    )
    training_path = (
        Path(__file__).resolve().parents[1]
        / "colab" / "QMini_AddSub_2M_One_Click.ipynb"
    )
    model_first = json.loads(model_first_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))

    model_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in model_first["cells"]
        if cell.get("cell_type") == "code"
    )
    training_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in training["cells"]
        if cell.get("cell_type") == "code"
    )
    ast.parse(model_code, filename=str(model_first_path))
    ast.parse(training_code, filename=str(training_path))

    model_first_markdown = "".join(model_first["cells"][0].get("source", []))
    assert "source_root=REPO_DIR" in model_code
    assert "parameter_layout.json" in model_code
    assert "source_code" in model_code
    assert "does not download datasets" in model_first_markdown

    assert "SOURCE_CODE_DIR = ARTIFACT_DIR / \"source_code\"" in training_code
    assert "shutil.copytree(SOURCE_CODE_DIR, REPO_DIR" in training_code
    assert "git clone" not in training_code
    assert "git fetch" not in training_code
    assert "source_root=REPO_DIR" in training_code
    assert "--initial-model" in training_code
