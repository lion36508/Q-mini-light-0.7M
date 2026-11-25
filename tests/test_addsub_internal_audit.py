import hashlib

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from qmini_addsub.sft import (
    CHAT_SPECIAL_TOKENS,
    _chat_special_token_ids,
    _validate_base_tokenizer_fingerprint,
)


def _tokenizer(vocab=None):
    if vocab is None:
        vocab = {token: index for index, token in enumerate(CHAT_SPECIAL_TOKENS)}
        vocab["hello"] = len(vocab)
    return Tokenizer(WordLevel(vocab, unk_token="<unk>"))


def test_sft_chat_token_contract_requires_all_unique_control_ids():
    tokenizer = _tokenizer()
    ids = _chat_special_token_ids(tokenizer)
    assert tuple(ids) == CHAT_SPECIAL_TOKENS
    assert len(set(ids.values())) == len(CHAT_SPECIAL_TOKENS)


def test_sft_chat_token_contract_rejects_missing_or_aliased_tokens():
    missing = _tokenizer({
        token: index
        for index, token in enumerate(CHAT_SPECIAL_TOKENS[:-1])
    })
    with pytest.raises(ValueError, match="missing required"):
        _chat_special_token_ids(missing)

    aliased_vocab = {
        token: index
        for index, token in enumerate(CHAT_SPECIAL_TOKENS)
    }
    aliased_vocab["<|assistant|>"] = aliased_vocab["<|user|>"]
    aliased = _tokenizer(aliased_vocab)
    with pytest.raises(ValueError, match="IDs must be unique"):
        _chat_special_token_ids(aliased)


def test_sft_base_checkpoint_must_match_exact_tokenizer_fingerprint(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text('{"vocab":"one"}', encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _validate_base_tokenizer_fingerprint(
        {"tokenizer_sha256": digest}, path
    ) == digest

    path.write_text('{"vocab":"two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the pretraining checkpoint"):
        _validate_base_tokenizer_fingerprint(
            {"tokenizer_sha256": digest}, path
        )


@pytest.mark.parametrize("fingerprint", [None, "", "xyz", "A" * 64])
def test_sft_base_checkpoint_requires_valid_lowercase_sha256(tmp_path, fingerprint):
    path = tmp_path / "tokenizer.json"
    path.write_text("tokenizer", encoding="utf-8")
    with pytest.raises(ValueError, match="valid tokenizer SHA-256"):
        _validate_base_tokenizer_fingerprint(
            {"tokenizer_sha256": fingerprint}, path
        )

import hashlib
import pytest


def test_dataset_prepare_is_transactional_on_upstream_failure(tmp_path, monkeypatch):
    from scripts import prepare_qmini_addsub_data as module

    output = tmp_path / "dataset"
    output.mkdir()

    def first_source(_cache, stage):
        (stage / "partial.txt").write_text("partial", encoding="utf-8")
        return {"splits": {}}

    def broken_source(_cache):
        raise RuntimeError("source unavailable")

    monkeypatch.setattr(module, "prepare_simple_wikipedia", first_source)
    monkeypatch.setattr(module, "load_oasst_conversations", broken_source)

    with pytest.raises(RuntimeError, match="source unavailable"):
        module.prepare_dataset(output, tmp_path / "cache")

    assert output.is_dir() and list(output.iterdir()) == []
    assert list(tmp_path.glob(".dataset.staging-*")) == []


def test_dataset_prepare_refuses_to_overwrite_existing_data(tmp_path, monkeypatch):
    from scripts import prepare_qmini_addsub_data as module

    output = tmp_path / "dataset"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        module, "prepare_simple_wikipedia",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    with pytest.raises(FileExistsError, match="not empty"):
        module.prepare_dataset(output, tmp_path / "cache")
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_dataset_prepare_rejects_cache_nested_under_output(tmp_path):
    from scripts import prepare_qmini_addsub_data as module

    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="outside the output"):
        module.prepare_dataset(output, output / ".cache")


def test_simplewiki_streams_batches_and_deduplicates_content(tmp_path, monkeypatch):
    from scripts import prepare_qmini_addsub_data as module

    source = tmp_path / "source.parquet"
    source.write_bytes(b"mock")
    monkeypatch.setattr(module, "hf_hub_download", lambda **kwargs: str(source))
    monkeypatch.setattr(module, "SIMPLE_WIKI_SIZE", source.stat().st_size)
    monkeypatch.setattr(module, "SIMPLE_WIKI_SHA256", hashlib.sha256(b"mock").hexdigest())
    monkeypatch.setattr(
        module, "stable_split",
        lambda page, seed: {"1": "train", "2": "validation", "3": "test", "4": "test"}[page],
    )

    rows = [
        {"page_id": "1", "title": "One article", "content_no_link": "A sufficiently long article body one.", "error": False},
        {"page_id": "2", "title": "Two article", "content_no_link": "A sufficiently long article body two.", "error": False},
        {"page_id": "3", "title": "Three article", "content_no_link": "A sufficiently long article body three.", "error": False},
        {"page_id": "4", "title": "One article", "content_no_link": "A sufficiently long article body one.", "error": False},
    ]

    class Batch:
        def __init__(self, values):
            self.values = values
        def to_pylist(self):
            return self.values

    class Reader:
        def __init__(self, path):
            assert path == source
        def iter_batches(self, *, batch_size, columns):
            assert batch_size == 512
            yield Batch(rows[:2])
            yield Batch(rows[2:])

    monkeypatch.setattr(module.pq, "ParquetFile", Reader)
    output = tmp_path / "stage"
    output.mkdir()
    report = module.prepare_simple_wikipedia(tmp_path / "cache", output)
    assert [report["splits"][x]["articles"] for x in ("train", "validation", "test")] == [1, 1, 1]
    assert report["dropped"]["duplicate_article"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dropout": False},
        {"moe_balance_coef": True},
        {"router_z_loss_coef": False},
        {"residual_init": "0.1"},
        {"dropout": float("nan")},
        {"moe_balance_coef": float("inf")},
    ],
)
def test_addsub_config_rejects_boolean_string_and_nonfinite_scalars(kwargs):
    from qmini_addsub.config import AddSubConfig

    with pytest.raises((TypeError, ValueError)):
        AddSubConfig(**kwargs).validate()


@pytest.mark.parametrize(
    "learning_rate,weight_decay",
    [
        (True, 0.01),
        (1e-3, True),
        ("0.001", 0.01),
        (float("nan"), 0.01),
    ],
)
def test_adamw_rejects_boolean_string_and_nonfinite_hyperparameters(
    learning_rate, weight_decay
):
    import torch
    from qmini_addsub.optim import build_adamw

    model = torch.nn.Linear(2, 2)
    with pytest.raises((TypeError, ValueError)):
        build_adamw(model, learning_rate, weight_decay)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"peak_learning_rate": True},
        {"peak_learning_rate": "0.001"},
        {"warmup_fraction": True},
        {"minimum_ratio": False},
        {"minimum_ratio": float("nan")},
    ],
)
def test_cosine_schedule_rejects_boolean_string_and_nonfinite_scalars(kwargs):
    from qmini_addsub.optim import cosine_warmup_lr

    values = {
        "step": 1,
        "total_steps": 100,
        "peak_learning_rate": 1e-3,
        "warmup_fraction": 0.03,
        "minimum_ratio": 0.1,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        cosine_warmup_lr(**values)


def test_tokenizer_builder_refuses_to_replace_existing_tokenizer(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys

    from scripts import train_qmini_addsub_tokenizer as builder

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / "pretrain_train.txt"
    source.write_text("A valid English sentence.\n", encoding="utf-8")
    manifest = {
        "format": "qmini-addsub-data-v1",
        "language": "English",
        "synthetic_training_records_included": 0,
        "pretraining": {
            "splits": {
                "train": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            }
        },
        "instruction_splits": {},
    }
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    output = tmp_path / "artifacts" / "tokenizer.json"
    output.parent.mkdir()
    output.write_text("previous-tokenizer", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_qmini_addsub_tokenizer",
            "--data-dir", str(data_dir),
            "--output", str(output),
            "--vocab-size", "264",
            "--pretrain-byte-budget", "1024",
        ],
    )
    with pytest.raises(FileExistsError, match="already exists"):
        builder.main()
    assert output.read_text(encoding="utf-8") == "previous-tokenizer"
    assert not output.with_name("model_config.json").exists()


def test_weight_quantizer_rejects_nonfloat_empty_and_nonfinite_inputs():
    import torch
    from qmini_addsub.quantization import quantize_weight_ste

    with pytest.raises(TypeError, match="floating-point"):
        quantize_weight_ste(torch.ones(2, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="empty"):
        quantize_weight_ste(torch.empty(0, 2))
    with pytest.raises(ValueError, match="Non-finite"):
        quantize_weight_ste(torch.tensor([[float("nan"), 1.0]]), scale=1.0)


@pytest.mark.parametrize("scale", [True, "1.0", float("nan"), 0.0])
def test_weight_quantizer_rejects_invalid_explicit_scale(scale):
    import torch
    from qmini_addsub.quantization import quantize_weight_ste

    with pytest.raises((TypeError, ValueError)):
        quantize_weight_ste(torch.ones(2, 2), scale=scale)


def test_numpy_weight_quantizer_rejects_bool_and_non_numeric_arrays():
    import numpy as np
    from qmini_addsub.quantization import quantize_to_codes

    with pytest.raises(TypeError, match="real numeric"):
        quantize_to_codes(np.asarray([True, False]))
    with pytest.raises(TypeError, match="real numeric"):
        quantize_to_codes(np.asarray(["1.0", "0.5"]))
    with pytest.raises(TypeError, match="boolean"):
        quantize_to_codes(np.asarray([1.0]), scale=True)
