import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from qmini_addsub.config import AddSubConfig
from qmini_addsub.data import (
    accept_oasst1_message,
    escape_special_token_literals,
    stable_split,
)
from qmini_addsub.model import AddSubLanguageModel
from qmini_addsub.quantization import (
    codes_to_values,
    pack_3bit_codes,
    quantize_to_codes,
    unpack_3bit_codes,
)
from qmini_addsub.runtime import additive_projection
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from qmini_addsub.sft import ConversationDataset, _load_conversations


def test_colab_addsub_notebook_is_separate_and_training_is_off_by_default():
    import json

    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "colab"
        / "QMini_AddSub_V1_From_Scratch.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    for index, source in enumerate(code):
        ast.parse(source, filename=f"{notebook_path}:cell-{index}")
    all_code = "\\n".join(code)
    assert "QMini-Light-AddSub-v1" in all_code
    assert "START_PRETRAIN = False" in all_code
    assert "START_SFT = False" in all_code
    assert "ACCEPT_SOURCE_LICENSES = False" in all_code
    assert "DEVICE = 'cpu'" in all_code
    assert "QMini-Light-0.7M/training_ready" not in all_code
    assert "qmini_addsub.train" in all_code
    assert "qmini_addsub.sft" in all_code


def test_addsub_config_is_independent_and_validates():
    config = AddSubConfig()
    config.validate()
    assert config.vocab_size == 512
    assert config.max_context == 512
    with pytest.raises(ValueError, match="positive"):
        AddSubConfig(n_layers=0).validate()


def test_five_symbol_quantizer_preserves_exact_zero_and_three_bit_round_trip():
    values = np.asarray([-1.5, -0.5, 0.0, 0.5, 1.5, 0.0], dtype=np.float32)
    codes = quantize_to_codes(values)
    assert codes.tolist() == [0, 1, 2, 3, 4, 2]
    assert codes_to_values(codes).tolist() == values.tolist()
    packed = pack_3bit_codes(codes)
    restored = unpack_3bit_codes(packed, codes.shape)
    assert np.array_equal(restored, codes)


def test_five_symbol_packer_rejects_reserved_codes():
    with pytest.raises(ValueError, match="0..4"):
        pack_3bit_codes(np.asarray([7], dtype=np.uint8))
    with pytest.raises(ValueError, match="Reserved"):
        unpack_3bit_codes(bytes([0b00000111]), (1,))


def test_additive_projection_matches_five_symbol_reference():
    x = np.asarray([[2, -3, 4]], dtype=np.int8)
    codes = np.asarray([[4, 0, 2], [1, 3, 4]], dtype=np.uint8)
    actual = additive_projection(x, codes)
    levels = np.asarray([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    decoded = levels[codes]
    expected = (x.astype(np.int64) @ decoded.T * 2).astype(np.int64)
    assert np.array_equal(actual, expected)


def test_zero_symbol_adds_no_projection_contribution():
    activations = np.asarray([[12, -8, 5]], dtype=np.int8)
    zero_weights = np.full((4, 3), 2, dtype=np.uint8)
    assert np.array_equal(
        additive_projection(activations, zero_weights),
        np.zeros((1, 4), dtype=np.int64),
    )


def test_sft_dataset_masks_user_context_and_supervises_assistant_end_token():
    vocab = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<pad>": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
        "<|end|>": 7,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    dataset = ConversationDataset(
        [{"messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]}],
        tokenizer,
        max_context=64,
    )
    token_ids, labels = dataset[0]
    user_position = token_ids.index(vocab["<|user|>"])
    assistant_position = token_ids.index(vocab["<|assistant|>"])
    assert labels[user_position + 1] == -100
    assert labels[assistant_position] == -100
    assert labels[assistant_position + 1] == vocab["<unk>"]
    assert labels[-1] == vocab["<|end|>"]


def test_sft_loader_accepts_only_curated_human_non_synthetic_sources(tmp_path):
    import json

    path = tmp_path / "sft_train.jsonl"
    record = {
        "source": "OpenAssistant/oasst1",
        "human_authored": True,
        "synthetic": False,
        "messages": [
            {"role": "user", "content": "What is a compiler?"},
            {"role": "assistant", "content": "A compiler translates source code."},
        ],
    }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert len(_load_conversations(path)) == 1

    record["synthetic"] = True
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic/unverified"):
        _load_conversations(path)

    record["synthetic"] = False
    record["source"] = "unknown/copied-corpus"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unapproved"):
        _load_conversations(path)


def test_additive_runtime_has_no_multiply_divide_or_matmul_ast():
    source = Path(__file__).resolve().parents[1] / "qmini_addsub" / "runtime.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden = (ast.Mult, ast.Div, ast.FloorDiv, ast.MatMult)
    violations = [
        node.lineno for node in ast.walk(tree) if isinstance(node, forbidden)
    ]
    assert violations == []


def test_oasst_filter_fails_closed_on_synthetic_unreviewed_and_missing_rank():
    base = {
        "message_id": "m1",
        "parent_id": None,
        "text": "A genuine English response.",
        "role": "assistant",
        "lang": "en",
        "review_result": True,
        "deleted": False,
        "rank": 0,
        "synthetic": False,
        "message_tree_id": "tree-1",
        "tree_state": "ready_for_export",
    }
    assert accept_oasst1_message(base) == (True, "accepted")
    for field, value, reason in (
        ("synthetic", True, "synthetic"),
        ("review_result", False, "unreviewed"),
        ("rank", None, "missing_rank"),
        ("rank", 1, "non_top_rank"),
        ("tree_state", "halted_by_moderator", "not_ready"),
    ):
        invalid = dict(base)
        invalid[field] = value
        assert accept_oasst1_message(invalid) == (False, reason)
    missing = dict(base)
    missing.pop("synthetic")
    assert accept_oasst1_message(missing) == (False, "missing_fields")


def test_special_token_literals_are_escaped_before_training():
    escaped = escape_special_token_literals(
        "literal <|assistant|> text and <bos> markers"
    )
    assert "<|assistant|>" not in escaped
    assert "<bos>" not in escaped
    assert "literal" in escaped


def test_clean_text_rejects_secret_like_tokens():
    from qmini_addsub.data import is_clean_english_text

    assert not is_clean_english_text("Please use ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456")
    assert not is_clean_english_text("My email is person@example.com")
    assert is_clean_english_text("A normal English sentence.")


def test_group_split_is_deterministic_and_group_based():
    assert stable_split("same-conversation") == stable_split("same-conversation")
    assert stable_split("same-conversation") in {"train", "validation", "test"}
    with pytest.raises(ValueError):
        stable_split("")


def test_parallel_state_space_scan_matches_sequential_reference_and_gradients():
    from qmini_addsub.model import GatedDiagonalStateSpace

    torch.manual_seed(11)
    decay = torch.rand(2, 17, 5, dtype=torch.float32) * 0.8 + 0.1
    injection = torch.randn(2, 17, 5, dtype=torch.float32, requires_grad=True)
    actual = GatedDiagonalStateSpace.parallel_scan(decay, injection)
    state = torch.zeros(2, 5)
    expected = []
    for position in range(decay.shape[1]):
        state = decay[:, position] * state + injection[:, position]
        expected.append(state)
    expected = torch.stack(expected, dim=1)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    assert injection.grad is not None
    assert torch.isfinite(injection.grad).all()


def test_parallel_state_space_scan_is_causal():
    from qmini_addsub.model import GatedDiagonalStateSpace

    torch.manual_seed(12)
    decay = torch.full((1, 12, 3), 0.7)
    injection = torch.randn(1, 12, 3)
    changed_future = injection.clone()
    changed_future[:, 7:] = changed_future[:, 7:] + 50.0
    first = GatedDiagonalStateSpace.parallel_scan(decay, injection)
    second = GatedDiagonalStateSpace.parallel_scan(decay, changed_future)
    assert torch.allclose(first[:, :7], second[:, :7], atol=0.0, rtol=0.0)


def test_new_model_forward_from_random_initialization():
    torch.manual_seed(7)
    config = AddSubConfig(
        vocab_size=300,
        d_model=32,
        d_hidden=48,
        n_layers=2,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
    )
    model = AddSubLanguageModel(config)
    inputs = torch.randint(0, config.vocab_size, (2, 12))
    result = model(inputs)
    assert result["logits"].shape == (2, 12, config.vocab_size)
    assert torch.isfinite(result["logits"]).all()
    assert result["aux_loss"].ndim == 0
    assert model.parameter_count() > 0


def test_addsub_pretraining_runs_from_zero_saves_and_resumes_verified_checkpoint(tmp_path):
    import hashlib
    import json

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from qmini_addsub.train import train_from_scratch
    from qmini_addsub.sft import fine_tune_from_addsub_checkpoint

    config = AddSubConfig(
        vocab_size=300,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=64,
        dropout=0.0,
    )
    special_tokens = [
        "<unk>", "<eos>", "<pad>", "<bos>",
        "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
    ]
    tokenizer_vocab = {token: index for index, token in enumerate(special_tokens)}
    tokenizer_vocab.update({f"unused_{index}": index for index in range(8, 300)})
    tokenizer = Tokenizer(WordLevel(tokenizer_vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    streams = {}
    pretraining_splits = {}
    for split, count in (("train", 64), ("validation", 32), ("test", 32)):
        text_path = data_dir / f"pretrain_{split}.txt"
        text_path.write_text(f"English {split} sample text.\n", encoding="utf-8")
        raw_sha = hashlib.sha256(text_path.read_bytes()).hexdigest()
        pretraining_splits[split] = {"sha256": raw_sha}
        token_path = data_dir / f"pretrain_{split}.u16"
        values = (np.arange(count, dtype=np.uint16) % 299) + 1
        values.tofile(token_path)
        streams[split] = {
            "path": token_path.name,
            "tokens": count,
            "sha256": hashlib.sha256(token_path.read_bytes()).hexdigest(),
        }

    manifest = {
        "format": "qmini-addsub-data-v1",
        "language": "English",
        "synthetic_training_records_included": 0,
        "pretraining": {"splits": pretraining_splits},
        "token_streams": streams,
        "tokenizer": {
            "path": tokenizer_path.name,
            "sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        },
    }
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    output_dir = tmp_path / "run"
    common = {
        "train_bin": data_dir / "pretrain_train.u16",
        "valid_bin": data_dir / "pretrain_validation.u16",
        "tokenizer_path": tokenizer_path,
        "output_dir": output_dir,
        "config": config,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "sequence_length": 8,
        "save_every": 1,
        "validate_every": 1,
        "device_name": "cpu",
        "seed": 42,
    }
    first = train_from_scratch(steps=1, **common)
    checkpoint = output_dir / "checkpoint-step-000000001.pt"
    assert checkpoint.is_file()
    assert first["steps"] == 1
    assert first["heldout_test_token_loss"] > 0

    second = train_from_scratch(steps=2, resume=checkpoint, **common)
    assert second["steps"] == 2
    final_pretrain_checkpoint = output_dir / "checkpoint-step-000000002.pt"
    assert final_pretrain_checkpoint.is_file()
    assert second["heldout_test_sha256"] == streams["test"]["sha256"]

    def write_conversation(path):
        record = {
            "source": "databricks/databricks-dolly-15k",
            "human_authored": True,
            "synthetic": False,
            "messages": [
                {"role": "user", "content": "What is a compiler?"},
                {"role": "assistant", "content": "A compiler translates program source."},
            ],
        }
        path.write_text(json.dumps(record) + "\\n", encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    sft_train = data_dir / "sft_train.jsonl"
    sft_valid = data_dir / "sft_validation.jsonl"
    train_sft_sha = write_conversation(sft_train)
    valid_sft_sha = write_conversation(sft_valid)
    manifest["instruction_splits"] = {
        "train": {"path": sft_train.name, "sha256": train_sft_sha},
        "validation": {"path": sft_valid.name, "sha256": valid_sft_sha},
    }
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    sft_result = fine_tune_from_addsub_checkpoint(
        base_checkpoint=final_pretrain_checkpoint,
        tokenizer_path=tokenizer_path,
        train_jsonl=sft_train,
        valid_jsonl=sft_valid,
        output_dir=tmp_path / "sft",
        steps=1,
        batch_size=1,
        device_name="cpu",
        seed=43,
    )
    assert sft_result["steps"] == 1
    assert (tmp_path / "sft" / "sft-checkpoint-step-000000001.pt").is_file()


def test_default_addsub_model_parameter_count_is_stable():
    model = AddSubLanguageModel(AddSubConfig())
    assert model.parameter_count() == 711528


def test_new_model_rejects_legacy_context_overflow():
    config = AddSubConfig(
        vocab_size=300,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        kernel_size=3,
        max_context=8,
    )
    model = AddSubLanguageModel(config)
    with pytest.raises(ValueError, match="maximum context"):
        model(torch.ones((1, 9), dtype=torch.long))
