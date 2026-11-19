import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from qmini_addsub.config import AddSubConfig
from qmini_addsub.data import accept_oasst1_message, stable_split
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
from qmini_addsub.sft import ConversationDataset


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


def test_group_split_is_deterministic_and_group_based():
    assert stable_split("same-conversation") == stable_split("same-conversation")
    assert stable_split("same-conversation") in {"train", "validation", "test"}
    with pytest.raises(ValueError):
        stable_split("")


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


def test_default_addsub_model_parameter_count_is_stable():
    model = AddSubLanguageModel(AddSubConfig())
    assert model.parameter_count() == 636168


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
