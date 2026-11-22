import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from qmini_addsub.config import AddSubConfig
from qmini_addsub.data import (
    accept_oasst1_message,
    escape_special_token_literals,
    normalize_training_text,
    stable_split,
)
from qmini_addsub.model import AddSubLanguageModel, AddSubLinear
from qmini_addsub.optim import build_adamw, cosine_warmup_lr
from qmini_addsub.quantization import (
    codes_to_values,
    pack_3bit_codes,
    quantize_to_codes,
    unpack_3bit_codes,
    quantize_weight_ste,
)
from qmini_addsub.runtime import additive_projection, saturating_add
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
    all_code = "\n".join(code)
    assert "QMini-Light-AddSub-v1" in all_code
    assert "PREPARE_PINNED_CORPUS = False" in all_code
    assert "IMPORT_USER_DATA = False" in all_code
    assert "BUILD_TOKENIZER_AND_STREAMS = False" in all_code
    assert "START_PRETRAIN = False" in all_code
    assert "START_SFT = False" in all_code
    assert "ACCEPT_SOURCE_LICENSES = False" in all_code
    assert "DEVICE = 'cpu'" in all_code
    assert "--sft-test" in all_code
    assert "--test-jsonl" in all_code
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

    tampered_padding = bytearray(packed)
    tampered_padding[-1] |= 0b10000000
    with pytest.raises(ValueError, match="padding bits"):
        unpack_3bit_codes(tampered_padding, codes.shape)
    with pytest.raises(ValueError, match="dimensions"):
        unpack_3bit_codes(packed, (True, 6))


def test_training_ste_quantizer_matches_exact_five_symbol_export_codebook():
    values = torch.tensor(
        [-2.0, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
         0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    actual = quantize_weight_ste(values)
    expected_codes = quantize_to_codes(values.detach().cpu().numpy())
    expected = torch.from_numpy(codes_to_values(expected_codes))
    assert torch.equal(actual.detach(), expected)
    actual.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


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
    first = train_from_scratch(steps=2, **common)
    checkpoint = output_dir / "checkpoint-step-000000001.pt"
    assert checkpoint.is_file()
    assert first["steps"] == 2
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
                {"role": "user", "content": f"What is a compiler in {path.stem}?"},
                {"role": "assistant", "content": f"A compiler translates program source for {path.stem}."},
            ],
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    sft_train = data_dir / "sft_train.jsonl"
    sft_valid = data_dir / "sft_validation.jsonl"
    sft_test = data_dir / "sft_test.jsonl"
    train_sft_sha = write_conversation(sft_train)
    valid_sft_sha = write_conversation(sft_valid)
    test_sft_sha = write_conversation(sft_test)
    manifest["instruction_splits"] = {
        "train": {"path": sft_train.name, "sha256": train_sft_sha},
        "validation": {"path": sft_valid.name, "sha256": valid_sft_sha},
        "test": {"path": sft_test.name, "sha256": test_sft_sha},
    }
    manifest["approved_instruction_sources"] = ["databricks/databricks-dolly-15k"]
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    sft_result = fine_tune_from_addsub_checkpoint(
        base_checkpoint=final_pretrain_checkpoint,
        tokenizer_path=tokenizer_path,
        train_jsonl=sft_train,
        valid_jsonl=sft_valid,
        test_jsonl=sft_test,
        output_dir=tmp_path / "sft",
        steps=2,
        batch_size=1,
        save_every=1,
        device_name="cpu",
        seed=43,
    )
    assert sft_result["steps"] == 2
    assert sft_result["heldout_test_response_loss"] > 0
    assert (tmp_path / "sft" / "sft-checkpoint-step-000000001.pt").is_file()

    sft_checkpoint = tmp_path / "sft" / "sft-checkpoint-step-000000001.pt"
    sft_state = torch.load(sft_checkpoint, map_location="cpu", weights_only=True)
    assert sft_state["format"] == "qmini-addsub-sft-v4"
    assert "torch_rng" in sft_state and "python_rng" in sft_state

    resumed_sft = fine_tune_from_addsub_checkpoint(
        base_checkpoint=final_pretrain_checkpoint,
        tokenizer_path=tokenizer_path,
        train_jsonl=sft_train,
        valid_jsonl=sft_valid,
        test_jsonl=sft_test,
        output_dir=tmp_path / "sft-resumed",
        steps=2,
        batch_size=1,
        save_every=1,
        device_name="cpu",
        seed=43,
        resume=sft_checkpoint,
    )
    assert resumed_sft["steps"] == 2
    assert resumed_sft["final_validation_response_loss"] > 0
    assert (tmp_path / "sft-resumed" / "sft-checkpoint-step-000000002.pt").is_file()


def test_default_addsub_model_parameter_count_is_stable():
    model = AddSubLanguageModel(AddSubConfig())
    assert model.parameter_count() == 710676


def test_swglu_experts_and_bounded_residual_gates_are_active():
    config = AddSubConfig()
    model = AddSubLanguageModel(config)
    block = model.blocks[0]
    assert block.conv_residual_logit.item() == 0.0
    assert block.state_residual_logit.item() == 0.0
    assert block.expert_residual_logit.item() == 0.0
    for expert in block.experts.experts:
        assert expert.up_gate.out_features == config.d_hidden * 2
        assert expert.down.in_features == config.d_hidden
    inputs = torch.randint(0, config.vocab_size, (2, 32))
    output = model(inputs)
    assert torch.isfinite(output["logits"]).all()
    assert torch.isfinite(output["aux_loss"])


def test_optimizer_uses_no_decay_for_vectors_and_cosine_warmup_is_bounded():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.LayerNorm(8),
    )
    optimizer = build_adamw(model, learning_rate=1e-3, weight_decay=0.1)
    decay_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    no_decay_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    assert id(model[0].weight) in decay_ids
    assert id(model[0].bias) in no_decay_ids
    assert id(model[1].weight) in no_decay_ids
    assert id(model[1].bias) in no_decay_ids
    assert optimizer.param_groups[1]["weight_decay"] == 0.0

    assert cosine_warmup_lr(1, 100, 1e-3) < 1e-3
    assert cosine_warmup_lr(3, 100, 1e-3) == 1e-3
    assert cosine_warmup_lr(100, 100, 1e-3) == 1e-4


def test_default_model_initialization_is_live_and_five_symbol_aligned():
    torch.manual_seed(29)
    model = AddSubLanguageModel(AddSubConfig())
    zero_decay_projections = {
        id(block.state_space.decay_projection) for block in model.blocks
    }
    linear_count = 0
    for module in model.modules():
        if not isinstance(module, AddSubLinear):
            continue
        linear_count += 1
        quantized = quantize_weight_ste(module.weight).detach()
        if id(module) in zero_decay_projections:
            assert torch.count_nonzero(quantized) == 0
            continue
        assert torch.isin(
            quantized,
            torch.tensor([-1.5, -0.5, 0.0, 0.5, 1.5]),
        ).all()
        assert torch.all(torch.count_nonzero(quantized, dim=1) == 4)

    embedding = quantize_weight_ste(model.token_embedding.weight).detach()
    assert torch.isin(
        embedding, torch.tensor([-1.5, -0.5, 0.0, 0.5, 1.5])
    ).all()
    assert torch.all(torch.count_nonzero(embedding, dim=1) == 4)

    for block in model.blocks:
        depthwise = quantize_weight_ste(block.conv.depthwise_weight).detach()
        assert torch.all(
            torch.count_nonzero(
                depthwise.reshape(depthwise.shape[0], -1), dim=1
            ) == 4
        )
    assert linear_count > 0


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


def test_training_text_cleaner_normalizes_unicode_controls_and_whitespace():
    cleaned, reason = normalize_training_text("Cafe\u0301\r\nhello\tworld\x00\n\n\nnext")
    assert reason == "accepted"
    assert cleaned == "Café\nhello world\n\nnext"


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("Contact me at person@example.com", "email"),
        ("Use ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456", "credential"),
        ("Call me at +1 (415) 555-0134", "phone_contact"),
        ("\x00\x01", "too_short"),
        ("123456", "no_letters"),
        ("x" * 32769, "too_long"),
    ],
)
def test_training_text_cleaner_rejects_sensitive_or_low_quality_fields(value, reason):
    _, actual_reason = normalize_training_text(value)
    assert actual_reason == reason


def test_oasst_rejects_missing_or_invalid_tree_identity():
    base = {
        "message_id": "m1",
        "parent_id": None,
        "text": "A reviewed English message.",
        "role": "prompter",
        "lang": "en",
        "review_result": True,
        "deleted": False,
        "rank": None,
        "synthetic": False,
        "message_tree_id": "tree-1",
        "tree_state": "ready_for_export",
    }
    invalid = dict(base, message_tree_id="")
    assert accept_oasst1_message(invalid) == (False, "invalid_tree_id")


def test_incremental_cached_forward_matches_full_sequence_logits():
    torch.manual_seed(17)
    model = AddSubLanguageModel(AddSubConfig(dropout=0.0)).eval()
    token_ids = torch.randint(0, model.config.vocab_size, (2, 19))
    with torch.inference_mode():
        expected = model(token_ids)["logits"]
        caches = model.init_cache(batch_size=token_ids.shape[0])
        actual_steps = []
        for position in range(token_ids.shape[1]):
            logits, caches = model.forward_step(token_ids[:, position], caches)
            actual_steps.append(logits)
        actual = torch.stack(actual_steps, dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_additive_runtime_rejects_negative_weight_codes():
    with pytest.raises(ValueError, match="integers in"):
        additive_projection(
            np.asarray([[1, -1]], dtype=np.int8),
            np.asarray([[-1, 2]], dtype=np.int8),
        )


def test_byte_bpe_config_requires_bytes_and_all_control_tokens():
    with pytest.raises(ValueError, match="256 byte symbols"):
        AddSubConfig(vocab_size=263).validate()



def test_user_dataset_importer_accepts_selected_local_splits_without_download(tmp_path):
    import json

    from scripts.import_qmini_addsub_dataset import import_user_dataset

    pretrain_paths = {}
    for split, content in (
        ("train", "  English train text.  \r\n"),
        ("validation", "English validation text.\n"),
        ("test", "English held-out test text.\n"),
    ):
        path = tmp_path / f"{split}.txt"
        path.write_text(content, encoding="utf-8", newline="")
        pretrain_paths[split] = path

    sft_train = tmp_path / "chosen_sft_train.jsonl"
    sft_validation = tmp_path / "chosen_sft_valid.jsonl"
    sft_test = tmp_path / "chosen_sft_test.jsonl"
    records = []
    for split, path in (("train", sft_train), ("validation", sft_validation), ("test", sft_test)):
        record = {
            "human_authored": True,
            "synthetic": False,
            "messages": [
                {"role": "user", "content": f"What is {split}?"},
                {"role": "assistant", "content": f"This is {split}."},
            ],
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        records.append(path)

    output = tmp_path / "imported"
    manifest = import_user_dataset(
        pretrain_train=pretrain_paths["train"],
        pretrain_validation=pretrain_paths["validation"],
        pretrain_test=pretrain_paths["test"],
        output_dir=output,
        pretrain_source="user-selected-English-corpus",
        pretrain_license="user-reviewed-license",
        confirm_rights_and_human_data=True,
        confirm_english_data=True,
        sft_train=sft_train,
        sft_validation=sft_validation,
        sft_test=sft_test,
        sft_source="user-selected-assistant-corpus",
        sft_license="user-reviewed-instruction-license",
    )
    assert manifest["synthetic_training_records_included"] == 0
    assert manifest["pretraining"]["splits"]["train"]["sha256"]
    assert manifest["approved_instruction_sources"] == ["user-selected-assistant-corpus"]
    imported_record = json.loads((output / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert imported_record["source"] == "user-selected-assistant-corpus"
    assert imported_record["human_authored"] is True
    assert imported_record["synthetic"] is False


def test_sft_loader_accepts_only_manifest_approved_custom_sources(tmp_path):
    import json

    path = tmp_path / "custom_sft.jsonl"
    record = {
        "source": "user-selected-corpus",
        "human_authored": True,
        "synthetic": False,
        "messages": [
            {"role": "user", "content": "Explain a test."},
            {"role": "assistant", "content": "A test checks expected behaviour."},
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unapproved"):
        _load_conversations(path)
    assert len(_load_conversations(path, {"user-selected-corpus"})) == 1


def test_user_dataset_importer_requires_explicit_rights_and_non_synthetic_confirmation(tmp_path):
    from scripts.import_qmini_addsub_dataset import import_user_dataset

    paths = []
    for name in ("train", "validation", "test"):
        path = tmp_path / f"{name}.txt"
        path.write_text("English text.\n", encoding="utf-8")
        paths.append(path)
    with pytest.raises(ValueError, match="Explicit confirmation"):
        import_user_dataset(
            pretrain_train=paths[0],
            pretrain_validation=paths[1],
            pretrain_test=paths[2],
            output_dir=tmp_path / "out",
            pretrain_source="test-corpus",
            pretrain_license="test-license",
        )



def test_user_dataset_importer_requires_explicit_english_confirmation(tmp_path):
    from scripts.import_qmini_addsub_dataset import import_user_dataset

    paths = []
    for name in ("train", "validation", "test"):
        path = tmp_path / f"{name}.txt"
        path.write_text("English text.\n", encoding="utf-8")
        paths.append(path)
    with pytest.raises(ValueError, match="English"):
        import_user_dataset(
            pretrain_train=paths[0],
            pretrain_validation=paths[1],
            pretrain_test=paths[2],
            output_dir=tmp_path / "out",
            pretrain_source="test-corpus",
            pretrain_license="test-license",
            confirm_rights_and_human_data=True,
        )



def test_byte_bpe_config_rejects_vocab_larger_than_uint16_stream_capacity():
    with pytest.raises(ValueError, match="between 264 and 65536"):
        AddSubConfig(vocab_size=65537).validate()



def test_moe_auxiliary_loss_ignores_right_padding_tokens():
    torch.manual_seed(211)
    model = AddSubLanguageModel(
        AddSubConfig(
            vocab_size=300,
            d_model=32,
            d_hidden=48,
            n_layers=2,
            n_experts=2,
            kernel_size=3,
            max_context=32,
            dropout=0.0,
        )
    ).eval()
    inputs = torch.randint(0, model.config.vocab_size, (1, 17))
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    mask[:, :9] = True
    with torch.inference_mode():
        padded = model(inputs, attention_mask=mask)
        unpadded = model(inputs[:, :9])
    torch.testing.assert_close(
        padded["logits"][:, :9],
        unpadded["logits"],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        padded["aux_loss"],
        unpadded["aux_loss"],
        rtol=1e-6,
        atol=1e-6,
    )


def test_model_rejects_empty_inputs_and_invalid_attention_masks():
    model = AddSubLanguageModel(
        AddSubConfig(
            vocab_size=300,
            d_model=32,
            d_hidden=48,
            n_layers=1,
            n_experts=2,
            kernel_size=3,
            max_context=32,
            dropout=0.0,
        )
    )
    with pytest.raises(ValueError, match="non-empty"):
        model(torch.empty((1, 0), dtype=torch.long))
    inputs = torch.ones((1, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="at least one valid token"):
        model(inputs, attention_mask=torch.zeros_like(inputs, dtype=torch.bool))
    with pytest.raises(ValueError, match="match input_ids"):
        model(inputs, attention_mask=torch.ones((1, 3), dtype=torch.bool))


def test_prompt_builder_rejects_a_user_message_that_would_be_fully_trimmed():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from qmini_addsub.generate import build_chat_prompt

    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
        "<|end|>": 7,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    with pytest.raises(ValueError, match="current user message is too long"):
        build_chat_prompt(
            tokenizer,
            [{"role": "user", "content": "question"}],
            max_context=8,
            system_text="a very long system prompt with many words",
        )



def test_saturating_integer_add_rejects_float_inputs_and_clamps():
    left = np.asarray([-120, -5, 0, 120], dtype=np.int16)
    right = np.asarray([-20, 2, 5, 20], dtype=np.int16)
    assert saturating_add(left, right).tolist() == [-127, -3, 5, 127]
    with pytest.raises(TypeError, match="integer arrays"):
        saturating_add(np.asarray([1.5]), np.asarray([2], dtype=np.int8))
    with pytest.raises(ValueError, match="identical shapes"):
        saturating_add(np.asarray([1], dtype=np.int8), np.asarray([[1]], dtype=np.int8))


def test_additive_projection_exhausts_all_single_activation_weight_pairs():
    activations = np.arange(-127, 128, dtype=np.int16).reshape(-1, 1)
    codes = np.arange(5, dtype=np.uint8).reshape(5, 1)
    result = additive_projection(activations, codes)
    expected = np.empty((255, 5), dtype=np.int64)
    for row, value in enumerate(range(-127, 128)):
        expected[row, 0] = -(value + value + value)
        expected[row, 1] = -value
        expected[row, 2] = 0
        expected[row, 3] = value
        expected[row, 4] = value + value + value
    assert np.array_equal(result, expected)
