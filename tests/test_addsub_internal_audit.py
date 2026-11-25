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
