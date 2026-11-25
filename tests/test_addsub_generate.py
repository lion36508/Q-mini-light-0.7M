from types import SimpleNamespace

import pytest
import torch

from qmini_addsub.generate import build_chat_prompt, _choose_next_token


class _TinyPromptTokenizer:
    _ids = {
        "<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3,
        "<|system|>": 4, "<|user|>": 5, "<|assistant|>": 6, "<|end|>": 7,
    }

    def token_to_id(self, token):
        return self._ids.get(token)

    def encode(self, text, add_special_tokens=False):
        pieces = text.split()
        return SimpleNamespace(ids=[8 + index for index, _ in enumerate(pieces)])


def test_chat_prompt_rejects_history_that_starts_with_assistant():
    with pytest.raises(ValueError, match="start with a user"):
        build_chat_prompt(
            _TinyPromptTokenizer(),
            [{"role": "assistant", "content": "Hello"},
             {"role": "user", "content": "Hi"}],
            max_context=64,
        )


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), -float("inf"), True])
def test_sampling_rejects_nonfinite_temperature(temperature):
    with pytest.raises(ValueError, match="temperature must be finite"):
        _choose_next_token(
            torch.zeros(8), temperature, top_k=4, top_p=0.9, greedy=False
        )


def test_sampling_rejects_nonfinite_logits():
    logits = torch.zeros(8)
    logits[3] = float("nan")
    with pytest.raises(ValueError, match="logits contain non-finite"):
        _choose_next_token(logits, 0.7, top_k=4, top_p=0.9, greedy=False)


def test_top_p_keeps_the_crossing_token_and_removes_the_tail(monkeypatch):
    captured = {}

    def capture_multinomial(probabilities, num_samples):
        captured["probabilities"] = probabilities.detach().clone()
        return torch.argmax(probabilities).reshape(1)

    monkeypatch.setattr(torch, "multinomial", capture_multinomial)
    logits = torch.log(torch.tensor([0.6, 0.3, 0.1], dtype=torch.float32))
    _choose_next_token(logits, 1.0, top_k=0, top_p=0.8, greedy=False)

    probabilities = captured["probabilities"]
    assert probabilities[0] > 0
    assert probabilities[1] > 0
    assert probabilities[2] == 0
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))


def test_top_k_keeps_exact_number_of_tokens_when_scores_tie(monkeypatch):
    captured = {}

    def capture_multinomial(probabilities, num_samples):
        captured["probabilities"] = probabilities.detach().clone()
        return torch.argmax(probabilities).reshape(1)

    monkeypatch.setattr(torch, "multinomial", capture_multinomial)
    _choose_next_token(torch.ones(8), 1.0, top_k=3, top_p=1.0, greedy=False)

    assert int(torch.count_nonzero(captured["probabilities"])) == 3
