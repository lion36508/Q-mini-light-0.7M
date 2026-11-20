from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = [
    "<unk>",
    "<pad>",
    "<bos>",
    "<eos>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the isolated Add/Sub v1 tokenizer from training-split text only."
    )
    parser.add_argument("--data-dir", default="data/qmini_addsub_v1")
    parser.add_argument("--output", default="artifacts/qmini_addsub_v1/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pretrain = data_dir / "pretrain_train.txt"
    sft = data_dir / "sft_train.jsonl"
    manifest = data_dir / "dataset_manifest.json"
    if not pretrain.is_file() or not sft.is_file() or not manifest.is_file():
        raise FileNotFoundError("Run scripts/prepare_qmini_addsub_data.py first.")
    if args.vocab_size < 263:
        raise ValueError("Byte-level BPE vocab_size must be at least 263.")

    # Only the training split is used. Validation/test text is never read here.
    trainer_corpus = Path(args.output).with_suffix(".training-text.tmp")
    trainer_corpus.parent.mkdir(parents=True, exist_ok=True)
    with trainer_corpus.open("w", encoding="utf-8") as target:
        with pretrain.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    target.write(line)
        with sft.open("r", encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                for message in record.get("messages", []):
                    text = str(message.get("content") or "").strip()
                    if text:
                        target.write(text + "\n")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevel()
    byte_alphabet = ByteLevelPreTokenizer.alphabet()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=byte_alphabet,
        show_progress=True,
    )
    tokenizer.train([str(trainer_corpus)], trainer)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    trainer_corpus.unlink(missing_ok=True)

    for token in SPECIAL_TOKENS:
        if tokenizer.token_to_id(token) is None:
            raise RuntimeError(f"Tokenizer is missing required token: {token}")
    if tokenizer.get_vocab_size() > args.vocab_size:
        raise RuntimeError("Tokenizer exceeded the requested vocabulary size.")
    print(json.dumps({
        "tokenizer": str(output),
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        "trained_on": ["pretrain_train.txt", "sft_train.jsonl"],
        "validation_test_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
