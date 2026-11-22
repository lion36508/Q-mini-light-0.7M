from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

from qmini_addsub.config import AddSubConfig


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
    if not pretrain.is_file() or not manifest.is_file():
        raise FileNotFoundError("Prepare or import the selected English pretraining split first.")
    if not 264 <= args.vocab_size <= 65536:
        raise ValueError("Byte-level BPE vocab_size must be between 264 and 65536 (uint16 token streams).")

    # Only the training split is used. Validation/test text is never read here.
    trainer_corpus = Path(args.output).with_suffix(".training-text.tmp")
    trainer_corpus.parent.mkdir(parents=True, exist_ok=True)
    with trainer_corpus.open("w", encoding="utf-8") as target:
        with pretrain.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    target.write(line)
        if sft.is_file():
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
    actual_vocab_size = tokenizer.get_vocab_size()
    if not 264 <= actual_vocab_size <= args.vocab_size:
        raise RuntimeError(
            f"Tokenizer vocabulary {actual_vocab_size} is outside the valid range "
            f"[264, {args.vocab_size}]."
        )
    model_config = AddSubConfig(vocab_size=actual_vocab_size)
    config_path = output.with_name("model_config.json")
    config_path.write_text(
        json.dumps(model_config.to_dict(), indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    trained_on = ["pretrain_train.txt"]
    if sft.is_file():
        trained_on.append("sft_train.jsonl")
    print(json.dumps({
        "tokenizer": str(output),
        "model_config": str(config_path),
        "vocab_size": actual_vocab_size,
        "special_tokens": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
        "trained_on": trained_on,
        "validation_test_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
