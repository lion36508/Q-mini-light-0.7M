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
from qmini_addsub.data import escape_special_token_literals
from qmini_addsub.train import sha256_file


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
    data_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        data_manifest.get("format") != "qmini-addsub-data-v1"
        or data_manifest.get("language") != "English"
        or data_manifest.get("synthetic_training_records_included") != 0
    ):
        raise ValueError("Tokenizer requires a verified English, non-synthetic Add/Sub dataset manifest.")
    expected_pretrain = data_manifest.get("pretraining", {}).get("splits", {}).get("train", {})
    if expected_pretrain.get("sha256") != sha256_file(pretrain):
        raise ValueError("Pretraining train text does not match the dataset manifest.")
    instruction_splits = data_manifest.get("instruction_splits", {})
    sft_train_entry = instruction_splits.get("train")
    if sft_train_entry is not None:
        if not sft.is_file():
            raise FileNotFoundError("Manifest declares an SFT training split, but sft_train.jsonl is missing.")
        if sft_train_entry.get("path") != sft.name or sft_train_entry.get("sha256") != sha256_file(sft):
            raise ValueError("SFT train file does not match the dataset manifest.")
        for split_name in ("validation", "test"):
            entry = instruction_splits.get(split_name)
            if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
                raise ValueError(f"Manifest is missing the SFT {split_name} split.")
            split_path = data_dir / entry["path"]
            if not split_path.is_file() or sha256_file(split_path) != entry["sha256"]:
                raise ValueError(f"SFT {split_name} file does not match the dataset manifest.")
    elif sft.is_file():
        raise ValueError("SFT train file exists but is not declared in the dataset manifest.")
    if not 264 <= args.vocab_size <= 65536:
        raise ValueError("Byte-level BPE vocab_size must be between 264 and 65536 (uint16 token streams).")

    # Only the training split is used. Validation/test text is never read here.
    trainer_corpus = Path(args.output).with_suffix(".training-text.tmp")
    output = Path(args.output)
    config_path = output.with_name("model_config.json")
    tokenizer_temporary = output.with_suffix(output.suffix + ".tmp")
    config_temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    trainer_corpus.parent.mkdir(parents=True, exist_ok=True)
    try:
        with trainer_corpus.open("w", encoding="utf-8") as target:
            with pretrain.open("r", encoding="utf-8") as source:
                for line in source:
                    text = line.strip()
                    if text:
                        target.write(escape_special_token_literals(text) + "\\n")
            if sft_train_entry is not None:
                with sft.open("r", encoding="utf-8") as source:
                    for line_number, line in enumerate(source, 1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"{sft}:{line_number}: invalid JSONL.") from exc
                        for message in record.get("messages", []):
                            content = str(message.get("content") or "").strip()
                            if content:
                                target.write(escape_special_token_literals(content) + "\n")

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
        tokenizer.save(str(tokenizer_temporary))
        config_temporary.write_text(
            json.dumps(model_config.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tokenizer_temporary.replace(output)
        config_temporary.replace(config_path)

        trained_on = ["pretrain_train.txt"]
        if sft_train_entry is not None:
            trained_on.append("sft_train.jsonl")
        print(json.dumps({
            "tokenizer": str(output),
            "model_config": str(config_path),
            "vocab_size": actual_vocab_size,
            "special_tokens": {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
            "trained_on": trained_on,
            "validation_test_used": False,
        }, indent=2))
    finally:
        trainer_corpus.unlink(missing_ok=True)
        tokenizer_temporary.unlink(missing_ok=True)
        config_temporary.unlink(missing_ok=True)



if __name__ == "__main__":
    main()
