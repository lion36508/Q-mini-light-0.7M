from __future__ import annotations

import argparse
from pathlib import Path

from qmini_addsub.train import encode_text_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode Add/Sub v1 pretraining train/validation text into uint16 token streams."
    )
    parser.add_argument("--data-dir", default="data/qmini_addsub_v1")
    parser.add_argument("--tokenizer", default="artifacts/qmini_addsub_v1/tokenizer.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tokenizer = Path(args.tokenizer)
    if not tokenizer.is_file():
        raise FileNotFoundError("Train the Add/Sub v1 tokenizer first.")
    outputs = {}
    for split in ("train", "validation", "test"):
        source = data_dir / f"pretrain_{split}.txt"
        target = data_dir / f"pretrain_{split}.u16"
        if not source.is_file():
            raise FileNotFoundError(source)
        count, digest = encode_text_file(source, tokenizer, target)
        outputs[split] = {"tokens": count, "sha256": digest, "path": str(target)}
    print(outputs)


if __name__ == "__main__":
    main()
