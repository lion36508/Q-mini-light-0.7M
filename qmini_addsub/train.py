from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .config import AddSubConfig
from .model import AddSubLanguageModel


CHECKPOINT_FORMAT = "qmini-addsub-training-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_text_file(
    text_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
) -> tuple[int, str]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Tokenizer must contain the <eos> token.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(text_path).open("r", encoding="utf-8") as source, output.open("wb") as target:
        for line in source:
            text = line.strip()
            if not text:
                continue
            ids = tokenizer.encode(text).ids
            if not ids:
                continue
            values = np.asarray(ids + [eos_id], dtype=np.uint16)
            values.tofile(target)
            count += int(values.size)
    if count < 2:
        raise ValueError("Encoded token stream is empty or too short.")
    return count, sha256_file(output)


def _sample_batch(
    tokens: np.ndarray,
    batch_size: int,
    sequence_length: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = len(tokens) - sequence_length - 1
    if maximum < 1:
        raise ValueError("Token stream is shorter than one training sequence.")
    starts = [rng.randrange(maximum) for _ in range(batch_size)]
    inputs = np.stack([tokens[start : start + sequence_length] for start in starts])
    targets = np.stack(
        [tokens[start + 1 : start + sequence_length + 1] for start in starts]
    )
    return (
        torch.as_tensor(inputs.astype(np.int64), device=device),
        torch.as_tensor(targets.astype(np.int64), device=device),
    )


def train_from_scratch(
    train_bin: str | Path,
    valid_bin: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    config: AddSubConfig | None = None,
    steps: int = 1000,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    sequence_length: int = 256,
    save_every: int = 100,
    validate_every: int = 100,
    seed: int = 20261119,
    device_name: str = "cpu",
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Train only from a new random initialization or this package's own checkpoint."""
    config = config or AddSubConfig()
    config.validate()
    if steps < 1 or batch_size < 1 or sequence_length < 2:
        raise ValueError("steps, batch_size and sequence_length must be positive.")
    if sequence_length > config.max_context:
        raise ValueError("sequence_length exceeds the configured context.")
    device = torch.device(device_name)
    train_path, valid_path = Path(train_bin), Path(valid_bin)
    tokenizer_path, output = Path(tokenizer_path), Path(output_dir)
    if not train_path.is_file() or not valid_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("Train stream, validation stream and tokenizer are required.")

    train_tokens = np.memmap(train_path, dtype=np.uint16, mode="r")
    valid_tokens = np.memmap(valid_path, dtype=np.uint16, mode="r")
    train_sha, valid_sha = sha256_file(train_path), sha256_file(valid_path)
    tokenizer_sha = sha256_file(tokenizer_path)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = AddSubLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    start_step = 0
    rng = random.Random(seed)

    if resume is not None:
        checkpoint_path = Path(resume)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        expected = {
            "format": CHECKPOINT_FORMAT,
            "config": config.to_dict(),
            "train_sha256": train_sha,
            "valid_sha256": valid_sha,
            "tokenizer_sha256": tokenizer_sha,
        }
        mismatches = [
            key for key, value in expected.items() if checkpoint.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "Resume checkpoint is not compatible with this Add/Sub v1 run: "
                + ", ".join(mismatches)
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        torch.set_rng_state(checkpoint["torch_rng"])
        rng.setstate(checkpoint["python_rng"])

    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"

    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.inference_mode():
            for _ in range(4):
                x, y = _sample_batch(
                    valid_tokens, batch_size, sequence_length, rng, device
                )
                result = model(x)
                loss = F.cross_entropy(
                    result["logits"].float().reshape(-1, config.vocab_size),
                    y.reshape(-1),
                )
                losses.append(float(loss))
        model.train()
        return float(sum(losses) / len(losses))

    def save_checkpoint(step: int, valid_loss: float | None) -> Path:
        path = output / f"checkpoint-step-{step:09d}.pt"
        payload = {
            "format": CHECKPOINT_FORMAT,
            "step": step,
            "config": config.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_sha256": train_sha,
            "valid_sha256": valid_sha,
            "tokenizer_sha256": tokenizer_sha,
            "valid_loss": valid_loss,
            "torch_rng": torch.get_rng_state(),
            "python_rng": rng.getstate(),
        }
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    model.train()
    last_loss = None
    for step in range(start_step + 1, steps + 1):
        inputs, targets = _sample_batch(
            train_tokens, batch_size, sequence_length, rng, device
        )
        optimizer.zero_grad(set_to_none=True)
        result = model(inputs)
        token_loss = F.cross_entropy(
            result["logits"].float().reshape(-1, config.vocab_size),
            targets.reshape(-1),
        )
        loss = token_loss + 0.01 * result["aux_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(token_loss.detach())

        if step % validate_every == 0 or step == steps:
            valid_loss = evaluate()
            record = {
                "step": step,
                "train_token_loss": last_loss,
                "valid_token_loss": valid_loss,
                "train_sha256": train_sha,
                "valid_sha256": valid_sha,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))
        else:
            valid_loss = None

        if step % save_every == 0 or step == steps:
            save_checkpoint(step, valid_loss)

    return {
        "format": CHECKPOINT_FORMAT,
        "steps": steps,
        "parameter_count": model.parameter_count(),
        "final_train_token_loss": last_loss,
        "output_dir": str(output),
        "note": "Training graph only; complete add/sub-only inference export is a separate gate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Q-Mini Add/Sub v1 from random initialization.")
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--valid-bin", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    result = train_from_scratch(
        train_bin=args.train_bin,
        valid_bin=args.valid_bin,
        tokenizer_path=args.tokenizer,
        output_dir=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        save_every=args.save_every,
        validate_every=args.validate_every,
        seed=args.seed,
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
