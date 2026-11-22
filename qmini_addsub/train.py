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
from .data import escape_special_token_literals
from .optim import build_adamw, cosine_warmup_lr


CHECKPOINT_FORMAT = "qmini-addsub-training-v3"


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
    tokenizer_sha = sha256_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Tokenizer must contain the <eos> token.")
    if tokenizer.get_vocab_size() > 65536:
        raise ValueError("Tokenizer vocabulary exceeds the uint16 token-stream capacity.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    try:
        with Path(text_path).open("r", encoding="utf-8") as source, temporary.open("wb") as target:
            for line in source:
                text = line.strip()
                if not text:
                    continue
                safe_text = escape_special_token_literals(text)
                ids = tokenizer.encode(safe_text, add_special_tokens=False).ids
                if not ids:
                    continue
                if any(token_id < 0 or token_id >= 65536 for token_id in ids):
                    raise ValueError("Tokenizer emitted a token ID outside uint16 range.")
                values = np.asarray(ids + [eos_id], dtype=np.uint16)
                values.tofile(target)
                count += int(values.size)
        if count < 2:
            raise ValueError("Encoded token stream is empty or too short.")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count, sha256_file(output)


def _sample_batch(
    tokens: np.ndarray,
    batch_size: int,
    sequence_length: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    last_start = len(tokens) - sequence_length - 1
    if last_start < 0:
        raise ValueError("Token stream is shorter than one training sequence.")
    starts = [rng.randrange(last_start + 1) for _ in range(batch_size)]
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
    if save_every < 1 or validate_every < 1:
        raise ValueError("save_every and validate_every must be positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if sequence_length > config.max_context:
        raise ValueError("sequence_length exceeds the configured context.")
    device = torch.device(device_name)
    train_path, valid_path = Path(train_bin), Path(valid_bin)
    tokenizer_path, output = Path(tokenizer_path), Path(output_dir)
    if not train_path.is_file() or not valid_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("Train stream, validation stream and tokenizer are required.")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_sha = sha256_file(tokenizer_path)
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab {tokenizer.get_vocab_size()} does not match "
            f"Add/Sub config vocab {config.vocab_size}."
        )
    manifest_path = train_path.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "A verified dataset_manifest.json is required beside the Add/Sub token streams."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "qmini-addsub-data-v1" or manifest.get("language") != "English":
        raise ValueError("Unsupported or non-English Add/Sub dataset manifest.")
    if manifest.get("synthetic_training_records_included") != 0:
        raise ValueError("Add/Sub v1 refuses a corpus containing synthetic training records.")
    pretraining = manifest.get("pretraining", {}).get("splits", {})
    for split, source_name in (
        ("train", "pretrain_train.txt"),
        ("validation", "pretrain_validation.txt"),
        ("test", "pretrain_test.txt"),
    ):
        source_path = train_path.parent / source_name
        expected_sha = pretraining.get(split, {}).get("sha256")
        if not source_path.is_file() or not expected_sha or sha256_file(source_path) != expected_sha:
            raise ValueError(f"{split} pretraining source does not match the manifest.")
    token_streams = manifest.get("token_streams", {})
    if token_streams.get("train", {}).get("sha256") != sha256_file(train_path):
        raise ValueError("Training token stream does not match the dataset manifest.")
    if token_streams.get("validation", {}).get("sha256") != sha256_file(valid_path):
        raise ValueError("Validation token stream does not match the dataset manifest.")
    test_stream = token_streams.get("test", {})
    test_path = train_path.parent / str(test_stream.get("path", ""))
    if not test_path.is_file() or test_stream.get("sha256") != sha256_file(test_path):
        raise ValueError("A separate, verified held-out test token stream is required.")
    if len({train_path.resolve(), valid_path.resolve(), test_path.resolve()}) != 3:
        raise ValueError("Train, validation and test token streams must be separate files.")
    tokenizer_meta = manifest.get("tokenizer", {})
    if tokenizer_meta.get("sha256") != tokenizer_sha:
        raise ValueError("Tokenizer fingerprint does not match the dataset manifest.")

    train_tokens = np.memmap(train_path, dtype=np.uint16, mode="r")
    valid_tokens = np.memmap(valid_path, dtype=np.uint16, mode="r")
    test_tokens = np.memmap(test_path, dtype=np.uint16, mode="r")
    if not len(train_tokens) or not len(valid_tokens) or not len(test_tokens):
        raise ValueError("Train, validation and test token streams must all be non-empty.")
    if (
        int(train_tokens.max()) >= config.vocab_size
        or int(valid_tokens.max()) >= config.vocab_size
        or int(test_tokens.max()) >= config.vocab_size
    ):
        raise ValueError("A token stream contains IDs outside the configured vocabulary.")
    train_sha, valid_sha = sha256_file(train_path), sha256_file(valid_path)
    test_sha = sha256_file(test_path)
    manifest_sha = sha256_file(manifest_path)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    run_contract = {
        "planned_steps": steps,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    model = AddSubLanguageModel(config).to(device)
    optimizer = build_adamw(model, learning_rate, weight_decay=0.1)
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
            "test_sha256": test_sha,
            "tokenizer_sha256": tokenizer_sha,
            "data_manifest_sha256": manifest_sha,
            "run_contract": run_contract,
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
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            saved_cuda_rng = checkpoint.get("cuda_rng_states")
            if not saved_cuda_rng or len(saved_cuda_rng) != torch.cuda.device_count():
                raise ValueError(
                    "CUDA RNG state is missing or device count changed; exact resume is unsafe."
                )
            torch.cuda.set_rng_state_all([state.cpu() for state in saved_cuda_rng])
        rng.setstate(checkpoint["python_rng"])
    if steps <= start_step:
        raise ValueError("steps must be greater than the checkpoint's completed step.")

    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    if resume is None and (
        metrics_path.exists() or any(output.glob("checkpoint-step-*.pt"))
    ):
        raise FileExistsError(
            "Pretraining output already contains a run. Use a fresh output folder or "
            "resume from a compatible checkpoint; old metrics will not be mixed."
        )

    def evaluate(tokens: np.ndarray) -> float:
        """Evaluate on fixed evenly spaced windows without advancing train RNG."""
        model.eval()
        max_start = len(tokens) - sequence_length - 1
        if max_start < 0:
            raise ValueError("Evaluation token stream is shorter than one sequence.")
        number_of_windows = min(64, max_start + 1)
        if number_of_windows == 1:
            starts = [0]
        else:
            stride = max(1, max_start // (number_of_windows - 1))
            starts = list(range(0, max_start + 1, stride))[:number_of_windows]

        total_loss = 0.0
        target_count = 0
        with torch.inference_mode():
            for offset in range(0, len(starts), batch_size):
                current = starts[offset : offset + batch_size]
                x = np.stack([tokens[start : start + sequence_length] for start in current])
                y = np.stack(
                    [tokens[start + 1 : start + sequence_length + 1] for start in current]
                )
                inputs = torch.as_tensor(x.astype(np.int64), device=device)
                targets = torch.as_tensor(y.astype(np.int64), device=device)
                result = model(inputs)
                total_loss += float(F.cross_entropy(
                    result["logits"].float().reshape(-1, config.vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                ))
                target_count += int(targets.numel())
        model.train()
        if not target_count:
            raise ValueError("Evaluation did not contain any target tokens.")
        return total_loss / target_count

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
            "test_sha256": test_sha,
            "tokenizer_sha256": tokenizer_sha,
            "data_manifest_sha256": manifest_sha,
            "run_contract": run_contract,
            "valid_loss": valid_loss,
            "torch_rng": torch.get_rng_state(),
            "python_rng": rng.getstate(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    model.train()
    last_loss = None
    for step in range(start_step + 1, steps + 1):
        current_lr = cosine_warmup_lr(step, steps, learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
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
            valid_loss = evaluate(valid_tokens)
            record = {
                "step": step,
                "learning_rate": current_lr,
                "train_token_loss": last_loss,
                "valid_token_loss": valid_loss,
                "train_sha256": train_sha,
                "valid_sha256": valid_sha,
                "test_sha256": test_sha,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))
        else:
            valid_loss = None

        if step % save_every == 0 or step == steps:
            save_checkpoint(step, valid_loss)

    heldout_test_loss = evaluate(test_tokens)
    return {
        "format": CHECKPOINT_FORMAT,
        "steps": steps,
        "parameter_count": model.parameter_count(),
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "final_train_token_loss": last_loss,
        "final_validation_token_loss": evaluate(valid_tokens),
        "heldout_test_token_loss": heldout_test_loss,
        "heldout_test_sha256": sha256_file(test_path),
        "output_dir": str(output),
        "data_manifest_sha256": manifest_sha,
        "note": "Training graph only; complete add/sub-only inference export is a separate gate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Q-Mini Add/Sub v1 from random initialization.")
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--valid-bin", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", default=None, help="Optional model_config.json written beside the trained tokenizer.")
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
    config = None
    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = AddSubConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    result = train_from_scratch(
        train_bin=args.train_bin,
        valid_bin=args.valid_bin,
        tokenizer_path=args.tokenizer,
        output_dir=args.output,
        config=config,
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
