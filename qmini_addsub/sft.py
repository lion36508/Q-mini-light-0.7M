from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

from .config import AddSubConfig
from .data import escape_special_token_literals, normalize_training_text
from .model import AddSubLanguageModel
from .train import CHECKPOINT_FORMAT as PRETRAIN_FORMAT
from .train import sha256_file


SFT_FORMAT = "qmini-addsub-sft-v1"
SYSTEM_TEXT = (
    "You are Q-Mini, a helpful English assistant. Answer the user's actual question "
    "clearly and accurately. Do not invent facts; state uncertainty when needed."
)


def _load_conversations(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be a JSON object.")
            if record.get("source") not in {
                "OpenAssistant/oasst1",
                "databricks/databricks-dolly-15k",
            }:
                raise ValueError(f"{path}:{line_number}: unapproved SFT source.")
            if record.get("human_authored") is not True or record.get("synthetic") is not False:
                raise ValueError(f"{path}:{line_number}: synthetic/unverified SFT record rejected.")
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            if any(not isinstance(item, dict) for item in messages):
                raise ValueError(f"{path}:{line_number}: every message must be an object.")
            roles = [item.get("role") for item in messages]
            if roles[0] != "user" or roles[-1] != "assistant":
                continue
            if any(role not in {"user", "assistant"} for role in roles):
                continue
            if any(roles[index] == roles[index - 1] for index in range(1, len(roles))):
                continue
            cleaned_messages = []
            for item in messages:
                cleaned, reason = normalize_training_text(item.get("content"))
                if reason != "accepted":
                    raise ValueError(
                        f"{path}:{line_number}: message rejected by text cleaner ({reason})."
                    )
                cleaned_item = dict(item)
                cleaned_item["content"] = cleaned
                cleaned_messages.append(cleaned_item)
            cleaned_record = dict(record)
            cleaned_record["messages"] = cleaned_messages
            records.append(cleaned_record)
    if not records:
        raise ValueError(f"No valid conversations found in {path}.")
    return records


class ConversationDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer: Tokenizer, max_context: int):
        self.samples = []
        special = {
            token: tokenizer.token_to_id(token)
            for token in (
                "<bos>", "<eos>", "<pad>", "<|system|>", "<|user|>",
                "<|assistant|>", "<|end|>",
            )
        }
        if any(value is None for value in special.values()):
            raise ValueError("Tokenizer is missing a required Add/Sub chat token.")
        special = {key: int(value) for key, value in special.items()}

        for record in records:
            messages = [{"role": "system", "content": SYSTEM_TEXT}]
            messages.extend(record["messages"])
            encoded = []
            for message in messages:
                role = message["role"]
                role_token = "<|system|>" if role == "system" else (
                    "<|user|>" if role == "user" else "<|assistant|>"
                )
                content = str(message.get("content") or "").strip()
                content_ids = tokenizer.encode(escape_special_token_literals(content)).ids
                if not content_ids:
                    continue
                encoded.append((role, special[role_token], content_ids))

            # Retain the most recent complete exchange; never cut into the final answer.
            def required_length(items):
                return 1 + sum(1 + len(ids) + 1 for _, _, ids in items)

            while len(encoded) > 3 and required_length(encoded) > max_context:
                encoded.pop(1)
                if len(encoded) > 2:
                    encoded.pop(1)
            if required_length(encoded) > max_context:
                last_role, last_token, last_ids = encoded[-1]
                if last_role != "assistant":
                    continue
                room = max_context - (required_length(encoded[:-1]) + 2)
                if room < 1:
                    continue
                encoded[-1] = (last_role, last_token, last_ids[:room])

            token_ids = [special["<bos>"]]
            label_ids = [-100]
            for role, role_token, content_ids in encoded:
                token_ids.append(role_token)
                label_ids.append(-100)
                token_ids.extend(content_ids)
                label_ids.extend(content_ids if role == "assistant" else [-100] * len(content_ids))
                token_ids.append(special["<|end|>"])
                label_ids.append(special["<|end|>"] if role == "assistant" else -100)
            if len(token_ids) > max_context or token_ids[-1] != special["<|end|>"]:
                continue
            if not any(value != -100 for value in label_ids[1:]):
                continue
            self.samples.append((token_ids, label_ids))

        if not self.samples:
            raise ValueError("No conversations fit the configured context.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _collate(batch: list[tuple[list[int], list[int]]], pad_id: int):
    maximum = max(len(tokens) for tokens, _ in batch)
    inputs = torch.full((len(batch), maximum - 1), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), maximum - 1), -100, dtype=torch.long)
    for index, (tokens, targets) in enumerate(batch):
        source = torch.as_tensor(tokens[:-1], dtype=torch.long)
        target = torch.as_tensor(targets[1:], dtype=torch.long)
        inputs[index, : source.numel()] = source
        labels[index, : target.numel()] = target
    return inputs, labels


def fine_tune_from_addsub_checkpoint(
    base_checkpoint: str | Path,
    tokenizer_path: str | Path,
    train_jsonl: str | Path,
    valid_jsonl: str | Path,
    output_dir: str | Path,
    steps: int = 800,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    seed: int = 20261119,
    device_name: str = "cpu",
    resume: str | Path | None = None,
) -> dict[str, Any]:
    device = torch.device(device_name)
    base_path, tokenizer_path = Path(base_checkpoint), Path(tokenizer_path)
    train_path, valid_path, output = Path(train_jsonl), Path(valid_jsonl), Path(output_dir)
    for path in (base_path, tokenizer_path, train_path, valid_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    base = torch.load(base_path, map_location=device, weights_only=True)
    if base.get("format") != PRETRAIN_FORMAT:
        raise ValueError("SFT accepts only a Q-Mini Add/Sub v1 pretraining checkpoint.")
    config = AddSubConfig.from_dict(base["config"])
    if int(base.get("step", 0)) < 1:
        raise ValueError("SFT refuses a step-zero or random-only base checkpoint.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match the Add/Sub base config.")

    manifest_path = train_path.parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Verified Add/Sub dataset_manifest.json is required for SFT.")
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data_manifest.get("format") != "qmini-addsub-data-v1":
        raise ValueError("Unsupported Add/Sub dataset manifest.")
    if data_manifest.get("synthetic_training_records_included") != 0:
        raise ValueError("SFT refuses a dataset manifest containing synthetic training records.")
    sft_manifest = data_manifest.get("instruction_splits", {})
    tokenizer_meta = data_manifest.get("tokenizer", {})
    if tokenizer_meta.get("sha256") != sha256_file(tokenizer_path):
        raise ValueError("SFT tokenizer does not match the curated dataset/tokenizer manifest.")
    for split_name, source_path in (("train", train_path), ("validation", valid_path)):
        entry = sft_manifest.get(split_name, {})
        if entry.get("path") != source_path.name or entry.get("sha256") != sha256_file(source_path):
            raise ValueError(f"{split_name} SFT file does not match the curated data manifest.")
    if train_path.resolve() == valid_path.resolve():
        raise ValueError("SFT training and validation files must be different.")
    data_manifest_sha = sha256_file(manifest_path)

    train_records = _load_conversations(train_path)
    valid_records = _load_conversations(valid_path)
    train_data = ConversationDataset(train_records, tokenizer, config.max_context)
    valid_data = ConversationDataset(valid_records, tokenizer, config.max_context)
    pad_id = int(tokenizer.token_to_id("<pad>"))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        collate_fn=lambda batch: _collate(batch, pad_id),
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: _collate(batch, pad_id),
    )

    torch.manual_seed(seed)
    model = AddSubLanguageModel(config).to(device)
    model.load_state_dict(base["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    hashes = {
        "base_sha256": sha256_file(base_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "train_sha256": sha256_file(train_path),
        "valid_sha256": sha256_file(valid_path),
        "data_manifest_sha256": data_manifest_sha,
    }
    start_step = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        expected = {"format": SFT_FORMAT, "config": config.to_dict(), **hashes}
        mismatch = [key for key, value in expected.items() if checkpoint.get(key) != value]
        if mismatch:
            raise ValueError("Incompatible Add/Sub SFT resume: " + ", ".join(mismatch))
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "sft_metrics.jsonl"
    iterator = iter(train_loader)

    def evaluate() -> float:
        model.eval()
        loss_sum = 0.0
        target_count = 0
        with torch.inference_mode():
            for inputs, labels in valid_loader:
                result = model(inputs.to(device))
                labels = labels.to(device)
                count = int((labels != -100).sum())
                if count:
                    loss_sum += float(F.cross_entropy(
                        result["logits"].float().reshape(-1, config.vocab_size),
                        labels.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    ))
                    target_count += count
        model.train()
        if not target_count:
            raise ValueError("Validation set contains no assistant response targets.")
        return loss_sum / target_count

    model.train()
    last_loss = None
    for step in range(start_step + 1, steps + 1):
        try:
            inputs, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, labels = next(iterator)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        result = model(inputs)
        target_count = int((labels != -100).sum())
        if target_count == 0:
            continue
        loss = F.cross_entropy(
            result["logits"].float().reshape(-1, config.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        ) + 0.01 * result["aux_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite SFT loss at step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())

        if step % 50 == 0 or step == steps:
            valid_loss = evaluate()
            row = {"step": step, "train_loss": last_loss, "valid_response_loss": valid_loss}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True))
        if step % 100 == 0 or step == steps:
            payload = {
                "format": SFT_FORMAT,
                "step": step,
                "config": config.to_dict(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                **hashes,
            }
            destination = output / f"sft-checkpoint-step-{step:09d}.pt"
            temporary = destination.with_suffix(".tmp")
            torch.save(payload, temporary)
            temporary.replace(destination)

    return {
        "format": SFT_FORMAT,
        "steps": steps,
        "train_records": len(train_data),
        "validation_records": len(valid_data),
        "parameter_count": model.parameter_count(),
        "final_train_loss": last_loss,
        "output_dir": str(output),
        "note": "SFT is training-only; full additive inference export remains a separate gate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Instruction-fine-tune an Add/Sub v1 pretraining checkpoint.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--valid-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    result = fine_tune_from_addsub_checkpoint(
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        train_jsonl=args.train_jsonl,
        valid_jsonl=args.valid_jsonl,
        output_dir=args.output,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device_name=args.device,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
