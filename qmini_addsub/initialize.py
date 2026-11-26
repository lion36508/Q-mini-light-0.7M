from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer

from .config import AddSubConfig, MAX_PARAMETER_BUDGET
from .model import AddSubLanguageModel


INITIAL_MODEL_FORMAT = "qmini-addsub-initial-model-v1"
MODEL_MANIFEST_FORMAT = "qmini-addsub-model-manifest-v1"
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
EXPECTED_PROFILE = {
    "d_model": 128,
    "d_hidden": 448,
    "n_layers": 4,
    "n_experts": 2,
    "kernel_size": 7,
    "max_context": 512,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_bootstrap_tokenizer() -> Tokenizer:
    """Build a dataset-independent UTF-8 byte-level tokenizer with stable IDs.

    It has one token for each possible byte plus the eight Q-Mini chat/control
    tokens. It has no learned BPE merges and downloads no corpus. A later
    training run therefore keeps this exact tokenizer and embedding shape.
    """
    alphabet = sorted(ByteLevelPreTokenizer.alphabet())
    if len(alphabet) != 256 or len(set(alphabet)) != 256:
        raise RuntimeError("ByteLevel tokenizer must expose exactly 256 unique byte symbols.")

    vocab = {token: index for index, token in enumerate(alphabet)}
    for token in SPECIAL_TOKENS:
        if token not in vocab:
            vocab[token] = len(vocab)

    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.add_special_tokens([
        AddedToken(token, normalized=False, special=True)
        for token in SPECIAL_TOKENS
    ])

    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size != 264:
        raise RuntimeError(
            f"Bootstrap tokenizer must contain 264 IDs, got {actual_vocab_size}."
        )
    token_ids = [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS]
    if any(token_id is None for token_id in token_ids) or len(set(token_ids)) != 8:
        raise RuntimeError("Bootstrap tokenizer control-token IDs are missing or duplicated.")

    sample = "Q-Mini Add/Sub: English assistant ready. Café – 2026!"
    encoded = tokenizer.encode(sample, add_special_tokens=False)
    decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
    if decoded != sample:
        raise RuntimeError(
            f"Bootstrap tokenizer failed UTF-8 round-trip: {decoded!r} != {sample!r}"
        )
    if not encoded.ids or max(encoded.ids) >= actual_vocab_size:
        raise RuntimeError("Bootstrap tokenizer emitted an invalid token ID.")
    return tokenizer


def _required_artifact_paths(artifact_dir: Path) -> dict[str, Path]:
    return {
        "tokenizer": artifact_dir / "tokenizer.json",
        "config": artifact_dir / "model_config.json",
        "initial_model": artifact_dir / "initial_model.pt",
        "manifest": artifact_dir / "model_manifest.json",
        "report": artifact_dir / "initialization_report.json",
        "model_card": artifact_dir / "MODEL_CARD.md",
    }


def _load_config(path: Path, tokenizer_vocab_size: int) -> AddSubConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Q-Mini 2M profile is missing: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Q-Mini Add/Sub profile must be a JSON object.")
    values = dict(values)
    for key, expected in EXPECTED_PROFILE.items():
        if values.get(key) != expected:
            raise ValueError(
                f"Unexpected 2M architecture profile: {key}={values.get(key)!r}; "
                f"expected {expected!r}."
            )
    values["vocab_size"] = tokenizer_vocab_size
    config = AddSubConfig.from_dict(values)
    if config.estimated_parameter_count() > MAX_PARAMETER_BUDGET:
        raise ValueError("The configured model exceeds the hard 2,000,000 parameter budget.")
    return config


def load_initial_model_checkpoint(
    model: AddSubLanguageModel,
    checkpoint_path: str | Path,
    tokenizer_sha256: str,
) -> str:
    """Load and verify the untrained bootstrap weights into the exact model graph."""
    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(f"Initialized Q-Mini model weights are missing: {source}")
    if model.wavebit_options is not None:
        raise ValueError("The standard bootstrap checkpoint cannot initialize WaveBit-HSRA mode.")

    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Initial model checkpoint must be a dictionary.")
    if checkpoint.get("format") != INITIAL_MODEL_FORMAT:
        raise ValueError("Unsupported initial-model checkpoint format.")
    if checkpoint.get("stage") != "initialized_untrained" or checkpoint.get("step") != 0:
        raise ValueError("Initial-model checkpoint must be an untrained step-zero model.")
    if checkpoint.get("config") != model.config.to_dict():
        raise ValueError("Initial-model config does not match the requested training model.")
    if checkpoint.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("Initial-model tokenizer fingerprint does not match the active tokenizer.")
    if checkpoint.get("parameter_count") != model.parameter_count():
        raise ValueError("Initial-model parameter count does not match the model graph.")
    if checkpoint.get("wavebit_options") is not None:
        raise ValueError("Initial-model checkpoint has unsupported WaveBit options.")

    state = checkpoint.get("model")
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise ValueError("Initial-model tensor names do not exactly match the model graph.")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Initial-model tensor {name!r} is not a tensor.")
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Initial-model tensor {name!r} is non-floating or non-finite.")

    model.load_state_dict(state, strict=True)
    return sha256_file(source)


def verify_model_bundle(output_root: str | Path) -> dict[str, Any]:
    """Verify the saved tokenizer, config, model weights, and their fingerprints."""
    root = Path(output_root).expanduser()
    artifact_dir = root / "artifacts"
    paths = _required_artifact_paths(artifact_dir)
    manifest_path = paths["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model manifest is missing: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != MODEL_MANIFEST_FORMAT:
        raise ValueError("Unsupported Q-Mini model manifest.")
    if manifest.get("stage") != "initialized_untrained":
        raise ValueError("The manifest does not describe an initialized, untrained model.")

    for key in ("tokenizer", "config", "initial_model", "report", "model_card"):
        entry = manifest.get("files", {}).get(paths[key].name)
        if not isinstance(entry, dict):
            raise ValueError(f"Model manifest is missing file metadata for {paths[key].name}.")
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
        if paths[key].stat().st_size != entry.get("size_bytes"):
            raise ValueError(f"Saved model file has an unexpected size: {paths[key].name}")
        if sha256_file(paths[key]) != entry.get("sha256"):
            raise ValueError(f"Saved model file failed SHA-256 verification: {paths[key].name}")

    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    config = AddSubConfig.from_dict(
        json.loads(paths["config"].read_text(encoding="utf-8"))
    )
    if tokenizer.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        raise ValueError("Saved tokenizer vocabulary does not match the model config.")
    checkpoint = torch.load(paths["initial_model"], map_location="cpu", weights_only=True)
    model = AddSubLanguageModel(config, seed=config.seed)
    tokenizer_sha = sha256_file(paths["tokenizer"])
    loaded_sha = load_initial_model_checkpoint(model, paths["initial_model"], tokenizer_sha)
    if loaded_sha != manifest.get("initial_model_sha256"):
        raise ValueError("Initial checkpoint SHA does not match the model manifest.")
    if model.parameter_count() != manifest.get("parameter_count"):
        raise ValueError("Saved model parameter count does not match its manifest.")
    if checkpoint.get("model") is None:
        raise ValueError("Initial model checkpoint has no model tensors.")

    sample_ids = tokenizer.encode("Q-Mini is ready.", add_special_tokens=False).ids
    sample = torch.tensor([sample_ids[:16]], dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        output = model(sample)
    if output["logits"].shape != (1, sample.shape[1], config.vocab_size):
        raise ValueError("Saved model failed the forward-pass shape check.")
    if not bool(torch.isfinite(output["logits"]).all()):
        raise ValueError("Saved model produced non-finite smoke-test logits.")

    return manifest


def initialize_model_artifacts(
    output_root: str | Path,
    config_template: str | Path,
    *,
    threads: int = 2,
) -> dict[str, Any]:
    """Create the complete, untrained Add/Sub model bundle without dataset access."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("CPU thread count must be a positive integer.")
    torch.set_num_threads(threads)

    root = Path(output_root).expanduser()
    if root.is_symlink():
        raise ValueError(f"Model workspace root must not be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)

    # Do not scan or rewrite cache contents. In particular, Hugging Face's
    # in-workspace snapshot-to-blob symlinks are left exactly as they are.
    layout = (
        root / "data",
        root / "cache",
        root / "artifacts",
        root / "evaluation",
        root / "runs" / "pretrain",
        root / "runs" / "sft",
    )
    for folder in layout:
        if folder.is_symlink():
            raise ValueError(f"Model workspace directory must not be a symlink: {folder}")
        folder.mkdir(parents=True, exist_ok=True)

    artifact_dir = root / "artifacts"
    paths = _required_artifact_paths(artifact_dir)
    existing = [path for path in paths.values() if path.exists() or path.is_symlink()]
    if len(existing) == len(paths):
        return verify_model_bundle(root)
    if existing:
        names = ", ".join(sorted(path.name for path in existing))
        raise FileExistsError(
            "A partial model bundle already exists and will not be overwritten: "
            f"{names}. Inspect the artifacts folder first."
        )

    tokenizer = build_bootstrap_tokenizer()
    tokenizer_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    config = _load_config(Path(config_template), tokenizer_vocab_size)

    model = AddSubLanguageModel(config, seed=config.seed).cpu()
    parameter_count = model.parameter_count()
    if parameter_count != config.estimated_parameter_count():
        raise RuntimeError(
            "Actual model parameter count does not match the config estimate: "
            f"{parameter_count:,} vs {config.estimated_parameter_count():,}."
        )
    if parameter_count > MAX_PARAMETER_BUDGET:
        raise RuntimeError("Constructed model exceeds the 2,000,000 parameter ceiling.")

    # Architecture smoke tests: causal sequence forward, one-token cached path,
    # finite logits, and a byte-tokenizer round-trip. These do not train weights.
    sample_text = "Q-Mini Add/Sub is ready for English training."
    sample_ids = tokenizer.encode(sample_text, add_special_tokens=False).ids
    if not sample_ids or len(sample_ids) > config.max_context:
        raise RuntimeError("Bootstrap text does not fit inside the configured model context.")
    input_ids = torch.tensor([sample_ids], dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        full_output = model(input_ids)
        caches = model.init_cache(batch_size=1, device="cpu")
        step_logits = None
        for token_id in sample_ids[:min(8, len(sample_ids))]:
            step_logits, caches = model.forward_step(
                torch.tensor([token_id], dtype=torch.long), caches
            )
    if full_output["logits"].shape != (1, len(sample_ids), config.vocab_size):
        raise RuntimeError("Model full-sequence smoke test returned an invalid logits shape.")
    if not bool(torch.isfinite(full_output["logits"]).all()):
        raise RuntimeError("Model full-sequence smoke test produced non-finite logits.")
    if step_logits is None or not bool(torch.isfinite(step_logits).all()):
        raise RuntimeError("Model cached one-token smoke test produced invalid logits.")

    config_payload = json.dumps(
        config.to_dict(), ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    staging = artifact_dir / ".qmini-model-bootstrap-staging"
    if staging.exists():
        raise FileExistsError(
            f"An incomplete model-bootstrap staging folder exists: {staging}. "
            "It was left untouched for inspection."
        )
    staging.mkdir()

    try:
        tokenizer_path = staging / paths["tokenizer"].name
        config_path = staging / paths["config"].name
        initial_path = staging / paths["initial_model"].name
        tokenizer.save(str(tokenizer_path))
        config_path.write_text(config_payload, encoding="utf-8")

        tokenizer_sha = sha256_file(tokenizer_path)
        initial_checkpoint = {
            "format": INITIAL_MODEL_FORMAT,
            "stage": "initialized_untrained",
            "step": 0,
            "created_at_utc": created_at,
            "initialization_seed": config.seed,
            "config": config.to_dict(),
            "parameter_count": parameter_count,
            "tokenizer_sha256": tokenizer_sha,
            "wavebit_options": None,
            "model": {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in model.state_dict().items()
            },
        }
        torch.save(initial_checkpoint, initial_path)

        model_card = f"""# Q-Mini Add/Sub 2M — initialized model

- Status: **initialized, not trained**
- Created (UTC): {created_at}
- Language target: English
- Actual parameters: {parameter_count:,} / {MAX_PARAMETER_BUDGET:,}
- Vocabulary: {config.vocab_size} byte-level tokens
- Maximum context: {config.max_context}
- Architecture: attention-free Causal Depthwise Convolution + Gated Diagonal State-Space block + sparse Top-1 two-expert SwiGLU MoE, with RMSNorm residual paths and tied input/output embeddings
- Initialization seed: {config.seed}
- Training data downloaded by this step: none
- Training steps completed: 0

## Files

- tokenizer.json: deterministic UTF-8 byte-level tokenizer with 256 byte symbols and eight Q-Mini control tokens; no corpus-trained merges
- model_config.json: exact architecture dimensions and parameter budget
- initial_model.pt: complete initialized PyTorch state dictionary and its config/tokenizer identity
- model_manifest.json: file hashes, architecture, and lifecycle status

## Important limitations

This is a complete **initialized model graph**, not a trained assistant. It will produce untrained output until pretraining and assistant fine-tuning are performed. The training graph uses five-symbol straight-through fake quantization while optimization is performed with PyTorch floating-point arithmetic. This bundle does not claim to be a complete multiply-free mobile inference runtime or a packed deployment export. A step-zero random initialization is intentionally not exported as a trained .qma archive.
"""
        report = {
            "format": "qmini-addsub-initialization-report-v1",
            "stage": "initialized_untrained",
            "created_at_utc": created_at,
            "device": "cpu",
            "cpu_threads": torch.get_num_threads(),
            "language": "English",
            "parameter_count": parameter_count,
            "parameter_budget": MAX_PARAMETER_BUDGET,
            "vocab_size": config.vocab_size,
            "max_context": config.max_context,
            "initialization_seed": config.seed,
            "dataset_downloaded": False,
            "training_started": False,
            "training_steps": 0,
            "smoke_tests": {
                "tokenizer_round_trip": True,
                "full_sequence_forward": True,
                "cached_one_token_forward": True,
                "finite_logits": True,
            },
            "tokenizer_sha256": tokenizer_sha,
        }
        report_path = staging / paths["report"].name
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        model_card_path = staging / paths["model_card"].name
        model_card_path.write_text(model_card, encoding="utf-8")

        file_entries = {}
        for key in ("tokenizer", "config", "initial_model", "report", "model_card"):
            staged_file = staging / paths[key].name
            file_entries[staged_file.name] = {
                "size_bytes": staged_file.stat().st_size,
                "sha256": sha256_file(staged_file),
            }
        manifest = {
            "format": MODEL_MANIFEST_FORMAT,
            "model_name": "Q-Mini Add/Sub 2M",
            "model_track": "qmini_addsub_2m_v1",
            "stage": "initialized_untrained",
            "created_at_utc": created_at,
            "language": "English",
            "parameter_count": parameter_count,
            "parameter_budget": MAX_PARAMETER_BUDGET,
            "config_sha256": file_entries[paths["config"].name]["sha256"],
            "tokenizer_sha256": tokenizer_sha,
            "initial_model_sha256": file_entries[paths["initial_model"].name]["sha256"],
            "architecture": {
                "type": "attention_free_causal_conv_gated_diagonal_ssm_sparse_moe",
                "blocks": config.n_layers,
                "d_model": config.d_model,
                "d_hidden": config.d_hidden,
                "experts": config.n_experts,
                "routing": "sparse_top1",
                "convolution": "causal_depthwise",
                "state_space": "gated_diagonal",
                "feed_forward": "SwiGLU",
                "normalization": "RMSNorm",
                "input_output_embeddings": "tied",
                "context_length": config.max_context,
            },
            "tokenizer": {
                "kind": "dataset_independent_utf8_byte_level_bpe",
                "vocabulary_size": config.vocab_size,
                "byte_symbols": 256,
                "control_tokens": SPECIAL_TOKENS,
                "learned_merges": 0,
                "can_be_replaced_after_training_starts": False,
            },
            "quantization": {
                "training_weight_method": "five_symbol straight-through fake quantization",
                "training_compute": "PyTorch floating point",
                "complete_multiply_free_mobile_runtime": False,
                "packed_trained_export_created": False,
            },
            "dataset": {
                "downloaded_by_initializer": False,
                "license_prompt_shown": False,
                "training_data_consumed": False,
            },
            "training": {
                "status": "not_started",
                "pretraining_steps": 0,
                "assistant_sft_steps": 0,
                "optimizer_state_created": False,
            },
            "files": file_entries,
        }
        manifest_path = staging / paths["manifest"].name
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Publish the manifest last: it is the completion marker for the bundle.
        for key in ("tokenizer", "config", "initial_model", "report", "model_card", "manifest"):
            source = staging / paths[key].name
            target = paths[key]
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Refusing to overwrite existing model artifact: {target}")
            os.replace(source, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return verify_model_bundle(root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize the complete Q-Mini Add/Sub model without downloading data or training."
    )
    parser.add_argument("--output-dir", required=True, help="Google Drive Add/Sub v1 workspace root.")
    parser.add_argument("--config-template", required=True, help="Repository 2M model profile JSON.")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    result = initialize_model_artifacts(
        args.output_dir,
        args.config_template,
        threads=args.threads,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
