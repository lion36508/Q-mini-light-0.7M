from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
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




SOURCE_SNAPSHOT_FORMAT = "qmini-addsub-source-snapshot-v1"
SOURCE_DIRECTORIES = ("qmini_addsub", "qmini", "scripts")
SOURCE_ROOT_FILES = ("pyproject.toml", "README.md", "configs/qmini_addsub_2m.json")
SOURCE_EXTENSIONS = {
    ".py", ".cpp", ".cc", ".h", ".hpp", ".json", ".toml",
    ".md", ".txt", ".sh", ".yml", ".yaml", ".cmake",
}


def _source_snapshot_files(source_root: Path) -> list[Path]:
    """Collect the model, native bridge, training and data-pipeline source files."""
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError(f"Model source root is not a safe directory: {source_root}")
    selected: set[Path] = set()
    for directory_name in SOURCE_DIRECTORIES:
        directory = source_root / directory_name
        if not directory.is_dir() or directory.is_symlink():
            raise FileNotFoundError(
                f"Self-contained model source directory is missing or unsafe: {directory}"
            )
        for candidate in directory.rglob("*"):
            if candidate.is_symlink():
                raise ValueError(f"Model source snapshot refuses symlinks: {candidate}")
            if candidate.is_file() and candidate.suffix.lower() in SOURCE_EXTENSIONS:
                selected.add(candidate)
    for file_name in SOURCE_ROOT_FILES:
        candidate = source_root / file_name
        if candidate.is_file() and not candidate.is_symlink():
            selected.add(candidate)
    required = {
        source_root / "qmini_addsub" / "__init__.py",
        source_root / "qmini_addsub" / "model.py",
        source_root / "qmini_addsub" / "config.py",
        source_root / "qmini_addsub" / "train.py",
        source_root / "qmini_addsub" / "generate.py",
        source_root / "qmini_addsub" / "sft.py",
        source_root / "qmini" / "__init__.py",
        source_root / "pyproject.toml",
        source_root / "configs" / "qmini_addsub_2m.json",
    }
    missing = sorted(str(path.relative_to(source_root)) for path in required if path not in selected)
    if missing:
        raise FileNotFoundError(
            "Self-contained model source is missing required project files: "
            + ", ".join(missing)
        )
    return sorted(selected, key=lambda item: item.relative_to(source_root).as_posix())


def _write_source_snapshot(
    source_root: Path,
    destination: Path,
    source_commit: str | None,
) -> dict[str, Any]:
    """Copy the complete Add/Sub implementation and its required package sources."""
    files = _source_snapshot_files(source_root)
    destination.mkdir(parents=True, exist_ok=False)
    entries = []
    python_count = 0
    for source in files:
        relative = source.relative_to(source_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.suffix == ".py":
            ast.parse(target.read_text(encoding="utf-8"), filename=relative.as_posix())
            python_count += 1
        entries.append({
            "path": relative.as_posix(),
            "size_bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        })

    readme = destination / "MODEL_PROJECT_README.md"
    readme.write_text(
        """# Q-Mini Add/Sub 2M — self-contained source package

This folder contains the Python implementation and native source required for
the Add/Sub model graph, tokenizer/data utilities, CPU training, assistant SFT,
reference generation, evaluation, quantization/export, and OxCore native bridge.

## Main modules

- qmini_addsub/model.py — Causal Depthwise Convolution + Gated Diagonal SSM + sparse Top-1 SwiGLU experts.
- qmini_addsub/config.py — exact validated model dimensions and parameter-budget calculation.
- qmini_addsub/initialize.py — dataset-free initializer, source/hash verification and initial checkpoint loader.
- qmini_addsub/train.py — PyTorch pretraining engine and exact resume.
- qmini_addsub/sft.py — assistant supervised fine-tuning engine.
- qmini_addsub/generate.py — PyTorch reference chat/inference path.
- qmini_addsub/runtime.py and qmini_addsub/oxcore/ — separate additive/lookup runtime primitives and native CPU implementation.
- qmini_addsub/quantization.py — five-symbol training fake quantization and packed weight codebooks.
- qmini_addsub/evaluate.py, export.py and scripts/ — evaluation, export and data/token stream tooling.
- qmini/ — shared package modules needed by this repository build.
- configs/qmini_addsub_2m.json — source 2M profile template.

## Run the source package

Install compatible dependencies in the active Python environment:
python -m pip install 'numpy>=1.24' 'tokenizers>=0.15'

PyTorch must already be installed. Then, from this folder:

- Verify saved artifacts: python -m qmini_addsub.initialize --help
- Train later: python -m qmini_addsub.train --help
- Assistant fine-tuning: python -m qmini_addsub.sft --help
- Reference chat: python -m qmini_addsub.generate --help
- Evaluation: python -m qmini_addsub.evaluate --help
- Packed weight export: python -m qmini_addsub.export --help

Use the exact artifacts in the parent artifacts folder: tokenizer.json,
model_config.json, and initial_model.pt. Do not replace the tokenizer or
config after training begins.

## Lifecycle

The initial checkpoint is a complete step-zero model state, not trained weights.
It requires pretraining and assistant SFT before its outputs are useful. PyTorch
training uses floating-point optimization with a five-symbol straight-through
weight quantizer. This source snapshot does not claim that a complete
multiply-free mobile inference engine has been finished.
""",
        encoding="utf-8",
    )
    entries.append({
        "path": "MODEL_PROJECT_README.md",
        "size_bytes": readme.stat().st_size,
        "sha256": sha256_file(readme),
    })
    entries.sort(key=lambda item: item["path"])
    digest_payload = {
        "format": SOURCE_SNAPSHOT_FORMAT,
        "source_commit": source_commit,
        "files": entries,
    }
    snapshot = {
        **digest_payload,
        "file_count": len(entries),
        "python_file_count": python_count,
        "sha256": _sha256_json(digest_payload),
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot["manifest_sha256"] = sha256_file(destination / "source_manifest.json")
    return snapshot


def _verify_source_snapshot(source_dir: Path, expected_sha256: str) -> dict[str, Any]:
    manifest_path = source_dir / "source_manifest.json"
    if source_dir.is_symlink() or not source_dir.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError("The self-contained model source snapshot is incomplete.")
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("Self-contained source manifest SHA-256 mismatch.")
    snapshot = json.loads(manifest_path.read_text(encoding="utf-8"))
    if snapshot.get("format") != SOURCE_SNAPSHOT_FORMAT:
        raise ValueError("Unsupported self-contained source snapshot format.")
    files = snapshot.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Self-contained source snapshot contains no indexed files.")
    seen = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Source snapshot file index contains an invalid entry.")
        relative = Path(entry.get("path", ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("Source snapshot contains an unsafe relative path.")
        normalized = relative.as_posix()
        if normalized in seen:
            raise ValueError(f"Source snapshot contains duplicate path: {normalized}")
        seen.add(normalized)
        target = source_dir / relative
        if target.is_symlink() or not target.is_file():
            raise FileNotFoundError(f"Self-contained source file is missing or unsafe: {target}")
        resolved = target.resolve(strict=True)
        resolved_root = source_dir.resolve(strict=True)
        if resolved_root not in resolved.parents:
            raise ValueError(f"Source snapshot path escapes its root: {target}")
        if target.stat().st_size != entry.get("size_bytes"):
            raise ValueError(f"Source snapshot file size mismatch: {normalized}")
        if sha256_file(target) != entry.get("sha256"):
            raise ValueError(f"Source snapshot file hash mismatch: {normalized}")

    digest_payload = {
        "format": snapshot["format"],
        "source_commit": snapshot.get("source_commit"),
        "files": files,
    }
    if _sha256_json(digest_payload) != snapshot.get("sha256"):
        raise ValueError("Source snapshot index digest mismatch.")
    return snapshot


def verify_self_contained_model_source(output_root: str | Path) -> dict[str, Any]:
    """Import and re-verify the bundle using only the Python source copied into Drive."""
    root = Path(output_root).expanduser().resolve()
    source_dir = root / "artifacts" / "source_code"
    manifest = json.loads(
        (root / "artifacts" / "model_manifest.json").read_text(encoding="utf-8")
    )
    expected_source_sha = manifest.get("source_snapshot", {}).get("manifest_sha256")
    snapshot = _verify_source_snapshot(source_dir, expected_source_sha)

    verification_code = r'''
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1]).resolve()
source_root = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(source_root))

import qmini_addsub
module_path = Path(qmini_addsub.__file__).resolve()
if source_root not in module_path.parents:
    raise RuntimeError(f"Imported model package outside saved source folder: {module_path}")

from qmini_addsub import model, train, sft, generate, evaluate, export, runtime
from qmini_addsub.initialize import verify_model_bundle

result = verify_model_bundle(workspace)
print(json.dumps({
    "source_package": str(module_path),
    "parameter_count": result["parameter_count"],
    "source_commit": result["source_snapshot"]["source_commit"],
    "isolated_source_import": True,
    "model_bundle_verified": True,
}, sort_keys=True))
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(source_dir) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    completed = subprocess.run(
        [sys.executable, "-c", verification_code, str(root), str(source_dir)],
        cwd=str(source_dir),
        env=environment,
        check=True,
        text=True,
        capture_output=True,
        timeout=180,
    )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("Self-contained model source verification returned no report.")
    result = json.loads(output_lines[-1])
    if result.get("isolated_source_import") is not True or result.get("model_bundle_verified") is not True:
        raise RuntimeError("Saved model source failed isolated import/verification.")
    return {
        "source_commit": snapshot.get("source_commit"),
        "source_file_count": snapshot["file_count"],
        "python_file_count": snapshot["python_file_count"],
        "source_manifest_sha256": snapshot["manifest_sha256"],
        **result,
    }


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
        "parameter_layout": artifact_dir / "parameter_layout.json",
        "source_code": artifact_dir / "source_code",
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
    """Verify source code, parameter layout, tokenizer, config and complete model weights."""
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

    for key in ("tokenizer", "config", "initial_model", "parameter_layout", "report", "model_card"):
        entry = manifest.get("files", {}).get(paths[key].name)
        if not isinstance(entry, dict):
            raise ValueError(f"Model manifest is missing file metadata for {paths[key].name}.")
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
        if paths[key].stat().st_size != entry.get("size_bytes"):
            raise ValueError(f"Saved model file has an unexpected size: {paths[key].name}")
        if sha256_file(paths[key]) != entry.get("sha256"):
            raise ValueError(f"Saved model file failed SHA-256 verification: {paths[key].name}")

    source_info = manifest.get("source_snapshot")
    if not isinstance(source_info, dict):
        raise ValueError("Model manifest does not contain a self-contained source snapshot.")
    source_snapshot = _verify_source_snapshot(
        paths["source_code"], source_info.get("manifest_sha256")
    )
    if source_snapshot.get("sha256") != source_info.get("sha256"):
        raise ValueError("Source snapshot index does not match the model manifest.")
    if source_snapshot.get("source_commit") != source_info.get("source_commit"):
        raise ValueError("Source snapshot commit does not match the model manifest.")

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

    expected_layout = []
    offset = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        expected_layout.append({
            "name": name,
            "shape": list(parameter.shape),
            "numel": count,
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "flat_offset_start": offset,
            "flat_offset_end": offset + count,
        })
        offset += count
    layout = json.loads(paths["parameter_layout"].read_text(encoding="utf-8"))
    if layout.get("format") != "qmini-addsub-parameter-layout-v1":
        raise ValueError("Unsupported parameter layout format.")
    if layout.get("parameters") != expected_layout or layout.get("parameter_count") != offset:
        raise ValueError("Saved parameter layout does not exactly match the model's named tensors.")
    if offset != model.parameter_count():
        raise ValueError("Named parameter layout does not account for every model parameter.")

    sample_ids = tokenizer.encode("Q-Mini is ready.", add_special_tokens=False).ids
    sample = torch.tensor([sample_ids[:16]], dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        output = model(sample)
        caches = model.init_cache(batch_size=1, device="cpu")
        step_logits = None
        for token_id in sample_ids[:min(8, len(sample_ids))]:
            step_logits, caches = model.forward_step(
                torch.tensor([token_id], dtype=torch.long), caches
            )
    if output["logits"].shape != (1, sample.shape[1], config.vocab_size):
        raise ValueError("Saved model failed the full-sequence forward-pass shape check.")
    if not bool(torch.isfinite(output["logits"]).all()):
        raise ValueError("Saved model produced non-finite full-sequence logits.")
    if step_logits is None or not bool(torch.isfinite(step_logits).all()):
        raise ValueError("Saved model failed the cached one-token forward test.")

    return manifest



def initialize_model_artifacts(
    output_root: str | Path,
    config_template: str | Path,
    *,
    threads: int = 2,
    source_root: str | Path | None = None,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Create the complete, untrained Add/Sub model bundle without dataset access."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("CPU thread count must be a positive integer.")
    torch.set_num_threads(threads)

    root = Path(output_root).expanduser()
    resolved_config_template = Path(config_template).expanduser().resolve(strict=True)
    inferred_source_root = resolved_config_template.parent.parent
    requested_source_root = inferred_source_root if source_root is None else Path(source_root).expanduser()
    if requested_source_root.is_symlink():
        raise ValueError("Model source root must not be a symlink.")
    source_root_path = requested_source_root.resolve(strict=True)
    if not source_root_path.is_dir() or source_root_path not in resolved_config_template.parents:
        raise ValueError("The 2M config template must be inside the selected source repository.")
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
        verified_manifest = verify_model_bundle(root)
        source_verification = verify_self_contained_model_source(root)
        if source_verification["parameter_count"] != verified_manifest["parameter_count"]:
            raise RuntimeError("Copied source verification loaded a different parameter count.")
        verified_manifest["source_verification"] = source_verification
        return verified_manifest
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
    parameter_layout = []
    parameter_offset = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        parameter_layout.append({
            "name": name,
            "shape": list(parameter.shape),
            "numel": count,
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "flat_offset_start": parameter_offset,
            "flat_offset_end": parameter_offset + count,
        })
        parameter_offset += count
    if parameter_offset != parameter_count:
        raise RuntimeError("Named parameter layout does not match the actual parameter count.")
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
        source_snapshot = _write_source_snapshot(
            source_root_path,
            staging / "source_code",
            source_commit,
        )
        tokenizer_path = staging / paths["tokenizer"].name
        config_path = staging / paths["config"].name
        initial_path = staging / paths["initial_model"].name
        tokenizer.save(str(tokenizer_path))
        config_path.write_text(config_payload, encoding="utf-8")
        parameter_layout_path = staging / paths["parameter_layout"].name
        parameter_layout_path.write_text(
            json.dumps({
                "format": "qmini-addsub-parameter-layout-v1",
                "parameter_count": parameter_count,
                "parameters": parameter_layout,
            }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

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
- parameter_layout.json: every named parameter, shape, element count, and flat offset range
- source_code/: complete Add/Sub Python/native source, training/inference modules, scripts, config template, and source hash manifest
- model_manifest.json: file hashes, architecture, source snapshot identity, and lifecycle status

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
            "source_snapshot_sha256": source_snapshot["manifest_sha256"],
            "source_commit": source_commit,
            "parameter_layout_sha256": sha256_file(parameter_layout_path),
        }
        report_path = staging / paths["report"].name
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        model_card_path = staging / paths["model_card"].name
        model_card_path.write_text(model_card, encoding="utf-8")

        file_entries = {}
        for key in ("tokenizer", "config", "initial_model", "parameter_layout", "report", "model_card"):
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
            "source_snapshot": {
                "root": "source_code",
                "manifest_sha256": source_snapshot["manifest_sha256"],
                "sha256": source_snapshot["sha256"],
                "source_commit": source_snapshot["source_commit"],
                "file_count": source_snapshot["file_count"],
                "python_file_count": source_snapshot["python_file_count"],
            },
            "parameter_layout": {
                "path": "parameter_layout.json",
                "parameter_count": parameter_count,
                "tensor_count": len(parameter_layout),
                "file_sha256": file_entries[paths["parameter_layout"].name]["sha256"],
            },
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
        os.replace(staging / "source_code", paths["source_code"])
        for key in ("tokenizer", "config", "initial_model", "parameter_layout", "report", "model_card", "manifest"):
            source = staging / paths[key].name
            target = paths[key]
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Refusing to overwrite existing model artifact: {target}")
            os.replace(source, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    verified_manifest = verify_model_bundle(root)
    source_verification = verify_self_contained_model_source(root)
    if source_verification["parameter_count"] != verified_manifest["parameter_count"]:
        raise RuntimeError("Copied source verification loaded a different parameter count.")
    verified_manifest["source_verification"] = source_verification
    return verified_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize the complete Q-Mini Add/Sub model without downloading data or training."
    )
    parser.add_argument("--output-dir", required=True, help="Google Drive Add/Sub v1 workspace root.")
    parser.add_argument("--config-template", required=True, help="Repository 2M model profile JSON.")
    parser.add_argument("--source-root", default=None, help="Repository root to snapshot; inferred from the config template when omitted.")
    parser.add_argument("--source-commit", default=None, help="Exact Git commit for the source snapshot.")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    result = initialize_model_artifacts(
        args.output_dir,
        args.config_template,
        threads=args.threads,
        source_root=args.source_root,
        source_commit=args.source_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
