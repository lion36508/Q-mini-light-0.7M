from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .config import AddSubConfig
from .model import AddSubLanguageModel
from .optim import assert_finite_training_state, build_adamw


def run_training_preflight(
    config_path: str | Path,
    *,
    device_name: str = "cpu",
    num_threads: int = 2,
    batch_size: int = 1,
    sequence_length: int = 16,
) -> dict[str, Any]:
    """Run one disposable forward/backward/optimizer step without touching datasets."""
    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
        raise ValueError("num_threads must be a positive integer.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 2:
        raise ValueError("sequence_length must be an integer of at least 2.")

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Add/Sub model profile not found: {path}")
    config = AddSubConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if sequence_length > config.max_context:
        raise ValueError("Preflight sequence length exceeds the model context.")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly selected but no CUDA device is available.")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Training preflight supports only CPU or CUDA devices.")
    torch.set_num_threads(num_threads)
    torch.manual_seed(config.seed)

    model = AddSubLanguageModel(config, seed=config.seed).to(device)
    actual_parameters = model.parameter_count()
    expected_parameters = config.estimated_parameter_count()
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"Parameter estimator mismatch: expected {expected_parameters:,}, "
            f"constructed {actual_parameters:,}."
        )
    if actual_parameters > 2_000_000:
        raise RuntimeError(
            f"Model has {actual_parameters:,} parameters; the absolute cap is 2,000,000."
        )

    token_count = batch_size * sequence_length
    input_ids = (
        torch.arange(token_count, dtype=torch.long, device=device)
        .reshape(batch_size, sequence_length)
        .remainder(config.vocab_size)
    )
    targets = input_ids.roll(shifts=-1, dims=1)
    optimizer = build_adamw(model, learning_rate=1e-4, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    result = model(input_ids)
    loss = (
        F.cross_entropy(
            result["logits"].float().reshape(-1, config.vocab_size),
            targets.reshape(-1),
        )
        + config.moe_balance_coef * result["aux_loss"]
        + config.router_z_loss_coef * result["router_z_loss"]
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Preflight produced a non-finite loss.")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    model.update_loss_free_router_bias(result["router_load"])
    assert_finite_training_state(model, optimizer, step=1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {
        "status": "training_engine_preflight_passed",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "precision": "float32",
        "parameter_count": actual_parameters,
        "parameter_budget": config.parameter_budget,
        "estimated_parameter_count": expected_parameters,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "loss": float(loss.detach().cpu()),
        "gradient_norm_before_clipping": float(grad_norm.detach().cpu()),
        "optimizer_step_completed": True,
        "synthetic_preflight_tokens_are_not_training_data": True,
    }
    if not math.isfinite(report["loss"]) or not math.isfinite(report["gradient_norm_before_clipping"]):
        raise FloatingPointError("Preflight diagnostics contain a non-finite value.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Add/Sub training graph with disposable synthetic tokens."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    args = parser.parse_args()
    report = run_training_preflight(
        args.config,
        device_name=args.device,
        num_threads=args.threads,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
