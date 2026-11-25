from __future__ import annotations

import math

import torch
from torch import nn


def build_adamw(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """AdamW with decay only on matrix/tensor weights, not biases or norm scales."""
    for name, value in (
        ("learning_rate", learning_rate),
        ("weight_decay", weight_decay),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number, not a boolean or string.")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for _name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("Model has no trainable parameters.")

    return torch.optim.AdamW(groups, lr=learning_rate)


def cosine_warmup_lr(
    step: int,
    total_steps: int,
    peak_learning_rate: float,
    warmup_fraction: float = 0.03,
    minimum_ratio: float = 0.1,
) -> float:
    """Linear warm-up followed by cosine decay to a non-zero LR floor."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("step must be a positive integer.")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError("total_steps must be a positive integer.")
    if step > total_steps:
        raise ValueError("step cannot exceed total_steps.")
    for name, value in (
        ("peak_learning_rate", peak_learning_rate),
        ("warmup_fraction", warmup_fraction),
        ("minimum_ratio", minimum_ratio),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number, not a boolean or string.")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if peak_learning_rate <= 0:
        raise ValueError("peak_learning_rate must be positive.")
    if not 0.0 < warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in (0, 1).")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in [0, 1].")

    warmup_steps = min(total_steps, max(1, int(total_steps * warmup_fraction)))
    if step <= warmup_steps:
        return peak_learning_rate * step / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return peak_learning_rate * (
        minimum_ratio + (1.0 - minimum_ratio) * cosine
    )



def assert_finite_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    """Fail immediately if an optimizer update corrupts model or moment state."""
    checks: list[torch.Tensor] = []
    labels: list[str] = []
    for name, parameter in model.named_parameters():
        checks.append(torch.isfinite(parameter).all())
        labels.append(f"model parameter {name!r}")
        state = optimizer.state.get(parameter, {})
        for state_name, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                checks.append(torch.isfinite(value).all())
                labels.append(f"optimizer state {state_name!r} for {name!r}")

    if not checks:
        raise ValueError("Cannot validate a model without parameters.")
    finite = torch.stack(checks)
    if not bool(finite.all()):
        first_bad = int((~finite).nonzero(as_tuple=False)[0, 0].item())
        raise FloatingPointError(
            f"Non-finite {labels[first_bad]} after optimizer step {step}."
        )



def reconcile_metrics_file(
    path: str,
    expected_identity: dict[str, object],
    maximum_committed_step: int,
    run_name: str,
) -> int:
    """Validate metrics provenance and discard rows beyond the resumed checkpoint.

    Metric rows can be written more frequently than checkpoints. On recovery,
    rows newer than the selected checkpoint describe updates that will be
    repeated, so they are atomically truncated rather than mixed with the
    resumed trajectory. Different data/tokenizer/schedule identities fail closed.
    """
    from pathlib import Path
    import json

    metrics_path = Path(path)
    if maximum_committed_step < 0:
        raise ValueError("maximum_committed_step must be non-negative.")
    if not metrics_path.exists():
        return 0

    rows = []
    try:
        for line_number, line in enumerate(
            metrics_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"{run_name} metrics row {line_number} must be a JSON object."
                )
            rows.append(row)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{run_name} metrics file is not valid JSONL.") from exc

    kept = []
    discarded = 0
    previous_step = 0
    run_contract = expected_identity.get("run_contract")
    planned_steps = (
        run_contract.get("planned_steps")
        if isinstance(run_contract, dict)
        else None
    )
    for row in rows:
        for key, expected_value in expected_identity.items():
            if key not in row or row[key] != expected_value:
                raise ValueError(
                    f"{run_name} metrics belong to a different dataset, tokenizer "
                    f"or run contract ({key})."
                )
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError(f"{run_name} metrics contain an invalid step.")
        if step <= previous_step:
            raise ValueError(
                f"{run_name} metrics steps must be strictly increasing."
            )
        if planned_steps is not None and step > planned_steps:
            raise ValueError(
                f"{run_name} metrics contain a step beyond the planned run horizon."
            )
        previous_step = step
        if step > maximum_committed_step:
            discarded += 1
        else:
            kept.append(row)

    if discarded:
        temporary = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                for row in kept:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            temporary.replace(metrics_path)
        finally:
            temporary.unlink(missing_ok=True)
    return discarded
