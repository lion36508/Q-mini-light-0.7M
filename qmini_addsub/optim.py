from __future__ import annotations

import math

import torch
from torch import nn


def build_adamw(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.1,
) -> torch.optim.AdamW:
    """AdamW with decay only on matrix/tensor weights, not biases or norm scales."""
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative.")

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
    if not math.isfinite(peak_learning_rate) or peak_learning_rate <= 0:
        raise ValueError("peak_learning_rate must be finite and positive.")
    if not math.isfinite(warmup_fraction) or not 0.0 < warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in (0, 1).")
    if not math.isfinite(minimum_ratio) or not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in [0, 1].")

    warmup_steps = min(total_steps, max(1, int(total_steps * warmup_fraction)))
    if step <= warmup_steps:
        return peak_learning_rate * step / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return peak_learning_rate * (
        minimum_ratio + (1.0 - minimum_ratio) * cosine
    )
