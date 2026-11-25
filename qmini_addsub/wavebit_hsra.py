"""WaveBit-HSRA components for the single evolving Q-Mini Add/Sub model.

The PyTorch path is a training surrogate, not a claim of float-free execution.
The integer routines are reference kernels for isolated primitives only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


WEIGHT_LEVELS = (-1.5, -0.5, 0.0, 0.5, 1.5)
POLARITY_BY_CODE = (-1, -1, 0, 1, 1)
PHASE_BY_CODE = (1, 0, 0, 0, 1)

DEFAULT_WAVEBIT_OPTIONS = {
    "group_size": 32,
    "rotation": 1,
    "gamma": 4,
    "ewgs_strength": 0.5,
    "context_threshold": 0.5,
    "context_temperature": 0.75,
    "state_limit": 16.0,
}


def validate_wavebit_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    if options is None:
        return None
    if not isinstance(options, dict):
        raise TypeError("wavebit_hsra options must be a dictionary.")
    unknown = set(options) - set(DEFAULT_WAVEBIT_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown WaveBit-HSRA options: {sorted(unknown)}")
    result = {**DEFAULT_WAVEBIT_OPTIONS, **options}
    if isinstance(result["group_size"], bool) or not isinstance(result["group_size"], int) or result["group_size"] < 1:
        raise ValueError("WaveBit group_size must be a positive integer.")
    if isinstance(result["gamma"], bool) or not isinstance(result["gamma"], int) or not 1 <= result["gamma"] <= 15:
        raise ValueError("WaveBit gamma must be an integer in [1, 15].")
    if isinstance(result["rotation"], bool) or not isinstance(result["rotation"], int):
        raise ValueError("WaveBit rotation must be an integer.")
    if not math.isfinite(float(result["ewgs_strength"])) or not 0.0 <= float(result["ewgs_strength"]) <= 2.0:
        raise ValueError("WaveBit ewgs_strength must be finite and in [0, 2].")
    for key in ("context_threshold", "context_temperature", "state_limit"):
        value = float(result[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"WaveBit {key} must be finite and positive.")
        result[key] = value
    result["ewgs_strength"] = float(result["ewgs_strength"])
    return result


def groupwise_scale(weight: torch.Tensor, group_size: int = 32, multiplier: float = 1.4) -> torch.Tensor:
    """Detached per-output-row scale, independently estimated for each input group."""
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise TypeError("WaveBit weights must be floating-point tensors.")
    if weight.ndim < 2 or weight.numel() == 0:
        raise ValueError("WaveBit weights must have at least two non-empty dimensions.")
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size must be a positive integer.")
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("scale multiplier must be finite and positive.")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("WaveBit weights contain non-finite values.")
    flat = weight.detach().reshape(weight.shape[0], -1)
    scaled = torch.empty_like(flat)
    for start in range(0, flat.shape[1], group_size):
        stop = min(start + group_size, flat.shape[1])
        block = flat[:, start:stop]
        scale = (block.abs().mean(dim=1, keepdim=True) * multiplier).clamp_min(1e-8)
        scaled[:, start:stop] = scale.expand_as(block)
    return scaled.reshape_as(weight).detach()


def _hard_five_symbol(normalized: torch.Tensor) -> torch.Tensor:
    x = normalized.clamp(-1.5, 1.5)
    return torch.where(
        x <= -1.0, -1.5,
        torch.where(x <= -0.25, -0.5,
                    torch.where(x <= 0.25, 0.0,
                                torch.where(x <= 1.0, 0.5, 1.5))),
    )


class _EWGSQuantize(torch.autograd.Function):
    """Hard five-symbol forward with a bounded EWGS-style straight-through gradient."""
    @staticmethod
    def forward(ctx, weight, hard_value, normalized, hard_normalized, strength):
        ctx.save_for_backward(normalized, hard_normalized)
        ctx.strength = float(strength)
        return hard_value

    @staticmethod
    def backward(ctx, grad_output):
        normalized, hard_normalized = ctx.saved_tensors
        correction = ctx.strength * torch.sign(grad_output) * (normalized - hard_normalized)
        factor = (1.0 + correction).clamp(0.5, 1.5)
        return grad_output * factor, None, None, None, None


def wavebit_quantize(
    weight: torch.Tensor,
    group_size: int = 32,
    ewgs_strength: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the existing five stored symbols as polarity p and phase bit g.

    One master parameter per connection is retained. No model parameters are
    added for the phase bit, group scales or the holographic state.
    """
    if not 0.0 <= ewgs_strength <= 2.0 or not math.isfinite(ewgs_strength):
        raise ValueError("ewgs_strength must be finite and in [0, 2].")
    scale = groupwise_scale(weight, group_size=group_size)
    normalized = weight / scale
    hard_normalized = _hard_five_symbol(normalized)
    hard_value = hard_normalized * scale
    quantized = _EWGSQuantize.apply(
        weight, hard_value, normalized.detach(), hard_normalized.detach(), ewgs_strength
    )
    normalized_surrogate = quantized / scale
    polarity_hard = torch.sign(hard_normalized)
    polarity_soft = torch.tanh(normalized_surrogate)
    polarity = polarity_hard.detach() + polarity_soft - polarity_soft.detach()
    phase_hard = (hard_normalized.abs() > 1.0).to(weight.dtype)
    phase_soft = torch.sigmoid((normalized_surrogate.abs() - 1.0) * 4.0)
    phase = phase_hard.detach() + phase_soft - phase_soft.detach()
    return quantized, polarity, phase, scale


def wavebit_context_gain(context: torch.Tensor) -> torch.Tensor:
    """Hard forward gains {0.5, 1, 2} with a smooth differentiable surrogate."""
    hard = torch.where(
        context < -0.5, torch.full_like(context, 0.5),
        torch.where(context > 0.5, torch.full_like(context, 2.0), torch.ones_like(context)),
    )
    soft = torch.exp2(context)
    return hard.detach() + soft - soft.detach()


def resize_context(context: torch.Tensor, width: int) -> torch.Tensor:
    """Deterministic phase-slot remapping for an expert's hidden width."""
    if context.shape[-1] == width:
        return context
    indices = torch.arange(width, device=context.device) % context.shape[-1]
    return context.index_select(-1, indices)


def wavebit_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    context: torch.Tensor,
    *,
    group_size: int = 32,
    ewgs_strength: float = 0.5,
) -> torch.Tensor:
    """WaveBit training projection, split into static and context-sensitive paths."""
    if inputs.shape != context.shape:
        raise ValueError("WaveBit linear context must match the input tensor shape.")
    _, polarity, phase, scale = wavebit_quantize(
        weight, group_size=group_size, ewgs_strength=ewgs_strength
    )
    static_weight = polarity * scale * 0.5
    dynamic_weight = polarity * phase * scale
    gain_delta = wavebit_context_gain(context) - 0.5
    output = F.linear(inputs, static_weight, None)
    output = output + F.linear(inputs * gain_delta, dynamic_weight, None)
    if bias is not None:
        output = output + bias
    return output


def wavebit_depthwise_conv1d(
    padded_inputs: torch.Tensor,
    padded_context: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    dilation: int,
    group_size: int,
    ewgs_strength: float,
) -> torch.Tensor:
    """Context-modulated depthwise convolution surrogate for WaveBit training."""
    if padded_inputs.shape != padded_context.shape:
        raise ValueError("Padded WaveBit inputs and contexts must have identical shapes.")
    _, polarity, phase, scale = wavebit_quantize(
        weight, group_size=group_size, ewgs_strength=ewgs_strength
    )
    static_weight = polarity * scale * 0.5
    dynamic_weight = polarity * phase * scale
    gain_delta = wavebit_context_gain(padded_context) - 0.5
    static = F.conv1d(
        padded_inputs, static_weight, None, dilation=dilation, groups=padded_inputs.shape[1]
    )
    dynamic = F.conv1d(
        padded_inputs * gain_delta, dynamic_weight, None,
        dilation=dilation, groups=padded_inputs.shape[1]
    )
    return static + dynamic + bias.view(1, -1, 1)


class HolographicRotationMemory(torch.nn.Module):
    """Parameter-free fixed-width recurrent context memory, O(width) per stream."""
    def __init__(
        self, width: int, rotation: int = 1, gamma: int = 4,
        threshold: float = 0.5, temperature: float = 0.75, state_limit: float = 16.0,
    ):
        super().__init__()
        if width < 1 or gamma < 1 or threshold <= 0 or temperature <= 0 or state_limit <= 0:
            raise ValueError("invalid WaveBit holographic memory settings")
        self.width = int(width)
        self.rotation = int(rotation) % self.width
        self.gamma = int(gamma)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.state_limit = float(state_limit)

    def step(self, inputs: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2 or inputs.shape[-1] != self.width or state.shape != inputs.shape:
            raise ValueError("WaveBit memory expects matching [batch, width] input and state")
        hard = torch.where(
            state > self.threshold, torch.ones_like(state),
            torch.where(state < -self.threshold, -torch.ones_like(state), torch.zeros_like(state)),
        )
        soft = torch.tanh(state / self.temperature)
        context = hard.detach() + soft - soft.detach()
        rotated = torch.roll(state, shifts=self.rotation, dims=-1)
        retention = 1.0 - (2.0 ** -self.gamma)
        updated = (rotated * retention + torch.tanh(inputs)).clamp(
            -self.state_limit, self.state_limit
        )
        return context, updated

    def forward(
        self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.width or inputs.shape[1] < 1:
            raise ValueError("WaveBit memory expects [batch, sequence, width]")
        state = inputs.new_zeros((inputs.shape[0], self.width)) if initial_state is None else initial_state
        if state.shape != (inputs.shape[0], self.width):
            raise ValueError("WaveBit memory initial state has incompatible shape")
        contexts = []
        for index in range(inputs.shape[1]):
            context, state = self.step(inputs[:, index, :], state)
            contexts.append(context)
        return torch.stack(contexts, dim=1), state


def integer_wavebit_projection(
    activation_codes: np.ndarray,
    weight_codes: np.ndarray,
    context_codes: np.ndarray,
    bias_half_units: np.ndarray | None = None,
) -> np.ndarray:
    """Integer reference accumulation in half-units (no multiply/dot/matmul)."""
    x = np.asarray(activation_codes)
    codes = np.asarray(weight_codes)
    context = np.asarray(context_codes)
    if x.dtype.kind not in "iu" or x.ndim < 1:
        raise TypeError("activation codes must be an integer tensor")
    if codes.dtype.kind not in "iu" or codes.ndim != 2 or np.any(codes < 0) or np.any(codes > 4):
        raise ValueError("weight codes must be [out, in] integers in [0, 4]")
    if context.dtype.kind not in "iu" or context.shape != x.shape:
        raise ValueError("context codes must match activation shape")
    if x.shape[-1] != codes.shape[-1] or np.any((context < -1) | (context > 1)):
        raise ValueError("WaveBit input/context dimensions or context values are invalid")
    flat_x = x.reshape(-1, x.shape[-1]).astype(np.int64, copy=False)
    flat_c = context.reshape(-1, context.shape[-1]).astype(np.int8, copy=False)
    polarity = np.asarray(POLARITY_BY_CODE, dtype=np.int8)[codes]
    phase = np.asarray(PHASE_BY_CODE, dtype=np.int8)[codes]
    output = np.zeros((flat_x.shape[0], codes.shape[0]), dtype=np.int64)
    for row in range(flat_x.shape[0]):
        for out_index in range(codes.shape[0]):
            acc = 0
            for feature in range(flat_x.shape[1]):
                p = int(polarity[out_index, feature])
                if p == 0:
                    continue
                term = int(flat_x[row, feature])
                if p < 0:
                    term = -term
                if phase[out_index, feature] == 0:
                    acc += term
                else:
                    c = int(flat_c[row, feature])
                    if c < 0:
                        acc += term
                    elif c == 0:
                        acc += term
                        acc += term
                    else:
                        acc += term << 2
            output[row, out_index] = acc
    if bias_half_units is not None:
        bias = np.asarray(bias_half_units)
        if bias.dtype.kind not in "iu" or bias.shape != (codes.shape[0],):
            raise ValueError("bias_half_units must be an integer output vector")
        output = np.add(output, bias.astype(np.int64, copy=False))
    return output.reshape(x.shape[:-1] + (codes.shape[0],))


def holographic_state_step_int(
    state: np.ndarray, input_codes: np.ndarray, rotation: int = 1,
    gamma: int = 4, threshold: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Integer rotate/shift/subtract memory reference; np.roll is not a cycle claim."""
    h = np.asarray(state)
    u = np.asarray(input_codes)
    if h.dtype.kind not in "iu" or u.dtype.kind not in "iu" or h.shape != u.shape or h.ndim < 1:
        raise ValueError("integer WaveBit state and input must have matching integer shapes")
    if isinstance(gamma, bool) or not isinstance(gamma, int) or gamma < 1 or threshold < 1:
        raise ValueError("gamma and threshold must be positive integers")
    rotated = np.roll(h.astype(np.int64, copy=False), shift=int(rotation), axis=-1)
    delta = np.right_shift(np.abs(rotated), gamma)
    retained = np.where(rotated >= 0, rotated - delta, rotated + delta)
    updated = retained + u.astype(np.int64, copy=False)
    context = np.where(h < -threshold, -1, np.where(h > threshold, 1, 0)).astype(np.int8)
    return context, updated


@dataclass
class CapacitorCharge:
    """Optional integer optimizer-state experiment; not enabled in default training."""
    codes: np.ndarray
    threshold: int = 1024

    def __post_init__(self):
        self.codes = np.asarray(self.codes, dtype=np.uint8).copy()
        if np.any(self.codes > 4):
            raise ValueError("Capacitor charge codes must be in [0, 4]")
        if isinstance(self.threshold, bool) or self.threshold < 1:
            raise ValueError("Capacitor charge threshold must be positive")
        self.charge = np.zeros(self.codes.shape, dtype=np.int64)

    def step(self, signed_impulse: np.ndarray) -> np.ndarray:
        impulse = np.asarray(signed_impulse)
        if impulse.dtype.kind not in "iu" or impulse.shape != self.codes.shape:
            raise ValueError("Capacitor impulses must be an integer array matching codes")
        self.charge += impulse.astype(np.int64, copy=False)
        positive = self.charge >= self.threshold
        negative = self.charge <= -self.threshold
        current = self.codes.astype(np.int16)
        current = np.where(positive, np.minimum(current + 1, 4), current)
        current = np.where(negative, np.maximum(current - 1, 0), current)
        self.codes = current.astype(np.uint8)
        self.charge = np.where(positive, self.charge - self.threshold, self.charge)
        self.charge = np.where(negative, self.charge + self.threshold, self.charge)
        return self.codes.copy()
