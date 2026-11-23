from __future__ import annotations

import math

import numpy as np
import torch


WEIGHT_SYMBOLS = (-1.5, -0.5, 0.0, 0.5, 1.5)
SYMBOL_COUNT = len(WEIGHT_SYMBOLS)
BITS_PER_WEIGHT = 3
FORMAT_NAME = "qmini-addsub-v1"


def quantize_weight_ste(weight: torch.Tensor) -> torch.Tensor:
    """STE quantization to the exact five-symbol deployment codebook.

    Midpoint ties choose the lower symbol, matching NumPy argmin in
    quantize_to_codes() and therefore the packed export path.
    """
    clipped = weight.clamp(-1.5, 1.5)
    quantized = torch.where(
        clipped <= -1.0,
        -1.5,
        torch.where(
            clipped <= -0.25,
            -0.5,
            torch.where(
                clipped <= 0.25,
                0.0,
                torch.where(clipped <= 1.0, 0.5, 1.5),
            ),
        ),
    )
    return weight + (quantized - weight).detach()


def initialize_five_symbol_weight_(
    weight: torch.Tensor,
    active_per_row: int = 4,
) -> torch.Tensor:
    """Initialize each row with sparse, exact five-codebook values.

    Four +/-0.5 entries per row give variance 1/fan_in, matching Kaiming
    fan-in variance while ensuring the first quantized forward pass is not
    an all-zero projection. This is a training-only initialization helper.
    """
    if weight.ndim < 2 or any(size < 1 for size in weight.shape):
        raise ValueError("Weight must have at least two non-empty dimensions.")
    if not weight.is_floating_point():
        raise TypeError("Five-symbol initialization requires a floating-point weight.")
    if isinstance(active_per_row, bool) or not isinstance(active_per_row, int):
        raise TypeError("active_per_row must be an integer.")
    if active_per_row < 1:
        raise ValueError("active_per_row must be positive.")

    flat = weight.reshape(weight.shape[0], -1)
    active = min(active_per_row, flat.shape[1])
    with torch.no_grad():
        flat.zero_()
        indices = torch.rand_like(flat).topk(active, dim=1).indices
        signs = torch.randint(
            0, 2, indices.shape, device=weight.device, dtype=torch.int64
        )
        values = (signs.to(dtype=weight.dtype) * 2.0 - 1.0) * 0.5
        flat.scatter_(1, indices, values)
    return weight


def quantize_to_codes(weight: np.ndarray) -> np.ndarray:
    """Map a floating weight array to codes 0..4 in the documented alphabet."""
    values = np.asarray(weight, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite weights cannot be quantized.")
    clipped = np.clip(values, -1.5, 1.5)
    levels = np.asarray(WEIGHT_SYMBOLS, dtype=np.float32)
    distances = np.abs(clipped[..., None] - levels)
    return np.argmin(distances, axis=-1).astype(np.uint8)


def codes_to_values(codes: np.ndarray) -> np.ndarray:
    values = np.asarray(codes)
    if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values > 4):
        raise ValueError("Add/Sub weight codes must be integers in [0, 4].")
    levels = np.asarray(WEIGHT_SYMBOLS, dtype=np.float32)
    return levels[values]


def pack_3bit_codes(codes: np.ndarray) -> bytes:
    """Pack eight 3-bit symbols into three bytes, low bits first.

    This is an offline storage operation, not part of the inference arithmetic.
    """
    values = np.asarray(codes)
    if values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values > 4):
        raise ValueError("Only integer codes 0..4 can be packed.")
    flat = values.reshape(-1)
    output = bytearray((len(flat) * 3 + 7) // 8)
    for index, item in enumerate(flat):
        bit_offset = index + index + index
        byte_index = bit_offset >> 3
        shift = bit_offset & 7
        code = int(item)
        output[byte_index] |= (code << shift) & 255
        if shift > 5:
            output[byte_index + 1] |= code >> (8 - shift)
    return bytes(output)


def unpack_3bit_codes(payload: bytes, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("Packed weights must be bytes-like.")
    if not isinstance(shape, (tuple, list)) or not shape:
        raise ValueError("Packed weight shape must be a non-empty tuple of dimensions.")
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, (int, np.integer))
        or dimension < 0
        for dimension in shape
    ):
        raise ValueError("Packed weight dimensions must be non-negative integers.")
    count = math.prod(int(dimension) for dimension in shape)
    expected = (count * 3 + 7) // 8
    if len(payload) != expected:
        raise ValueError(f"Packed size mismatch: expected {expected}, got {len(payload)}.")
    remainder_bits = (count * 3) % 8
    if remainder_bits and payload and (int(payload[-1]) >> remainder_bits):
        raise ValueError("Packed payload has non-zero unused padding bits.")
    output = np.empty(count, dtype=np.uint8)
    for index in range(count):
        bit_offset = index + index + index
        byte_index = bit_offset >> 3
        shift = bit_offset & 7
        value = int(payload[byte_index]) >> shift
        if shift > 5:
            value |= int(payload[byte_index + 1]) << (8 - shift)
        code = value & 7
        if code > 4:
            raise ValueError(f"Reserved Add/Sub v1 weight code {code} at index {index}.")
        output[index] = code
    return output.reshape(shape)
