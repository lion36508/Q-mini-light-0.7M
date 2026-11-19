from __future__ import annotations

import numpy as np
import torch


WEIGHT_SYMBOLS = (-1.5, -0.5, 0.0, 0.5, 1.5)
SYMBOL_COUNT = len(WEIGHT_SYMBOLS)
BITS_PER_WEIGHT = 3
FORMAT_NAME = "qmini-addsub-v1"


def quantize_weight_ste(weight: torch.Tensor) -> torch.Tensor:
    """Training-only straight-through estimator for the five-symbol codebook."""
    clipped = weight.clamp(-1.5, 1.5)
    quantized = torch.round(clipped * 2.0) / 2.0
    return weight + (quantized - weight).detach()


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
    if values.dtype.kind not in "iu" or np.any(values > 4):
        raise ValueError("Add/Sub weight codes must be integers in [0, 4].")
    levels = np.asarray(WEIGHT_SYMBOLS, dtype=np.float32)
    return levels[values]


def pack_3bit_codes(codes: np.ndarray) -> bytes:
    """Pack eight 3-bit symbols into three bytes, low bits first.

    This is an offline storage operation, not part of the inference arithmetic.
    """
    values = np.asarray(codes)
    if values.dtype.kind not in "iu" or np.any(values > 4):
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
    count = int(np.prod(shape, dtype=np.int64))
    expected = (count * 3 + 7) // 8
    if len(payload) != expected:
        raise ValueError(f"Packed size mismatch: expected {expected}, got {len(payload)}.")
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
