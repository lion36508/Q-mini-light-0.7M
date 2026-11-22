from __future__ import annotations

import numpy as np


def build_contribution_table() -> np.ndarray:
    """Build the fixed contribution LUT for int8 activations and five weights.

    The output is stored at twice the activation unit so half-step weights are
    represented exactly. The table is built once; inference only indexes it.
    """
    table = np.empty((256, 5), dtype=np.int16)
    for encoded in range(256):
        value = encoded - 128
        triple = value + value + value
        table[encoded, 0] = -triple
        table[encoded, 1] = -value
        table[encoded, 2] = 0
        table[encoded, 3] = value
        table[encoded, 4] = triple
    table.setflags(write=False)
    return table


CONTRIBUTION_TABLE = build_contribution_table()


def additive_projection(
    activation_codes: np.ndarray,
    weight_codes: np.ndarray,
    bias_accumulator: np.ndarray | None = None,
) -> np.ndarray:
    """Integer projection whose numerical accumulation uses addition only.

    Inputs are signed int8 activation codes. Weight codes are 0..4 for
    {-1.5, -0.5, 0, +0.5, +1.5}. The result is expressed in doubled units,
    avoiding fractional accumulators. No multiply, divide, matmul or dot
    product operation is used in this inference primitive.
    """
    activations = np.asarray(activation_codes)
    weights = np.asarray(weight_codes)
    if activations.dtype.kind not in "iu" and activations.dtype.kind != "i":
        raise TypeError("Activation codes must be integer values.")
    if activations.ndim < 1:
        raise ValueError("Activation tensor must have a feature dimension.")
    if weights.ndim != 2:
        raise ValueError("Weight codes must have shape [out_features, in_features].")
    if activations.shape[-1] != weights.shape[-1]:
        raise ValueError("Input feature count does not match weight feature count.")
    if np.any(activations < -127) or np.any(activations > 127):
        raise ValueError("Activation codes must be in [-127, 127].")
    if weights.dtype.kind not in "iu" or np.any(weights < 0) or np.any(weights > 4):
        raise ValueError("Weight codes must be integers in [0, 4].")

    encoded = activations.astype(np.int16) + 128
    weights = weights.astype(np.uint8, copy=False)
    result_shape = activations.shape[:-1] + (weights.shape[0],)
    accumulator = np.zeros(result_shape, dtype=np.int64)

    for feature in range(activations.shape[-1]):
        contribution = CONTRIBUTION_TABLE[
            encoded[..., feature, None],
            weights[:, feature],
        ]
        accumulator = np.add(accumulator, contribution, dtype=np.int64)

    if bias_accumulator is not None:
        bias = np.asarray(bias_accumulator)
        if bias.shape != (weights.shape[0],) or bias.dtype.kind not in "iu":
            raise ValueError("Bias must be an integer vector matching out_features.")
        accumulator = np.add(accumulator, bias, dtype=np.int64)
    return accumulator


def saturating_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Add integer activation arrays and saturate to the signed int8 range."""
    a = np.asarray(left)
    b = np.asarray(right)
    if a.dtype.kind not in "iu" or b.dtype.kind not in "iu":
        raise TypeError("Residual operands must be integer arrays.")
    if a.shape != b.shape:
        raise ValueError("Residual operands must have identical shapes.")
    int64_max = np.iinfo(np.int64).max
    if (
        (a.dtype.kind == "u" and a.size and int(a.max()) > int64_max)
        or (b.dtype.kind == "u" and b.size and int(b.max()) > int64_max)
    ):
        raise ValueError("Residual operands exceed the supported int64 accumulator range.")
    total = np.add(a.astype(np.int64), b.astype(np.int64))
    return np.clip(total, -127, 127).astype(np.int8)
