from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import QMiniConfig
from .quantization import pack_q2, unpack_quaternary


def _write_npz_array(archive: Any, key: str, value: np.ndarray) -> None:
    """Write one NPY member into an NPZ ZIP archive without buffering the whole archive."""
    with archive.open(f"{key}.npy", mode="w", force_zip64=True) as handle:
        np.lib.format.write_array(handle, np.asarray(value), allow_pickle=False)



def _q2_contract(q2_version: int) -> dict[str, Any]:
    if q2_version == 2:
        codebook = {"00": -1.0, "01": -0.5, "10": 0.5, "11": 1.0}
        return {
            "version": 2,
            "codebook": codebook,
            "row_scale": "2 * mean(abs(original_output_row))",
            "all_zero_row_scale": 0.0,
        }
    if q2_version == 3:
        codebook = {"00": -1.0, "01": 0.0, "10": 0.5, "11": 1.0}
        return {
            "version": 3,
            "codebook": codebook,
            "row_scale": "max(abs(original_output_row))",
            "all_zero_row_scale": 0.0,
            "thresholds": {
                "negative_one": "w_norm < -0.5",
                "zero": "-0.5 <= w_norm < 0.25",
                "positive_half": "0.25 <= w_norm < 0.75",
                "positive_one": "w_norm >= 0.75",
            },
            "zero_symbol": "01",
        }
    raise ValueError(f"Unsupported Q2 version: {q2_version!r}.")


def export_packed_state_stream(
    state_items: Any,
    config: QMiniConfig,
    output_path: str | Path,
) -> Path:
    """Stream named tensors into a versioned Q2 NPZ without retaining the full state dict.

    Q2 matrices use the configured codebook. With Q2-v3 and
    quantize_embeddings=False, the tied vocabulary embedding matrix is stored
    as FP16 instead. One-dimensional parameters are stored as FP16 and row
    scales as FP32. The caller may
    yield tensors from one module at a time, which avoids allocating a dense
    multi-billion-parameter model just to create its deployment artifact.
    """
    import os
    import zipfile

    config.validate()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    parameter_count = 0

    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as archive:
            for index, item in enumerate(state_items):
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    raise TypeError("state_items must yield (name, tensor) pairs.")
                name, tensor = item
                if not isinstance(name, str) or not name or name in seen:
                    raise ValueError(f"Missing or duplicate state tensor name: {name!r}")
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"State value for {name} is not a torch.Tensor.")
                if tensor.ndim < 1 or any(int(size) < 1 for size in tensor.shape):
                    raise ValueError(f"State tensor {name} has an invalid shape.")
                seen.add(name)
                shape = [int(size) for size in tensor.shape]
                parameter_count += int(tensor.numel())

                value = tensor.detach().to(
                    device="cpu", dtype=torch.float32
                ).contiguous()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite state tensor rejected: {name}")

                store_embedding_fp16 = (
                    name == "embedding.weight"
                    and config.quantization_version == "q2-v3"
                    and not config.quantize_embeddings
                )
                if value.ndim >= 2 and not store_embedding_fp16:
                    rows = int(value.shape[0])
                    flat = value.reshape(rows, -1)
                    packed, scales = pack_q2(flat, config.quantization_version)
                    packed_key, scale_key = f"p_{index}", f"s_{index}"
                    _write_npz_array(archive, packed_key, packed)
                    _write_npz_array(archive, scale_key, scales)
                    specs.append({
                        "name": name,
                        "kind": "q2",
                        "shape": shape,
                        "packed_key": packed_key,
                        "scale_key": scale_key,
                        "in_features": int(flat.shape[1]),
                    })
                else:
                    value_key = f"f_{index}"
                    _write_npz_array(
                        archive, value_key, value.numpy().astype(np.float16)
                    )
                    specs.append({
                        "name": name,
                        "kind": "fp16_matrix" if value.ndim >= 2 else "fp16",
                        "shape": shape,
                        "value_key": value_key,
                    })

            expected_parameters = config.estimated_parameter_count()
            if parameter_count != expected_parameters:
                raise ValueError(
                    f"Streamed state has {parameter_count:,} parameters; "
                    f"config expects {expected_parameters:,}."
                )

            q2_version = 3 if config.quantization_version == "q2-v3" else 2
            contract = _q2_contract(q2_version)
            codebook = contract["codebook"]
            metadata = {
                "format": f"qmini-packed-v{q2_version}",
                "q2_contract": contract,
                "config": config.to_dict(),
                "tensor_count": len(specs),
                "quaternary_tensor_count": sum(
                    item["kind"] == "q2" for item in specs
                ),
                "fp16_tensor_count": sum(
                    item["kind"] in {"fp16", "fp16_matrix"} for item in specs
                ),
                "fp16_vector_tensor_count": sum(
                    item["kind"] == "fp16" for item in specs
                ),
                "fp16_matrix_tensor_count": sum(
                    item["kind"] == "fp16_matrix" for item in specs
                ),
                "tensors": specs,
                "quaternary_codes": codebook,
            }
            _write_npz_array(
                archive,
                "metadata_json",
                np.asarray(json.dumps(metadata, separators=(",", ":"))),
            )

        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def export_packed_state(
    state_dict: dict[str, torch.Tensor],
    config: QMiniConfig,
    output_path: str | Path,
) -> Path:
    """Write a portable versioned Q2 NPZ snapshot from an in-memory state dictionary."""
    return export_packed_state_stream(state_dict.items(), config, output_path)


def _reconstruct_packed_state_impl(path: str | Path) -> tuple[QMiniConfig, dict[str, np.ndarray]]:
    """Reconstruct dequantized FP32 arrays from a packed Q-Mini archive."""
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        archive_format = metadata.get("format")
        if archive_format not in {"qmini-packed-v1", "qmini-packed-v2", "qmini-packed-v3"}:
            raise ValueError("Unsupported Q-Mini packed export format.")
        q2_version = int(archive_format.rsplit("v", 1)[1])
        if q2_version in {2, 3}:
            expected_contract = _q2_contract(q2_version)
            if (
                metadata.get("q2_contract") != expected_contract
                or metadata.get("quaternary_codes") != expected_contract["codebook"]
            ):
                raise ValueError(f"Q2-v{q2_version} archive contract/codebook metadata is missing or invalid.")
        config = QMiniConfig.from_dict(metadata["config"])
        expected_quantization = "q2-v2" if q2_version in {1, 2} else "q2-v3"
        if q2_version in {2, 3} and config.quantization_version != expected_quantization:
            raise ValueError(
                f"Q2-v{q2_version} archive config must declare "
                f"quantization_version={expected_quantization!r}."
            )
        state: dict[str, np.ndarray] = {}
        for spec in metadata["tensors"]:
            shape = tuple(int(x) for x in spec["shape"])
            if spec["kind"] == "q2":
                if spec["name"] == "embedding.weight" and not config.quantize_embeddings:
                    raise ValueError("Q2 embedding payload conflicts with quantize_embeddings=false.")
                value = unpack_quaternary(
                    archive[spec["packed_key"]],
                    archive[spec["scale_key"]],
                    int(spec["in_features"]),
                    format_version=q2_version,
                ).reshape(shape)
            elif spec["kind"] == "fp16":
                if len(shape) >= 2:
                    raise ValueError(f"FP16 vector tensor {spec['name']} has matrix shape.")
                value = np.asarray(archive[spec["value_key"]], dtype=np.float32).reshape(shape)
            elif spec["kind"] == "fp16_matrix":
                if (
                    q2_version != 3
                    or config.quantize_embeddings
                    or spec["name"] != "embedding.weight"
                    or len(shape) < 2
                ):
                    raise ValueError("FP16 matrix encoding is reserved for the Q2-v3 tied embedding.")
                value = np.asarray(archive[spec["value_key"]], dtype=np.float32).reshape(shape)
            else:
                raise ValueError(f"Unknown tensor encoding: {spec['kind']!r}")
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite tensor found in packed archive: {spec['name']}")
            state[spec["name"]] = value
        if len(state) != int(metadata["tensor_count"]):
            raise ValueError("Packed archive tensor count does not match its metadata.")
    return config, state


def reconstruct_packed_state(
    path: str | Path,
) -> tuple[QMiniConfig, dict[str, np.ndarray]]:
    """Reconstruct a packed state while normalizing malformed archive errors."""
    target = Path(path)
    try:
        return _reconstruct_packed_state_impl(target)
    except FileNotFoundError:
        raise
    except (KeyError, AttributeError, TypeError, OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"Malformed Q-Mini packed archive at {target}: missing or invalid "
            "metadata or tensor payload."
        ) from exc


def _verify_packed_export_impl(path: str | Path) -> dict[str, Any]:
    """Validate packed arrays one tensor at a time without expanding a 5B model."""
    target = Path(path)
    packed_weight_bytes = 0
    scale_bytes = 0
    vector_bytes = 0
    matrix_fp16_bytes = 0
    parameter_count = 0
    q2_count = 0
    fp16_count = 0
    fp16_matrix_count = 0

    with np.load(target, allow_pickle=False) as archive:
        if "metadata_json" not in archive:
            raise ValueError("Packed archive is missing metadata_json.")
        metadata = json.loads(str(archive["metadata_json"].item()))
        archive_format = metadata.get("format")
        if archive_format not in {"qmini-packed-v1", "qmini-packed-v2", "qmini-packed-v3"}:
            raise ValueError("Unsupported Q-Mini packed export format.")
        q2_version = int(archive_format.rsplit("v", 1)[1])
        if q2_version in {2, 3}:
            expected_contract = _q2_contract(q2_version)
            if (
                metadata.get("q2_contract") != expected_contract
                or metadata.get("quaternary_codes") != expected_contract["codebook"]
            ):
                raise ValueError(f"Q2-v{q2_version} archive contract/codebook metadata is missing or invalid.")
        config = QMiniConfig.from_dict(metadata["config"])
        expected_quantization = "q2-v2" if q2_version in {1, 2} else "q2-v3"
        if q2_version in {2, 3} and config.quantization_version != expected_quantization:
            raise ValueError(
                f"Q2-v{q2_version} archive config must declare "
                f"quantization_version={expected_quantization!r}."
            )
        specs = metadata.get("tensors")
        if not isinstance(specs, list) or len(specs) != int(metadata.get("tensor_count", -1)):
            raise ValueError("Packed archive tensor manifest count is invalid.")

        seen: set[str] = set()
        for spec in specs:
            name = str(spec.get("name", ""))
            if not name or name in seen:
                raise ValueError(f"Missing or duplicate tensor name in packed archive: {name!r}")
            seen.add(name)
            shape = tuple(int(value) for value in spec.get("shape", []))
            if not shape or any(value < 1 for value in shape):
                raise ValueError(f"Invalid shape in packed archive for {name}.")
            parameter_count += math.prod(shape)

            if spec.get("kind") == "q2":
                if name == "embedding.weight" and not config.quantize_embeddings:
                    raise ValueError("Q2 embedding payload conflicts with quantize_embeddings=false.")
                if len(shape) < 2:
                    raise ValueError(f"Q2 tensor {name} has rank less than two.")
                in_features = math.prod(shape[1:])
                if int(spec.get("in_features", -1)) != in_features:
                    raise ValueError(f"Q2 input dimension mismatch for {name}.")
                packed = np.asarray(archive[spec["packed_key"]])
                scales = np.asarray(archive[spec["scale_key"]])
                if packed.dtype != np.uint8 or packed.shape != (
                    shape[0], (in_features + 3) // 4
                ):
                    raise ValueError(f"Invalid packed Q2 payload for {name}.")
                if scales.dtype != np.float32 or scales.shape != (shape[0],):
                    raise ValueError(f"Invalid Q2 row scales for {name}.")
                if not np.isfinite(scales).all() or (
                    np.any(scales <= 0) if q2_version == 1 else np.any(scales < 0)
                ):
                    raise ValueError(f"Invalid or non-finite Q2 scales for {name}.")
                packed_weight_bytes += int(packed.nbytes)
                scale_bytes += int(scales.nbytes)
                q2_count += 1
            elif spec.get("kind") == "fp16":
                value = np.asarray(archive[spec["value_key"]])
                if len(shape) >= 2 or value.dtype != np.float16 or value.shape != shape:
                    raise ValueError(f"Invalid FP16 vector payload for {name}.")
                if not np.isfinite(value).all():
                    raise ValueError(f"Non-finite FP16 parameter found in {name}.")
                vector_bytes += int(value.nbytes)
                fp16_count += 1
            elif spec.get("kind") == "fp16_matrix":
                value = np.asarray(archive[spec["value_key"]])
                if (
                    q2_version != 3
                    or config.quantize_embeddings
                    or name != "embedding.weight"
                    or len(shape) < 2
                    or value.dtype != np.float16
                    or value.shape != shape
                ):
                    raise ValueError(f"Invalid FP16 matrix payload for {name}.")
                if not np.isfinite(value).all():
                    raise ValueError(f"Non-finite FP16 matrix found in {name}.")
                matrix_fp16_bytes += int(value.nbytes)
                fp16_count += 1
                fp16_matrix_count += 1
            else:
                raise ValueError(f"Unknown tensor encoding in packed archive: {name}.")

    expected_fp16_matrices = (
        1 if q2_version == 3 and not config.quantize_embeddings else 0
    )
    if fp16_matrix_count != expected_fp16_matrices:
        raise ValueError(
            "Packed archive FP16 matrix count does not match the configured embedding policy."
        )
    if parameter_count != config.estimated_parameter_count():
        raise ValueError(
            f"Packed archive has {parameter_count:,} parameters; config expects "
            f"{config.estimated_parameter_count():,}."
        )
    return {
        "format": archive_format,
        "q2_version": q2_version,
        "q2_contract": metadata.get("q2_contract", _q2_contract(q2_version)) if q2_version in {2, 3} else {"version": 1, "codebook": {"00": -2.0, "01": -1.0, "10": 1.0, "11": 2.0}},
        "tensor_count": len(seen),
        "parameter_count": parameter_count,
        "q2_tensor_count": q2_count,
        "fp16_tensor_count": fp16_count,
        "fp16_matrix_tensor_count": fp16_matrix_count,
        "packed_weight_bytes": packed_weight_bytes,
        "row_scale_bytes": scale_bytes,
        "fp16_vector_bytes": vector_bytes,
        "fp16_matrix_bytes": matrix_fp16_bytes,
        "decoded_payload_bytes": packed_weight_bytes + scale_bytes + vector_bytes + matrix_fp16_bytes,
        "artifact_bytes": target.stat().st_size,
        "estimated_raw_layout_bytes": config.estimated_packed_export_bytes(),
        "context": config.max_context,
    }



def verify_packed_export(path: str | Path) -> dict[str, Any]:
    """Verify a packed archive and normalize malformed-container errors."""
    target = Path(path)
    try:
        return _verify_packed_export_impl(target)
    except FileNotFoundError:
        raise
    except (KeyError, AttributeError, TypeError, OSError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"Malformed Q-Mini packed archive at {target}: missing or invalid "
            "metadata or tensor payload."
        ) from exc
