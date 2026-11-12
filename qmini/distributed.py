"""Cloud/distributed training safety primitives.

This module intentionally does not claim to implement FSDP/ZeRO training.
It provides a fail-closed topology/resource preflight that can be used before
allocating model tensors or launching expensive cloud jobs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import socket
from typing import Any

import torch

from .config import QMiniConfig


@dataclass(frozen=True)
class DistributedPlan:
    backend: str
    world_size: int
    total_parameters: int
    fp32_adamw_state_bytes: int
    idealized_sharded_state_bytes_per_rank: int
    activation_memory_bytes: int | None
    estimated_total_bytes_per_rank: int | None
    safety_factor: float
    distributed_training_implemented: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_distributed_resources(
    config: QMiniConfig,
    *,
    world_size: int,
    backend: str = "fsdp",
    activation_memory_bytes_per_rank: int | None = None,
    safety_factor: float = 1.3,
) -> DistributedPlan:
    """Estimate lower-bound state memory; never represents a real FSDP run."""
    config.validate()
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer.")
    if backend not in {"fsdp", "zero3"}:
        raise ValueError("backend must be 'fsdp' or 'zero3'.")
    if activation_memory_bytes_per_rank is not None and (
        isinstance(activation_memory_bytes_per_rank, bool)
        or not isinstance(activation_memory_bytes_per_rank, int)
        or activation_memory_bytes_per_rank < 0
    ):
        raise ValueError("activation_memory_bytes_per_rank must be a non-negative integer.")
    if isinstance(safety_factor, bool) or not isinstance(safety_factor, (int, float)):
        raise ValueError("safety_factor must be a finite number >= 1.")
    if not 1.0 <= float(safety_factor) < 10.0:
        raise ValueError("safety_factor must be in [1, 10).")

    total = config.estimated_parameter_count()
    state_bytes = config.estimated_adamw_training_state_bytes()
    shard = (state_bytes + world_size - 1) // world_size
    activation = activation_memory_bytes_per_rank
    estimated = None
    if activation is not None:
        estimated = int((shard + activation) * float(safety_factor))
    return DistributedPlan(
        backend=backend,
        world_size=world_size,
        total_parameters=total,
        fp32_adamw_state_bytes=state_bytes,
        idealized_sharded_state_bytes_per_rank=shard,
        activation_memory_bytes=activation,
        estimated_total_bytes_per_rank=estimated,
        safety_factor=float(safety_factor),
    )


def validate_torchrun_environment(env: dict[str, str] | None = None) -> dict[str, int | str]:
    """Validate torchrun rendezvous variables without initializing a process group."""
    values = os.environ if env is None else env
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError("Missing torchrun environment variables: " + ", ".join(missing))
    try:
        rank = int(values["RANK"])
        world_size = int(values["WORLD_SIZE"])
        local_rank = int(values["LOCAL_RANK"])
        port = int(values["MASTER_PORT"])
    except (TypeError, ValueError) as exc:
        raise ValueError("RANK, WORLD_SIZE, LOCAL_RANK and MASTER_PORT must be integers.") from exc
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid torchrun rank topology.")
    if not 1 <= port <= 65535:
        raise ValueError("MASTER_PORT must be in [1, 65535].")
    host = values["MASTER_ADDR"].strip()
    if not host or any(ch.isspace() for ch in host):
        raise ValueError("MASTER_ADDR must be a non-empty hostname or IP address.")
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "master_addr": host,
        "master_port": port,
    }


def cloud_runtime_report() -> dict[str, Any]:
    """Capture hardware facts for a reproducible preflight report."""
    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({
                "index": index,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "multi_processor_count": int(props.multi_processor_count),
            })
    return {
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": len(devices),
        "devices": devices,
        "distributed_backend_available": {
            "nccl": torch.distributed.is_nccl_available(),
            "gloo": torch.distributed.is_gloo_available(),
        },
    }
