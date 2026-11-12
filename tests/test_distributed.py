import pytest

from qmini.config import QMiniConfig
from qmini.distributed import (
    cloud_runtime_report,
    plan_distributed_resources,
    validate_torchrun_environment,
)


def _config():
    return QMiniConfig(
        vocab_size=128, d_model=32, d_hidden=48, n_layers=2,
        n_experts=2, kernel_size=5, dilations=(1, 2),
        max_context=64, max_thinking_steps=2, thinking_steps=1,
    )


def test_distributed_plan_is_explicitly_idealized_and_not_implemented():
    config = _config()
    one = plan_distributed_resources(config, world_size=1)
    two = plan_distributed_resources(
        config, world_size=2, activation_memory_bytes_per_rank=1024
    )
    assert one.fp32_adamw_state_bytes == config.estimated_parameter_count() * 16
    assert two.idealized_sharded_state_bytes_per_rank == (
        one.fp32_adamw_state_bytes + 1
    ) // 2
    assert two.estimated_total_bytes_per_rank is not None
    assert two.distributed_training_implemented is False


@pytest.mark.parametrize("kwargs", [
    {"world_size": 0},
    {"world_size": -1},
    {"world_size": True},
    {"world_size": 2, "backend": "ddp"},
    {"world_size": 2, "safety_factor": 0.9},
    {"world_size": 2, "activation_memory_bytes_per_rank": -1},
])
def test_distributed_plan_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        plan_distributed_resources(_config(), **kwargs)


def test_torchrun_environment_validates_topology():
    report = validate_torchrun_environment({
        "RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "1",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500",
    })
    assert report["rank"] == 1
    assert report["world_size"] == 4


@pytest.mark.parametrize("env", [
    {},
    {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
     "MASTER_ADDR": "localhost", "MASTER_PORT": "70000"},
    {"RANK": "2", "WORLD_SIZE": "2", "LOCAL_RANK": "0",
     "MASTER_ADDR": "localhost", "MASTER_PORT": "29500"},
    {"RANK": "x", "WORLD_SIZE": "2", "LOCAL_RANK": "0",
     "MASTER_ADDR": "localhost", "MASTER_PORT": "29500"},
])
def test_torchrun_environment_rejects_invalid_topology(env):
    with pytest.raises(ValueError):
        validate_torchrun_environment(env)


def test_cloud_runtime_report_has_stable_schema():
    report = cloud_runtime_report()
    assert "torch_version" in report
    assert "cuda_available" in report
    assert "devices" in report
    assert isinstance(report["devices"], list)
