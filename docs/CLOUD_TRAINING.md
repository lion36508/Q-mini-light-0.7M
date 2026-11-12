# Cloud training readiness

## Current status

The reference trainer in `qmini/train.py` is single-process/single-device. This
repository does **not** yet provide an FSDP, DeepSpeed ZeRO-3, expert-parallel,
or tensor-parallel training implementation. Do not launch the 5B profile with
the existing `qmini train` command and do not interpret a sharding estimate
as proof of multi-GPU support.

## First implementation slice: fail-closed preflight

`qmini.distributed` adds:
- a deterministic FP32 parameter + gradient + AdamW-moment lower-bound estimate;
- an explicitly idealized per-rank sharding estimate for FSDP/ZeRO-3 planning;
- optional activation-memory budget and safety factor;
- strict torchrun environment validation (without silently initializing a process group);
- a cloud hardware/software report.

Example:

```python
from qmini.config import QMiniConfig
from qmini.distributed import plan_distributed_resources, cloud_runtime_report

config = QMiniConfig.from_json("configs/qmini_5b.json")
plan = plan_distributed_resources(
    config,
    world_size=8,
    backend="fsdp",
    activation_memory_bytes_per_rank=24 * 1024**3,  # measured, not guessed
)
print(plan.to_dict())
print(cloud_runtime_report())
```

The estimate divides the FP32 AdamW state lower bound by world size. It excludes
FSDP all-gather peaks, activations unless supplied, communication buckets,
fragmentation, data-loader memory, CPU offload, optimizer implementation
details, checkpoint staging and expert imbalance. The safety factor is a
planning reserve, not a measured guarantee.

## Required cloud training sequence

1. Capture the runtime report and exact GPU model/count/memory.
2. Run the 5M profile on one GPU; prove loss decreases on fixed held-out data.
3. Run a two-process tiny equivalence test against the single-device reference.
4. Implement and test FSDP full-shard or ZeRO-3 as a separate trainer backend.
5. Save sharded model + optimizer + scheduler + scaler + per-rank RNG + sampler
   state and verify interrupted resume.
6. Test global token-weighted loss and gradient clipping across ranks.
7. Profile peak memory including all-gather/reshard and optimizer steps.
8. Scale through 25M/152M before any 5B attempt.

## 5B memory boundary

The 5,132,630,288-parameter profile needs approximately 82.12 GB decimal
(76.48 GiB) for FP32 parameters, gradients and two FP32 Adam moments alone.
That excludes activations and all runtime overhead. Mixed precision, sharding,
activation checkpointing and an adequate multi-GPU fabric are prerequisites;
they are not implemented by this preflight module.

## From-scratch guarantee

Every rank must instantiate Q-Mini from its config with random initialization
and synchronized RNG semantics. No external pretrained weights, downloaded
model checkpoints, or implicit `--init-from` paths are permitted in the
pretraining entry point. SFT, if later enabled, may load only a Q-Mini-produced
pretraining checkpoint with a verified manifest and a fresh optimizer.
