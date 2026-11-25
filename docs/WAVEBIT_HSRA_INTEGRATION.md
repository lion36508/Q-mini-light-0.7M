# WaveBit-HSRA integration in Q-Mini's existing new model

WaveBit-HSRA is integrated as an opt-in training mode of the existing
zero-inclusive `qmini_addsub.AddSubLanguageModel`. It is **not** a third model
or a separate architecture/checkpoint family. The original zero-free model
track and its files are not modified.

## What is integrated

- Group-wise scales are computed from existing master weights in blocks of 32;
  no trainable scale parameters are added.
- The five existing stored weight symbols are decoded into static polarity
  and a phase-sensitivity bit. The phase bit is decoded from the same master
  parameter, not stored as a second tensor.
- Bounded EWGS-style gradients are used by the WaveBit training surrogate.
- Each existing Add/Sub block gets a parameter-free fixed-width holographic
  rotation state. Its per-token context modulates its linear projections,
  depthwise convolution, SSM projections, MoE router/experts and tied output
  projection. The input embedding also uses the group-wise quantizer; the
  output head remains tied to the same embedding parameter.
- Incremental generation carries the holographic state and convolution
  context history in the existing cache path.
- The optional `CapacitorCharge` integer helper is included for a later
  optimizer experiment, but is deliberately not enabled in the default
  optimizer. Its learning effect has not yet been established.

## Parameter/checkpoint compatibility

The WaveBit memories and group scales are not trainable model parameters and
are not entries in `state_dict`. The WaveBit path retains the same parameter
count and state-dict keys as the baseline Add/Sub model. Old checkpoints with
no `run_contract.wavebit_hsra` load using the original Add/Sub behavior.
New WaveBit runs record their settings in the training run contract; SFT and
generation reconstruct the same mode from that metadata.

Example CPU pretraining invocation (after the verified English dataset and
tokenizer have been prepared):

```bash
python -m qmini_addsub.train \
  --train-bin data/train.bin \
  --valid-bin data/validation.bin \
  --tokenizer data/tokenizer.json \
  --output runs/qmini_addsub_wavebit \
  --device cpu \
  --wavebit-hsra \
  --wavebit-group-size 32 \
  --wavebit-rotation 1 \
  --wavebit-gamma 4 \
  --wavebit-ewgs-strength 0.5
```

The mode remains opt-in while it is being scientifically compared with the
original path. A baseline run remains available by omitting `--wavebit-hsra`.

## Explicit limitations

- PyTorch training still uses floating-point autograd and dense operators.
  This integration is not a zero-float runtime and makes no 3x speed claim.
- The integer projection and rotation functions are reference primitives,
  not the complete native model. ARM64/x86 kernels and end-to-end fixed-point
  parity still need separate implementation and benchmarks.
- The current packed archive is per-tensor scaled. WaveBit uses group-wise
  scales, so the legacy exporter rejects WaveBit checkpoints rather than
  silently writing incorrect packed weights. A versioned group-wise archive
  and full integer runtime are a later gate.
- The capacitor-charge helper is not active in AdamW training; it must be
  evaluated as a separate optimizer ablation before activation.
- Validation-loss, held-out English quality, latency, energy and memory
  comparisons against baseline have not yet been run.

The existing Add/Sub model remains the single zero-inclusive evolving model.
All WaveBit additions are implemented inside that model path, not alongside it.
