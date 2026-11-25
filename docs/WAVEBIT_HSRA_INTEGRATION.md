# WaveBit-HSRA integration in Q-Mini's existing new model

WaveBit-HSRA is integrated as an opt-in training mode of the existing
zero-inclusive `qmini_addsub.AddSubLanguageModel`. It is **not** a third model
or a separate architecture/checkpoint family. The original zero-free model
track and its files are not modified.

## What is integrated

- Group-wise scales are computed in aligned blocks of 30 weights; no trainable
  scale parameters are added.
- Octave-Shift uses base levels `{-1, -0.5, 0, +0.5, +1}`. Each triplet shares
  one mode bit that boosts all three weights by four.
- The packed payload costs 8 bits per three weights (2.667 bits/weight),
  before scale metadata, headers, row offsets or alignment.
- Codes 250..254 skip 4/8/16/32/64 all-zero triplet-groups; 255 is followed by
  an extended skip count. The skip units are groups of three weights.
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
  --wavebit-group-size 30 \
  --wavebit-rotation 1 \
  --wavebit-gamma 4 \
  --wavebit-ewgs-strength 0.5 \
  --wavebit-octave-temperature 0.25
```

The mode remains opt-in while it is being scientifically compared with the
original path. A baseline run remains available by omitting `--wavebit-hsra`.

## Explicit limitations

- PyTorch training still uses floating-point autograd and dense operators.
  This integration is not a zero-float runtime and makes no 3x speed claim.
- The 256-entry Octave LUT decodes a byte into three signed half-unit
  coefficients. Both the Python integer reference and the
  `qmini_octave_linear_i8_v1` C++ ABI use lookup, add, subtract and shifts,
  returning half-unit accumulators before separately stored scales are applied.
- The C++ ABI includes an 8-lane ARM NEON shift/add path and a scalar fallback.
  Its output is not yet wired into the complete Add/Sub generation runtime;
  native-versus-Python projection parity is tested separately.
- This is a small L1-friendly software lookup table, not a single ARM NEON TBL
  instruction; the NEON TBL table operand is limited to four 16-byte vectors.
- End-to-end fixed-point scale application, normalization, gates, SSM and
  output requantization still require integration and target-device benchmarks.
- The current packed archive is per-tensor scaled. WaveBit uses group-wise
  scales, so the legacy exporter rejects WaveBit checkpoints rather than
  silently writing incorrect packed weights. A versioned group-wise archive
  and full integer runtime are a later gate.
- The capacitor-charge helper is not active in AdamW training; it must be
  evaluated as a separate optimizer ablation before activation.
- One shared mode bit cannot independently boost one weight within a triplet;
  independent nine-level choices for three weights require at least 10 bits.
- Skip savings occur only for clustered all-zero triplet-groups and are not
  guaranteed for dense/non-zero weight distributions.
- Validation-loss, held-out English quality, latency, energy and memory
  comparisons against ternary and previous Add/Sub baselines have not yet run.

The existing Add/Sub model remains the single zero-inclusive evolving model.
All WaveBit additions are implemented inside that model path, not alongside it.



## x2 + Hadamard research ablation (same model, opt-in)

The new controls extend the existing model path; they do not create another
model class or parameter set:

- octave_boost_factor=2 selects a shared x2 boost for each packed weight
  triplet. The default remains x4 so old WaveBit run contracts and old packed
  semantics are unchanged. The factor is recorded in the checkpoint run
  contract and must be supplied to the Python integer decoder.
- hadamard=True applies a fixed, orthonormal blockwise Walsh-Hadamard
  transform to activations entering each expert's down projection, followed
  by per-token symmetric fake quantization at the configured bit width
  (default 4). The transform uses power-of-two blocks and transforms any
  power-of-two tail, so the 170-wide default expert hidden vector remains
  exactly 170 features.
- The Hadamard transform has no trainable parameters. Existing model
  parameter count and state_dict keys remain unchanged.
- The setting is opt-in and affects the training/reference graph only. It
  does not establish lower validation loss, better English chat quality,
  lower latency or lower energy until matched training runs and held-out
  evaluation are completed.

Example CPU command-line options:

    python -m qmini_addsub.train \\
      --train-bin data/train.bin \\
      --valid-bin data/validation.bin \\
      --tokenizer data/tokenizer.json \\
      --output runs/qmini_addsub_x2_hadamard \\
      --device cpu \\
      --wavebit-hsra \\
      --wavebit-octave-boost-factor 2 \\
      --wavebit-hadamard \\
      --wavebit-hadamard-block-size 8 \\
      --wavebit-activation-bits 4

The C++ qmini_octave_linear_i8_v1 ABI remains x4-compatible. A new
qmini_octave_linear_i8_v2 ABI accepts an explicit shared x2 or x4 boost factor;
the Python wrapper keeps routing x4 calls through v1 when available and uses
v2 for x2. Both native paths use the same LUT + add/sub + shift accumulator,
and native x2 is compared against the Python integer reference in CI.

The Hadamard + activation fake-quant path remains in the PyTorch training
surrogate. It is not yet integrated into the integer inference runtime, and a
versioned group-wise packed archive is still required before complete WaveBit
checkpoints can be exported. The existing exporter continues to reject
WaveBit checkpoints.
