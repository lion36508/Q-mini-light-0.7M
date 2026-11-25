# Q-Mini Light 0.7M profile

This file describes the legacy Light model only. The active legacy configuration is `configs/qmini_light_0.7m.json`.

| Setting | Value |
|---|---:|
| Vocabulary target | 4,096 |
| Width | 64 |
| Expert hidden width | 128 |
| Main blocks | 6 |
| Experts per block | 2 |
| Routing | Sparse Top-1 |
| Context | 512 tokens |
| Kernel | Causal depthwise convolution, width 10 |
| Dilations | 1, 2, 4, 8, 16, 32 |
| State-space path | Gated diagonal SSM enabled |
| Weight format | Q2-v2, four symbols, 2 bits/symbol |

The documented reference profile has 705,742 parameters. The exact count should be confirmed from the instantiated configuration/model before publishing a build artifact.

This profile is the legacy line in `qmini/`. It is not the Add/Sub model. Preserve its checkpoints, tokenizer and existing Drive location. The legacy inference path uses multiplication/dot products and is not Add/Sub-only.

No large capacity-only profiles are part of the Light target. This repository does not claim a trained 5M/134M/152M/853M/5B model or distributed training support.
