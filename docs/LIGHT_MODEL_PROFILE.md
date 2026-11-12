# Q-Mini Light 0.7M profile

This repository is the isolated lightweight development track. Its active model configuration is
`configs/qmini_light_0.7m.json`, with an estimated **705,742 total parameters** at a 4,096-token vocabulary.

## Model shape

| Setting | Light value |
|---|---:|
| Vocabulary target | 4,096 |
| Model width | 64 |
| Expert hidden width | 128 |
| Main causal blocks | 6 |
| Experts per block | 2 (Top-1 routing retained) |
| Routing | Top-1 |
| Context limit | 4,096 tokens |
| Causal convolution kernel | 10 |
| Dilations | 1, 2, 4, 8, 16, 32 |
| State-space path | Gated diagonal SSM enabled |
| Shared refinement | 1–4 passes |
| Matrix-weight format | Q2-v2, four-level 2-bit codebook |
| Activation path | Signed INT8 fake quantization |

The Light profile reduces the 5B-scale width, hidden size and expert count rather than deleting the mechanisms. It keeps both experts active as a Top-1 sparse-routing design (16 experts become 2), and preserves the gated diagonal state-space path, four-step shared refinement, Q2-v2 quantization and checkpoint recovery. The model keeps the same implementation family as Q-Mini: text-only causal sequence processing, no Transformer self-attention or Q/K/V attention, the gated diagonal state-space path, sparse Top-1 experts, shared refinement, Q2-v2 matrix quantization, local tokenizer/data preparation, resumable checkpoints, evaluation and the existing CI/test suite. The width, hidden size, layer count, expert count and vocabulary are smaller to reach the Light parameter budget.

## Isolation and defaults

- The source snapshot was imported from `lion36508/Q-Mini` into this separate repository.
- Colab builders, runner package paths, CLI config defaults and Drive artifact folders point to this Light repository and its own `QMini-Light-0.7M` Drive area.
- The `qmini_5b.json`, 5M and larger research configs are retained as copied reference configs. They are not selected by any Light Colab build/training entry point.
- Changes made here do not write to or update `lion36508/Q-Mini`. The original 5B project remains separate and unchanged by this Light-track work.
- A parameter count is a config/layout estimate until the model is instantiated and the generated artifact manifest confirms the count. The model is randomly initialized; a successful build is not evidence of trained assistant capability.

## Training and validation

Use the Training-Ready Builder to create a 0.7M package with its own tokenizer, dataset fingerprint, initial Q2 artifact and dormant runtime. Start actual training only from the separate runner after reviewing data licensing, GPU memory, free storage and the checkpoint/optimizer budget.

The CI suite must pass before treating a code change as stable. Learning quality still needs fixed held-out data, measured post-training improvement, independent task/conversation benchmarks and real training runs.
