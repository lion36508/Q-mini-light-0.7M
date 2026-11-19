# Q-Mini Light 0.7M

A from-scratch, text-only language-model research project. This repository contains a legacy 0.7M baseline and a separately specified next-generation Add/Sub model line. It is **not yet a validated general-purpose assistant or a complete Android runtime**.

## Model lines

| Line | Location | Status |
|---|---|---|
| Q-Mini Light legacy | `qmini/` | Existing 705,742-parameter research baseline; preserve its code, Q2-v2 artifacts and checkpoints |
| Q-Mini Add/Sub v1 | `qmini_addsub/` | Separate design contract; implementation and training are not yet complete |

Do not mix the two model lines. The Add/Sub loader must never read legacy Q2-v2 checkpoints, and the legacy package must remain unchanged by Add/Sub development.

## Current Colab entry points

- **[English CPU resume trainer V12](colab/QMini_English_CPU_One_Click_Resume.ipynb)** — the current English conversation-data training pipeline. It uses pinned DailyDialog and OpenAssistant OASST1 source files, verifies data hashes/manifests, and resumes only a compatible checkpoint. Running this notebook can start/resume training in Google Drive; CI does not run model training.
- **[Existing Drive data CPU chat](colab/QMini_Existing_Drive_Data_CPU_Chat.ipynb)** — an existing-checkpoint chat experiment.

See [the Colab notebook index](colab/README.md). Other notebook variants are historical experiments retained for reproducibility; do not treat TinyStories, Hindi/Hinglish, T4 V6–V8, V9 or teacher-distillation notebooks as the current training route.

## Training-data policy

Production training inputs must have source provenance, a pinned revision, license information, exact source hashes, conversion manifests and reproducible train/validation separation.

The current conversation-data pipeline uses:

- **DailyDialog** — human dialogue data, pinned train/validation/test files. License: CC BY-NC-SA 4.0 (non-commercial terms apply).
- **OpenAssistant OASST1** — English messages only, Apache-2.0. The conversion pipeline requires ready-for-export trees, positive review, non-deleted and explicitly non-synthetic messages; assistant alternatives with a nonzero rank are excluded.

Synthetic/random examples are allowed only in clearly labeled development checks:
- `scripts/learning_proof.py`: a controlled toy learning task, not language training.
- `scripts/benchmark.py`: synthetic token IDs, not natural prompt quality or a real-device benchmark.
- `qmini/preflight.py` and `tests/`: model-graph preflight and regression fixtures, not production corpus data.

Never label model-generated synthetic replies as human-written data. The OASST1 field-level filters and tests are documented in [the project cleanup audit](docs/PROJECT_HISTORY_AND_CLEANUP_AUDIT_HI.md).

## Legacy Q-Mini Light baseline

The legacy architecture is attention-free and includes causal depthwise convolution, a gated diagonal state-space path, sparse Top-1 experts, shared refinement and Q2-v2 matrix-weight export.

Its current production Q2-v2 codebook is:

`{-1.0, -0.5, +0.5, +1.0}`

It has **four** symbols and no zero symbol for an individual entry in a nonzero row. The old model's inference path uses multiplication/dot-product, convolution and gating operations. Do not describe it as add/sub-only.

The legacy profile and artifact format remain available for backward compatibility. See [legacy architecture](docs/ARCHITECTURE.md), [Q2 quantization contract](docs/Q2_QUANTIZATION_SPEC.md) and [strict 2-bit migration status](docs/STRICT_2BIT_MIGRATION.md).

## Add/Sub v1 — separate next-generation model

The new model's required weight alphabet is:

`{-1.5, -0.5, 0, +0.5, +1.5}`

Five symbols require at least 3 bits per fixed-width weight; 2 bits cannot encode five values. The deployed Add/Sub inference path must use addition/subtraction for numerical accumulation, with versioned lookup tables for coefficient contributions and nonlinear functions. Runtime multiplication, division, matmul, dot products, convolution multiply and hidden scale multiplication are forbidden.

The model has a separate package, config, tokenizer, checkpoint, export format and Drive folder. No FP16 warm-up stage is prescribed, and no legacy checkpoints are reused. See [Add/Sub v1 architecture contract](docs/ADD_SUB_V1_SPEC.md).

**Status:** specification only. No Add/Sub model code, trained weights, performance or quality results are claimed.

## Repository layout

- `qmini/` — legacy model, configuration, tokenizer, data pipeline, trainer, generation, export and runtime.
- `qmini_addsub/` — separate Add/Sub v1 package boundary and specification.
- `configs/` — legacy model profiles.
- `scripts/` — data preparation, benchmark and development utilities.
- `tests/` — unit and regression tests, including synthetic test fixtures.
- `colab/` — active and historical Colab notebooks; see its index.
- `docs/` — architecture, training, evaluation, data provenance and project audit.

## Development checks

Install the project and run the tests:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

The CI suite also compiles the native C++ projection primitive and cross-compiles its AArch64/NEON source. Cross-compilation is not a real Android-device benchmark.

## Capability and safety notes

- A passing unit-test suite does not prove conversational quality.
- A synthetic learning proof does not prove English language understanding.
- A packed Q2 archive does not prove quality preservation.
- An AArch64 cross-compile does not prove complete Android compatibility.
- User Google Drive checkpoints and training logs are not stored in this Git repository.
- No model training is started by CI or by this README.

For the history, confirmed defects, data-quality policy and cleanup decisions, read [Project history and cleanup audit (Hindi)](docs/PROJECT_HISTORY_AND_CLEANUP_AUDIT_HI.md).
