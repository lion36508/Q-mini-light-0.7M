# Q-Mini Light 0.7M

A from-scratch, text-only language-model research project. This repository contains a legacy 0.7M baseline and a separately specified next-generation Add/Sub model line. It is **not yet a validated general-purpose assistant or a complete Android runtime**.

## Exactly two model lines

This repository intentionally maintains only two model implementations: the preserved legacy Q-Mini Light baseline and the independent Q-Mini Add/Sub v1. Other JSON files under `configs/` are capacity, planning or smoke profiles for the legacy codebase—not additional implemented model families. Historical notebooks are archived and are not active training routes.


| Line | Location | Status |
|---|---|---|
| Q-Mini Light legacy | `qmini/` | Existing 705,742-parameter research baseline; preserve its code, Q2-v2 artifacts and checkpoints |
| Q-Mini Add/Sub v1 | `qmini_addsub/` | Independent training/data prototype; full multiply-free inference runtime remains incomplete |

Do not mix the two model lines. The Add/Sub loader must never read legacy Q2-v2 checkpoints, and the legacy package must remain unchanged by Add/Sub development.

## Q-Mini Add/Sub v1 — independent from-scratch model

The new line is implemented in qmini_addsub/. It reuses the legacy model's architectural ideas—causal depthwise convolution, gated diagonal state-space memory and sparse Top-1 experts—but uses new code, an independent 512-token byte-BPE tokenizer, separate checkpoints and its own training/data pipeline.

The QAT weight alphabet is {-1.5, -0.5, 0, +0.5, +1.5}; fixed-width packing therefore needs 3 bits per weight. The random-init pretraining model, source-preparation scripts and response-masked SFT stage are available. No legacy checkpoint is loaded, and no FP16 warm-up stage is required.

Training uses three pinned, provenance-recorded English corpora: WikiText-103 raw for language pretraining, reviewed/non-synthetic English OASST1 conversations, and human-authored Databricks Dolly instruction pairs. TinyStories, Alpaca/UltraChat and model-generated teacher data are excluded. DailyDialog is excluded from the default mix because its license has non-commercial/share-alike terms. Review dataset licenses before use/redistribution.

See qmini_addsub/README.md for build instructions and docs/ADD_SUB_V1_SPEC.md for the architecture contract.

**Important limitation:** the current additive runtime implements the integer projection primitive only. Full model inference—convolution, recurrent-state, gates, experts, nonlinear LUTs, output projection and compiled device runtime—must pass its audit before Add/Sub v1 is described or used as a deployed multiply-free model. No Add/Sub weights have been trained yet.

## Current Colab entry points

- **[Add/Sub v1 — from-scratch English model](colab/QMini_AddSub_V1_From_Scratch.ipynb)** — separate experimental model line with a new random initialization, independent data pipeline and checkpoints. Both pretraining and SFT are OFF by default; source licenses must be acknowledged. Its full multiply-free inference runtime is still incomplete.

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

## Repository layout

- `qmini/` — legacy model, configuration, tokenizer, data pipeline, trainer, generation, export and runtime.
- `qmini_addsub/` — independent Add/Sub v1 configuration, training model, data filters, tokenizer, training/SFT and additive projection prototype.
- `configs/` — legacy model profiles.
- `scripts/` — data preparation, benchmark and development utilities.
- `tests/` — unit and regression tests, including synthetic test fixtures.
- `colab/` — three active entry points for the two model lines; historical experiments are preserved under `colab/archive/legacy-experiments/`.
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
