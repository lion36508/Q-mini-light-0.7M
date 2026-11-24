# Active Colab entry points — Add/Sub is the primary model

There are exactly two model lines in this repository. The new Add/Sub model is the primary development/training target. The old Q-Mini Light Q2-v2 notebooks are retained only for compatibility, rollback and comparison. Never mix their datasets, tokenizers, checkpoints, configs or Drive folders.

## 1. Primary model: Q-Mini Add/Sub

- **Training:** [Q-Mini Add/Sub English training notebook](QMini_AddSub_V1_From_Scratch.ipynb).
- Package: qmini_addsub/.
- Independent random initialization, English-first data preparation, tokenizer, pretraining, response-masked SFT and held-out evaluation.
- CPU-only by default. All data download/import, tokenizer construction, pretraining, SFT, evaluation and export switches are OFF until explicitly reviewed and enabled.
- Uses a new isolated Drive folder at Q mini 2 light/trainability_finewiki_v1. Do not overwrite the older trainability_v2 data/checkpoints.
- No old Q2-v2 checkpoint is loaded.
- The training model is experimental and the full integer-only Android runtime is not complete.

## 2. Legacy model: Q-Mini Light Q2-v2 (compatibility only)

- **Existing-checkpoint chat:** [Existing Drive data CPU chat](QMini_Existing_Drive_Data_CPU_Chat.ipynb).
- **Legacy resume/debug only:** [English CPU resume trainer — V12](QMini_English_CPU_One_Click_Resume.ipynb).
- Package: qmini/.
- Preserve its checkpoints/artifacts. Do not use it as the default route for new Add/Sub development.

## Replacement rule

The new Add/Sub model becomes the sole default only after real-data English pretraining and SFT, independent held-out quality checks, a complete audited integer-only graph, ARM64 scalar/NEON parity, and end-to-end measurements on the target Realme C73 5G. Until that is proven, legacy remains available for rollback but is not the primary development model.

See docs/PHONE_RUNTIME_READINESS.md, docs/EXPERIMENTAL_TECHNOLOGY_ROADMAP_2026.md and qmini_addsub/README.md for status and requirements.

## Legacy regression fixtures

Historical notebook snapshots used by CI are stored under tests/fixtures/legacy_colab/. They protect backward compatibility only; they are not current training instructions.
