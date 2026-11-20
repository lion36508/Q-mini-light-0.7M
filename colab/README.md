# Active Colab entry points — exactly two model lines

Only the following two model families are active in this repository. Do not mix their datasets, tokenizers, checkpoints, configs or Drive folders.

## 1. Legacy Q-Mini Light 0.7M

- **Training:** [English CPU resume trainer — V12](QMini_English_CPU_One_Click_Resume.ipynb). Uses the pinned DailyDialog/OpenAssistant preparation path and resumes only compatible legacy checkpoints.
- **Chat/demo:** [Existing Drive data CPU chat](QMini_Existing_Drive_Data_CPU_Chat.ipynb). This is a runtime demonstration for an existing legacy checkpoint; it does not create a new model family.

Package: `qmini/`. Legacy artifacts and checkpoints stay in their existing format and location. This model uses its documented Q2 format and is not an Add/Sub-only runtime.

## 2. New Q-Mini Add/Sub v1

- **From-scratch training:** [Q-Mini Add/Sub v1 notebook](QMini_AddSub_V1_From_Scratch.ipynb). English-first, independent random initialization, separate tokenizer/data/checkpoint paths, CPU by default. Pretraining and SFT are OFF until explicitly enabled.

Package: `qmini_addsub/`. It must never load legacy `qmini/` weights or artifacts. The current implementation has the training/data pipeline and an additive projection reference primitive; the complete full-graph multiply-free inference runtime and export are still under development. It is not yet a trained or deployable assistant.

## Archive policy

Older notebook variants were moved, without changing their contents, to [the legacy experiment archive](archive/README.md). They include historical Hindi/TinyStories runs, repeated English V-series notebooks, initial build/runner notebooks and teacher-distillation experiments. They are retained for historical links and checkpoint provenance only; do not start new training from them.

## Data hygiene

- The old model keeps its separate document cleaner, deduplication and source-manifest path.
- Add/Sub v1 uses pinned sources, strict OASST1 review/deletion/synthetic/rank filters, Dolly record hygiene checks, exact-content deduplication and whole-conversation group splitting.
- The Add/Sub text cleaner normalizes Unicode and whitespace, rejects email addresses, credential-like secrets and clearly labelled phone-contact strings, and records rejection reasons. Source metadata—not a text heuristic—is the English-language authority.
- Synthetic/random fixtures under `tests/`, `scripts/learning_proof.py` and benchmark utilities are development-only. They are never production training data.
