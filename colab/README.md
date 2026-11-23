# Active Colab entry points — exactly two model lines

Only the following two model families are active in this repository. Do not mix their datasets, tokenizers, checkpoints, configs or Drive folders.

## 1. Legacy Q-Mini Light 0.7M

- **Training:** [English CPU resume trainer — V12](QMini_English_CPU_One_Click_Resume.ipynb). Uses the pinned DailyDialog/OpenAssistant preparation path and resumes only compatible legacy checkpoints.
- **Chat/demo:** [Existing Drive data CPU chat](QMini_Existing_Drive_Data_CPU_Chat.ipynb). This is a runtime demonstration for an existing legacy checkpoint; it does not create a new model family.

Package: `qmini/`. Legacy artifacts and checkpoints stay in their existing format and location. This model uses its documented Q2 format and is not an Add/Sub-only runtime.

## 2. New Q-Mini Add/Sub v1

- **From-scratch training:** [Q-Mini Add/Sub v1 notebook](QMini_AddSub_V1_From_Scratch.ipynb). English-first, independent random initialization, separate tokenizer/data/checkpoint paths, CPU by default. Pretraining and SFT are OFF until explicitly enabled.

Package: `qmini_addsub/`. It must never load legacy `qmini/` weights or artifacts. The current implementation has the training/data pipeline and an additive projection reference primitive; the complete full-graph multiply-free inference runtime and export are still under development. It is not yet a trained or deployable assistant.

## Legacy regression fixtures

Historical notebook snapshots used by CI are stored under
`tests/fixtures/legacy_colab/`, outside the active Colab folder. They are kept
only to protect legacy compatibility tests. Do not use them as current training
instructions.

Only the three notebooks linked above are supported entry points. The legacy
and Add/Sub model packages, tokenizers, datasets and checkpoints must remain
separate.
