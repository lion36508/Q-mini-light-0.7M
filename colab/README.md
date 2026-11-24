# Colab training entry points — Add/Sub is primary

**Current scope: Google Colab only.** The new Q-Mini Add/Sub model is developed, trained, resumed and evaluated through its Colab notebook. Phone/Android deployment is explicitly out of scope in this phase.

## 1. Primary training notebook — Q-Mini Add/Sub

- Notebook: [Q-Mini Add/Sub Colab-only English training](QMini_AddSub_V1_From_Scratch.ipynb)
- Package: `qmini_addsub/`
- Independent random initialization, pinned English corpus preparation, tokenizer build, pretraining, response-masked assistant SFT and held-out evaluation.
- CPU-only by default, conservative thread count, separate Drive folder `Q mini 2 light/trainability_finewiki_v1`.
- SFT uses the experimental self-paced difficulty sampler with 35% uniform replay; metrics and checkpoint state preserve the sampler's state for exact resume.
- All data preparation and training switches are OFF by default. Review source licenses and available storage before enabling each stage.
- No legacy Q2-v2 checkpoint is loaded.

## 2. Legacy Q-Mini Light Q2-v2 — compatibility only

- Existing checkpoint chat: [Existing Drive data CPU chat](QMini_Existing_Drive_Data_CPU_Chat.ipynb).
- Legacy resume/debug only: [English CPU resume trainer — V12](QMini_English_CPU_One_Click_Resume.ipynb).
- Package: `qmini/`.
- Preserve the old checkpoints, data and tokenizer. Do not use this route for new Add/Sub development.

## 3. Phone experiments are parked separately

Phone-specific prototypes are kept under `experiments/phone/realme_c17/`. They are not imported by the active model, not part of the training notebook, and not part of the Add/Sub CI path. The current phase does not build or benchmark Android/mobile inference.

See `docs/EXPERIMENTAL_TECHNOLOGY_ROADMAP_2026.md` for the Colab research plan and `qmini_addsub/README.md` for the SFT learning recipe.
