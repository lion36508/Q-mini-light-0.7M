# Colab notebook index

## Current entry points

1. **[English CPU resume trainer — V12](QMini_English_CPU_One_Click_Resume.ipynb)**  
   Current English conversation-data pipeline. It uses pinned DailyDialog and OASST1 sources, verifies their manifests and resumes only a verified compatible checkpoint. Running the notebook starts/resumes training in the user's Google Drive; CI never launches this training job.
2. **[Existing Drive data CPU chat](QMini_Existing_Drive_Data_CPU_Chat.ipynb)**  
   Existing-checkpoint chat/runtime experiment. It does not create a new training run.

## Historical / experimental notebooks

Other notebooks in this directory are retained for reproducibility and old checkpoint compatibility. They include older TinyStories, Hindi/Hinglish, T4 V6–V8, V9, SFT and teacher-distillation experiments. They are not the current recommended training path and must not be mixed with the V12 corpus/checkpoint lineage.

Do not delete a historical notebook until its Drive artifacts, README links, tests and checkpoint provenance have been mapped. A later archive-only change can move old notebooks without changing their contents or deleting old weights.

## Data rule

Only the datasets named in the active notebook and their verified manifests are training inputs. Synthetic token IDs in CI, preflight and benchmarks are development fixtures only. Teacher-generated or synthetic examples must never be relabeled as human-written data.
