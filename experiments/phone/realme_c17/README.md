# Parked phone experiment — Realme C17 track

**Status: paused and outside the active Q-Mini model.** The Q-Mini Add/Sub model and its training pipeline are for Google Colab only in the current project phase.

This folder preserves phone-specific native experiments inside the same repository without coupling them to the main model, tokenizer, training, checkpoints, or Colab notebook.

## Scope separation

- Active model development: `qmini_addsub/` and `colab/QMini_AddSub_V1_From_Scratch.ipynb` only.
- Phone-specific native prototypes and future Android/ARM work: this folder only.
- No mobile runtime, native phone package, Android app, phone export, ARM/NEON optimization, or phone performance goal is part of the active model roadmap.
- The native projection file here is a preserved experimental prototype. It is not imported by `qmini_addsub`, not compiled by the primary CI pipeline, and not a production phone runtime.

## Device identification note

The latest instruction names Realme C17. Earlier repository notes described a Realme C73 / RMX3945 target, which is inconsistent with that new label. This experiment is therefore parked under the requested `realme_c17` name, but no chipset, RAM, Android version, or performance claim is assumed until the exact device model is verified.

## Re-activation requirements

Only resume this folder when phone deployment is explicitly back in scope. Then first verify the exact device model and system version; define a separate runtime contract; implement and test the whole inference graph; and benchmark the actual handset. Do not change the Colab model's architecture or training recipe merely to satisfy this parked phone experiment.

## Preserved prototype

`native_projection.cpp` contains an experimental five-symbol projection implementation. It is retained for traceability and future comparison only. Its prior host/ARM64 CI checks do not prove Android integration or phone performance.