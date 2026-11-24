# Q-Mini Light 0.7M — Add/Sub is the primary development model

A from-scratch, English-first, text-only language-model research project. The new Q-Mini Add/Sub line is the primary model being developed for future assistant and phone use. The old Q-Mini Light/Q2-v2 line is retained only as a compatibility and comparison baseline until Add/Sub passes the full replacement gate.

**Important:** Add/Sub is not yet a validated general-purpose assistant or a complete Android runtime. No full real-data Add/Sub training run or Realme C73 on-device benchmark has been completed.

## Model lineup — exactly two lines

| Line | Package | Role and status |
|---|---|---|
| **Q-Mini Add/Sub (primary)** | qmini_addsub/ | New 760,340-parameter independent model, English training pipeline, SFT, evaluation, packed weight archive and reference chat. Primary development target; full integer-only phone runtime and trained-quality validation remain incomplete. |
| Q-Mini Light legacy Q2-v2 | qmini/ | Preserved 705,742-parameter baseline and old artifacts. Compatibility/reference only; not the default route for new model development. |

The legacy package and checkpoints are preserved for rollback and comparison. Do not load Q2-v2 checkpoints into Add/Sub, overwrite old Drive folders, or delete the legacy artifacts before the new model passes its replacement gate. This is a change in which model is the primary target, not permission to destroy the only known rollback copy.

## New Q-Mini Add/Sub model

The Add/Sub reference architecture has:
- Causal depthwise convolution for local token mixing.
- Gated diagonal state-space recurrence for bounded recurrent context.
- Sparse Top-1 SwiGLU experts.
- RMSNorm and learned residual branch gates initialized conservatively.
- Router load-balancing plus a separately weighted router z-loss.
- Independent 1,024-token byte-level BPE tokenizer and 512-token context.
- Five-symbol weight alphabet {-1.5, -0.5, 0, +0.5, +1.5}. Five values require 3-bit fixed-width packing; this is not a 2-bit or 1.58-bit model.

The architecture has **760,340 parameters** in its current default profile. The current training implementation uses floating-point PyTorch operations and STE/QAT; that is allowed for training. The complete deployed inference graph must ultimately use the specified integer lookup/comparison/add/subtract path.

## English knowledge and assistant-data plan

The pinned Add/Sub data-preparation path uses three complementary sources:
1. **FineWiki English Wikipedia sample** for factual next-token pretraining. Pinned source revision e0bbfbb, file en.jsonl.zst, SHA-256 checked before parsing, approximately a 1% sample from an August 2025 Wikimedia Enterprise snapshot. License: CC-BY-SA-4.0.
2. **OpenAssistant OASST1** for positively reviewed, explicitly non-synthetic English conversations, using strict ready-for-export, review, deletion and assistant-rank filters. License: Apache-2.0.
3. **Databricks Dolly 15k** for human-authored English instruction/response pairs. License: CC-BY-SA-3.0.

The pipeline deduplicates and splits whole source documents/conversation groups before tokenization/training, records hashes and manifests, and stages output transactionally so a failed preparation cannot publish a partial final dataset. Review attribution/share-alike obligations before redistribution. The FineWiki sample is not the entire English Wikipedia dump or a complete knowledge base.

No model-generated conversations are labelled as human-authored training data. Synthetic examples are reserved for clearly marked engineering tests. All user-facing model training should remain English-first unless a separately versioned multilingual experiment is approved.

## Current training route

Use **Q-Mini Add/Sub — FineWiki English assistant training**:
- Notebook: colab/QMini_AddSub_V1_From_Scratch.ipynb
- Package: qmini_addsub/
- New isolated Drive path: Q mini 2 light/trainability_finewiki_v1
- CPU-only by default; conservative thread count.
- Data preparation, tokenizer build, pretraining, SFT, held-out evaluation and export are OFF by default. Enable each only after reviewing source licenses, storage and run identity.
- The current notebook suggests a configurable 15,000-step pretraining schedule and 2,000-step SFT starting schedule. These are proposals, not verified optimal settings or a guarantee of runtime or answer quality.

The legacy English CPU resume and existing-Drive chat notebooks remain available only for legacy checkpoint compatibility/testing:
- colab/QMini_English_CPU_One_Click_Resume.ipynb
- colab/QMini_Existing_Drive_Data_CPU_Chat.ipynb

Do not continue new feature development on the legacy Q2-v2 route. Do not mix model packages, datasets, tokenizers, checkpoints, configs or Drive folders.

## Phone deployment target

The initial target is the user's Realme C73 5G (reported model identifier RMX3945). The published specification lists a MediaTek Dimensity 6300 and 4 GB physical RAM. Dynamic RAM expansion is not physical RAM and cannot be counted as guaranteed process memory.

The current .qma archive is a weight container with float32 auxiliary tensors, not a deployed engine. The current integer prototype covers additive projection and bounded residual addition, including a new native C++ scalar/ARM-NEON projection kernel. It is not a complete phone inference graph. Integer convolution, SSM, gating, experts, normalization, activation LUTs, output projection, Android packaging and full ARM64 runtime are still incomplete.

See docs/PHONE_RUNTIME_READINESS.md for the device profile, proposed measurement protocol and replacement gates. The provisional targets in that file are goals, not measured speed or memory results.

## Replacement gate — when Add/Sub can replace the old default

The Add/Sub model becomes the sole user-facing default only after:
1. Real-data English pretraining and assistant SFT complete with immutable dataset, tokenizer, config and checkpoint hashes.
2. Independent held-out tests pass for response quality, multi-turn continuity, task performance, factual consistency and repetition/collapse.
3. The complete integer-only graph passes an operator audit and reference-parity tests; no float32 auxiliary inference path remains.
4. ARM64 scalar and NEON implementations pass correctness tests and the complete model is measured on the actual Realme C73.
5. Load, chat, export/resume and rollback are tested without deleting the legacy artifacts.

Until then, Add/Sub is the **primary development target**, but it must not be described as a finished assistant or phone-ready release.

## Evaluation and engineering checks

Install and run the local suite:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

CI also performs CPU-only tests, native compilation, ARM64 cross-compilation, training/resume fixtures, quantization/export checks, and tiny smoke evaluations. A passing test suite verifies software properties; it does not establish natural-language capability. The tiny legacy smoke fixture has previously returned identical placeholder answers for every prompt, so it must not be presented as evidence of assistant quality.

Track at minimum:
- Held-out English language loss and response loss.
- Task accuracy with separate math, logic, factual QA and instruction-following slices.
- Multi-turn continuity and context/state tracking.
- Blank, repeated, identical, truncated and malformed-answer rates.
- Expert load balance and routing starvation.
- Tokens/second, peak RAM, load time, packed size and sustained thermal performance.
- Quantized-vs-reference error and actual ARM64 device behavior.

## Repository layout

- qmini_addsub/ — primary Add/Sub model, independent training, SFT, generation, export and integer-runtime prototype.
- qmini/ — preserved legacy Q2-v2 model and compatibility tools.
- configs/ — legacy model profiles and development fixtures.
- scripts/ — pinned dataset preparation, tokenizer/stream building, benchmarks and development utilities.
- tests/ — unit, regression and synthetic engineering fixtures.
- colab/ — Add/Sub primary training notebook plus legacy compatibility notebooks.
- docs/ — architecture, provenance, evaluation, cleanup audit, experimental roadmap and phone-readiness contract.

For detailed architecture and data lineage, see qmini_addsub/README.md, docs/ADD_SUB_V1_SPEC.md, docs/EVALUATION.md and docs/EXPERIMENTAL_TECHNOLOGY_ROADMAP_2026.md.
