# Q-Mini — Colab-only Add/Sub learning model

Q-Mini is an English-first, from-scratch language-model research project. **The active new-model work is designed and trained through Google Colab.** The goal of this phase is to improve learning efficiency, knowledge retention, assistant behavior and reasoning on the training/evaluation pipeline—not to deploy the model on a phone.

## Exactly two model lines

| Model | Package | Current role |
|---|---|---|
| **Q-Mini Add/Sub (primary)** | `qmini_addsub/` | The new independent model, trained from random initialization through the Colab notebook. Its current default reference profile has 760,340 parameters. |
| Q-Mini Light Q2-v2 (legacy) | `qmini/` | Compatibility and comparison baseline only. Preserve old checkpoints and artifacts; do not use it as the active new-model training route. |

Do not mix model weights, tokenizers, datasets, checkpoints or Drive folders. Do not delete the legacy model until the new Add/Sub model passes its quality and reproducibility gates.

## Active training entry point — Google Colab

Use only `colab/QMini_AddSub_V1_From_Scratch.ipynb` for active Add/Sub training.

- CPU-only by default, with a conservative thread cap. No GPU quota is required.
- Uses a separate Drive directory: `Q mini 2 light/trainability_finewiki_v1`.
- Dataset preparation, tokenizer build, pretraining, SFT and evaluation are OFF by default; each stage is explicitly enabled after checking the data and outputs.
- The notebook downloads/uses pinned English data only after source-license acceptance.
- Existing Colab checkpoints are resumed only when their model, data, tokenizer, schedule and training contracts match.

The legacy notebooks remain only to inspect or test old checkpoints. They are not the active Add/Sub training path.

## New model architecture

The current Add/Sub research profile has 760,340 parameters and includes:
- Causal depthwise convolution for local token mixing.
- Gated diagonal state-space recurrence for recurrent context.
- Sparse Top-1 SwiGLU experts with load-balancing and router z-loss.
- RMSNorm and learned residual branch gates initialized conservatively.
- An independent 1,024-token English-first byte-level BPE tokenizer and 512-token context.
- Five-symbol QAT weight alphabet `{-1.5, -0.5, 0, +0.5, +1.5}`. Five symbols require three fixed-width bits; this is neither 2-bit nor ternary 1.58-bit quantization.

Training uses floating-point PyTorch operations and quantization-aware training. The model's real assistant capability is not established until it has completed real-data pretraining, assistant fine-tuning and independent held-out evaluation.

## Improved learning recipe

The Add/Sub SFT stage now uses an experimental **self-paced difficulty curriculum**:
- A uniform warm-up first measures the model's current response loss on individual human-authored training examples.
- The sampler then focuses progressively on moderately difficult examples based on their EMA response loss.
- At least 35% uniform replay remains in the sampling distribution so easy/common language and previously learned facts continue to be practised.
- The Colab notebook can run a separate pure-uniform control (`1.0`) and stores its SFT checkpoints/evaluation/export artifacts in a separate variant folder.
- Sampling state, schedule, per-example difficulty history and random state are stored in the SFT checkpoint for exact resume.

This design is inspired by self-paced curriculum research, including SPaCe (ACL 2026), but it is a Q-Mini-specific experiment—not proof of better intelligence. It must beat a uniform-sampling control under matched seeds, tokens and compute before being promoted as the default final recipe.

## English data policy

The pinned data path provides three complementary sources:
1. FineWiki English Wikipedia sample for factual next-token pretraining (source revision and compressed SHA-256 are pinned; license CC-BY-SA-4.0).
2. Positively reviewed, explicitly non-synthetic English OpenAssistant OASST1 conversations (Apache-2.0).
3. Human-authored Databricks Dolly instruction examples (CC-BY-SA-3.0).

Whole articles/conversation groups are split before chunking or tokenization, exact source hashes are recorded, and preparation publishes the final dataset only after every source and split succeeds. Review attribution/share-alike obligations before redistribution. No model-generated content is labelled as human-authored.

## Current scope: no phone or Android deployment

Phone deployment is **not part of the active model task**. No Realme device, Android app, mobile RAM limit, ARM/NEON optimisation or phone latency target should constrain the Colab model architecture or training algorithm.

Earlier phone-specific prototype work is preserved separately under `experiments/phone/realme_c17/`. It is parked, is not imported by `qmini_addsub/`, is not part of the main training notebook, and is not compiled by the active Add/Sub CI path. The current device label needs independent confirmation before that experimental folder is ever resumed.

## Evaluation gates

A passing unit-test suite is not a language-quality result. Promotion requires:
- Held-out English language loss and assistant response loss improvement.
- Non-collapsed, non-repetitive outputs across independent prompts and multi-turn conversations.
- Separate evaluation slices for instruction following, factual QA, arithmetic, logical reasoning, summarisation and context retention.
- No train/validation/test leakage; all manifests and artifact hashes match.
- Ablations comparing uniform SFT sampling with self-paced curriculum under matched compute and at least multiple seeds.

Keep the old model as a rollback baseline until the new model passes these gates. Do not claim human-level reasoning or general intelligence from a small synthetic smoke test.

## Development checks

CI runs software tests and tiny controlled learning/evaluation fixtures. Those checks cover implementation and reproducibility, not real-world knowledge or assistant quality.

```bash
python -m pip install -e ".[dev]"
pytest -q
```

See `qmini_addsub/README.md`, `docs/ADD_SUB_V1_SPEC.md`, `docs/EVALUATION.md`, `docs/EXPERIMENTAL_TECHNOLOGY_ROADMAP_2026.md` and `experiments/phone/realme_c17/README.md` for implementation boundaries and status.
