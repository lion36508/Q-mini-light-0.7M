# Q-Mini Add/Sub v1

An independent next-generation model line inside the same repository. It borrows the ideas of the legacy Q-Mini—causal depthwise convolution, gated diagonal state-space memory and sparse Top-1 experts—but implements its own code, configuration, tokenizer, training checkpoints and data lineage.

**Default experimental profile:** 512-token vocabulary, 96-wide state, four blocks with causal dilations 1/2/4/8, two Top-1 SwiGLU experts, 170-wide expert bottleneck, and 512-token context. The instantiated training model has **710,676 parameters**. This size is a design baseline, not evidence of better model quality.

Each block uses RMSNorm, causal depthwise convolution, an input-dependent gated diagonal state-space update (parallel associative scan), sparse Top-1 SwiGLU experts and learned bounded residual branch gates. Hard activation clipping is avoided so gradients are not zeroed at a clipping boundary. The same old-model ideas have been independently implemented, not imported from the legacy `qmini/` package.

## Implementation status

**Implemented in this branch**
- Independent random-initialized training model and validated configuration.
- Five-symbol QAT weight alphabet: {-1.5, -0.5, 0, +0.5, +1.5}.
- Three-bit packing/unpacking with invalid-code rejection.
- Integer additive projection reference primitive; its AST regression test prohibits multiply, divide and matmul operators.
- Pinned non-synthetic English data preparation.
- Independent byte-level BPE tokenizer and token-stream preparation scripts.
- Random-init pretraining with linear warm-up/cosine learning-rate decay, vector/norm no-decay AdamW groups and Add/Sub-only checkpoint resume.
- Response-masked instruction fine-tuning from an Add/Sub v1 checkpoint.
- Cached one-token reference generation with per-layer causal-convolution history and diagonal-SSM state, plus an interactive chat CLI.

**Not yet complete**
- The cached PyTorch generation path avoids reprocessing the full prompt, but it is a reference path and still uses floating-point operators. The complete integer-only inference graph is not yet wired through additive lookup tables; the current integer runtime implements the projection primitive only.
- Full-model export/loader, calibrated activation/state/gating LUTs and compiled ARM64/Android runtime are not implemented.
- No Add/Sub weights have been trained and no quality/phone performance results exist.

Do not use the PyTorch training graph as a production inference runtime. The training graph intentionally uses floating-point operations and optimizer arithmetic; the project requirement is that the deployed inference graph use only lookup, comparisons, integer addition and subtraction.

## Legacy and new model separation

| Legacy model | Add/Sub v1 |
|---|---|
| qmini package | qmini_addsub package |
| Q2-v2 four symbols | Five symbols including exact zero |
| Existing tokenizer/checkpoints | Independent tokenizer/checkpoints |
| Existing Drive data | Separate QMini-Light-AddSub-v1 folder |
| Existing code is preserved | Random initialization only |

Never load a legacy Q2-v2 checkpoint into this package. Do not overwrite or migrate old model files.

## English training-data mix

The default preparation script uses:

1. **WikiText-103 raw** for next-token language pretraining. The source archive is SHA-256 pinned; source metadata reports CC-BY-SA/GFDL terms (upstream raw metadata does not pin one single version).
2. **OpenAssistant OASST1** for English human-annotated multi-turn conversation data. Only ready-for-export, positively reviewed, non-deleted, explicitly non-synthetic messages and rank-0 assistant answers are retained. The pinned source revision/file hash is verified. License: Apache-2.0.
3. **Databricks Dolly 15k** for human-authored instruction/response examples. Dataset revision is pinned and source file SHA-256 is recorded at preparation time. License: CC-BY-SA-3.0.

DailyDialog is deliberately excluded from this default mix because its dataset card lists CC-BY-NC-SA-4.0 non-commercial/share-alike conditions. TinyStories, Alpaca/UltraChat, teacher-generated outputs and other model-generated corpora are excluded from production training.

Review each source's terms before use or redistribution. Attribution/share-alike requirements can affect downstream releases. The preparation script requires the explicit --accept-source-licenses flag.

## Reproducible build steps

Install the data-preparation extras without replacing the environment's existing PyTorch:

    python -m pip install -e ".[addsub-data]"

Prepare pinned data:

    python scripts/prepare_qmini_addsub_data.py --accept-source-licenses

Train the tokenizer using only training splits, then encode pretraining text:

    python scripts/train_qmini_addsub_tokenizer.py
    python scripts/encode_qmini_addsub_streams.py

Start a new model from random initialization:

    python -m qmini_addsub.train --train-bin data/qmini_addsub_v1/pretrain_train.u16 --valid-bin data/qmini_addsub_v1/pretrain_validation.u16 --tokenizer artifacts/qmini_addsub_v1/tokenizer.json --output runs/qmini_addsub_v1_pretrain --device cuda --steps 2000

After a checkpoint exists, the PyTorch reference chat can be opened with `python -m qmini_addsub.generate --checkpoint <checkpoint.pt> --tokenizer artifacts/qmini_addsub_v1/tokenizer.json --interactive`. This is for correctness testing, not the multiplication-free phone runtime.

Instruction fine-tuning may begin only from a checkpoint produced by this Add/Sub v1 pretraining code:

    python -m qmini_addsub.sft --base-checkpoint runs/qmini_addsub_v1_pretrain/checkpoint-step-000002000.pt --tokenizer artifacts/qmini_addsub_v1/tokenizer.json --train-jsonl data/qmini_addsub_v1/sft_train.jsonl --valid-jsonl data/qmini_addsub_v1/sft_validation.jsonl --output runs/qmini_addsub_v1_sft --device cuda

Paths/step counts above are examples; only run after checking data preparation, available storage/compute and validation. No training begins during CI or by importing this package.

## Inference arithmetic contract

The required weight codebook is {-1.5, -0.5, 0, +0.5, +1.5}. Five symbols require at least three bits; they cannot be encoded in fixed two-bit storage.

The real inference path must not use multiplication, division, matmul, dot products, convolution multiply, floating rescaling or fused multiply-add. Quantized activation/weight contributions are obtained from versioned lookup tables and accumulated by addition/subtraction. Zero maps to no contribution. This contract is not considered satisfied until the entire graph—including convolution, recurrent state, expert routing, nonlinearities, embedding/output projection and sampling—is audited.

See the full architecture and acceptance contract in docs/ADD_SUB_V1_SPEC.md.


## Data-cleaning contract

Pinned English sources pass through a fail-closed, versioned cleaner before the SFT split is written. Text is normalized to Unicode NFC, line endings and horizontal whitespace are normalized, and unsafe control characters are removed. Empty/short and overlong fields, email addresses, credential-like strings and explicitly labelled phone-contact strings are rejected with per-field reason counters in `dataset_manifest.json`.

OASST1 still requires `lang == "en"`, `ready_for_export`, positive review, `deleted is False`, `synthetic is False`, and rank-zero assistant responses. Dolly is pinned to a fixed dataset revision and is treated as an English source. This cleaner does not claim to infer language from arbitrary text: language provenance comes from those source contracts. Exact duplicate conversations are removed before stable whole-conversation train/validation/test splitting. Raw pinned downloads remain in the cache and are not overwritten.

Synthetic tests and benchmarks remain outside the production corpus. A cleaning pass never converts a synthetic sample into a human-authored one.
