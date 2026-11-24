# Q-Mini Add/Sub — Colab-only primary model

An independent English-first model developed and trained through the repository's Google Colab notebook. This phase is about learning quality and model behavior, not phone deployment. The legacy Q2-v2 package and artifacts remain separate. It borrows the ideas of the legacy Q-Mini—causal depthwise convolution, gated diagonal state-space memory and sparse Top-1 experts—but implements its own code, configuration, tokenizer, training checkpoints and data lineage.

**Default experimental profile:** 1,024-token vocabulary, 96-wide state, four blocks with causal dilations 1/2/4/8, two Top-1 SwiGLU experts, 170-wide expert bottleneck, and 512-token context. The instantiated training model has **760,340 parameters**. This size is a design baseline, not evidence of better model quality.

Each block uses RMSNorm, causal depthwise convolution, an input-dependent gated diagonal state-space update (parallel associative scan), sparse Top-1 SwiGLU experts and learned bounded residual branch gates. Hard activation clipping is avoided so gradients are not zeroed at a clipping boundary. The same old-model ideas have been independently implemented, not imported from the legacy `qmini/` package.

## Colab SFT learning algorithm — self-paced curriculum v1

The assistant SFT stage uses a difficulty-aware sampler rather than uniform-only sampling:

- **Uniform warm-up:** first collect per-example assistant-response losses from the current model.
- **Adaptive difficulty focus:** rank training examples by an exponential moving average of response loss. The target difficulty gradually moves from the 55th to the 80th percentile, focusing on examples that are challenging but not only the most pathological/high-noise records.
- **Spaced replay:** at least 35% of the sampling distribution remains uniform, so easy/common language and already-learned behavior continue to be practised.
- **Learning-state resume:** per-example EMA losses, update counts, sampler schedule and RNG state are stored in the SFT checkpoint. A different curriculum contract cannot be resumed silently.

This is inspired by the self-paced curriculum direction explored in SPaCe (Findings of ACL 2026): https://aclanthology.org/2026.findings-acl.171/. Its published results do not prove a gain at Q-Mini's 760K scale. The Colab notebook must compare self-paced sampling with a uniform-sampling control under matched data, seeds and optimizer-step budgets before the curriculum is treated as a quality improvement.

## Add/Sub v2 algorithm-stability update

The training algorithm now includes three explicit stability controls while preserving the 760,340-parameter default architecture and the five-symbol codebook:

- Residual branch gates initialize at **0.1**, not 0.5. The convolution, state-space and expert branches begin as small residual corrections and can learn stronger contributions from data.
- Sparse Top-1 routing keeps the existing load-balancing objective and adds a separately weighted **router z-loss** on the log-partition. This discourages uncontrolled router-logit growth without hiding the coefficient inside the load-balancing term.
- Every training/SFT metric row records **per-layer expert assignment fractions** plus the two router losses. Padding tokens are excluded from the balance and z-loss statistics.

The pretraining and SFT checkpoint format identifiers are bumped so a run made with the older loss contract cannot be resumed accidentally. The offline exporter can still read prior shape-compatible checkpoint formats for conversion; this does not make their training trajectories interchangeable.

These changes are algorithmic stability measures, not a claim of improved language quality. That requires a real, reviewed English corpus and held-out validation after training.
 
## Implementation status

**Implemented in this branch**
- Independent random-initialized training model and validated configuration.
- Five-symbol QAT weight alphabet: {-1.5, -0.5, 0, +0.5, +1.5}.
- Three-bit packing/unpacking with invalid-code and non-canonical padding rejection.
- Checkpoint-to-archive export of every five-symbol model weight, a separate auxiliary tensor sidecar, SHA-256 manifest validation and a strict reconstruction loader with parity/tamper tests.
- A Python additive projection reference primitive for isolated arithmetic research. Native phone-specific projection experiments are parked under `experiments/phone/realme_c17/` and are not part of the primary model package or active CI.
- Pinned non-synthetic English data preparation.
- Independent byte-level BPE tokenizer and token-stream preparation scripts.
- Random-init pretraining with linear warm-up/cosine learning-rate decay, vector/norm no-decay AdamW groups and Add/Sub-only checkpoint resume.
- Response-masked instruction fine-tuning from an Add/Sub v1 checkpoint, with deterministic Python/Torch RNG checkpoint state and resumable SFT v4 checkpoints, plus a separate held-out SFT test split.
- Cached one-token reference generation with per-layer causal-convolution history and diagonal-SSM state, plus an interactive chat CLI.

**Not yet complete**
- The packed `.qma` export is an offline model-weight container. Its auxiliary normalization, bias and gate tensors are stored as float32; its loader reconstructs the PyTorch reference model. It is not the integer-only deployed inference graph.
- The cached PyTorch generation path avoids reprocessing the full prompt and remains a floating-point reference implementation for Colab evaluation. It is not a mobile runtime.
- Phone-specific integer inference is out of scope for the active Colab model. Preserved experiments live under `experiments/phone/realme_c17/` and must not constrain the current architecture.
- No Add/Sub weights have yet completed the real-data Colab training recipe; assistant quality has not been established.

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

## Bring your own selected dataset (no download)

The model does not require the repository's optional pinned corpus. After reviewing a dataset yourself, put three already-separated UTF-8 English text files in a folder:

- `train.txt`
- `validation.txt`
- `test.txt`

The files must be distinct and should already be split by document/source group to avoid leakage. The importer records the user-declared source, license, file hashes and split paths; it cannot independently prove that a source is human-authored or that plain-text splits have no document-level overlap. You must explicitly attest both that the data is English and that you reviewed the rights/non-synthetic provenance; these flags are declarations, not automatic language, license or authorship verification.

Import only the local files (no network download is performed):

    python scripts/import_qmini_addsub_dataset.py --pretrain-train /path/train.txt --pretrain-validation /path/validation.txt --pretrain-test /path/test.txt --pretrain-source "dataset name and revision" --pretrain-license "exact source license" --confirm-rights-and-human-data --confirm-english-data --output data/qmini_addsub_v1

For supervised assistant training, supply all three files `--sft-train`, `--sft-validation` and `--sft-test`, plus `--sft-source` and `--sft-license`. Each JSONL record must contain alternating `user`/`assistant` messages, `human_authored: true`, and `synthetic: false`. The importer normalizes messages and rejects exact duplicate conversations across every pair of splits. SFT reports validation response loss during training and held-out test response loss only after training is complete. Do not mark generated or unknown-provenance text as human-authored.

Then build the tokenizer and streams. SFT data is optional at this stage; the tokenizer is trained on pretraining train text alone when no SFT train file exists. The tokenizer script also writes `model_config.json` with the exact vocabulary size it actually produced; pass this file to training so tokenizer and model cannot silently disagree:

    python scripts/train_qmini_addsub_tokenizer.py
    python scripts/encode_qmini_addsub_streams.py

The pinned FineWiki/OASST1/Dolly preparation command remains optional. No dataset is downloaded by importing the model package or by running CI. The Add/Sub Colab notebook has data import, tokenizer build, pretraining and SFT switches OFF by default.

## English training-data mix

The default preparation script uses three complementary English sources:

1. **FineWiki English Wikipedia sample** for factual next-token pretraining. It is a pinned, SHA-256-verified 1% random sample (about 210 MB compressed) from an August 2025 Wikimedia Enterprise English Wikipedia snapshot. The source was cleaned by FineWiki to remove template/markup noise and retains readable article structure. Every Wikipedia page ID is deterministically assigned to exactly one of train/validation/test before its text is chunked, so chunks from one article cannot leak across splits. License: CC-BY-SA-4.0; preserve attribution to Wikipedia contributors and FineWiki and review share-alike obligations.
2. **OpenAssistant OASST1** for English human-annotated multi-turn conversation data. Only ready-for-export, positively reviewed, non-deleted, explicitly non-synthetic messages and rank-0 assistant answers are retained. The pinned source revision/file hash is verified. License: Apache-2.0.
3. **Databricks Dolly 15k** for human-authored instruction/response examples. Dataset revision is pinned and source file SHA-256 is recorded at preparation time. License: CC-BY-SA-3.0.

The complete FineWiki English dataset is much larger than this sample and is deliberately not downloaded by default; the 1% sample is the practical CPU/Drive starting point, not a claim that it contains all Wikipedia knowledge. Wikipedia supplies factual breadth, while OASST1 and Dolly provide assistant-style dialogue and instructions. No single corpus is "the world's best" across all tasks.

DailyDialog is deliberately excluded from this default mix because its dataset card lists CC-BY-NC-SA-4.0 non-commercial/share-alike conditions. TinyStories, Alpaca/UltraChat, teacher-generated outputs and other model-generated corpora are excluded from production training.

Review each source's terms before use or redistribution. Attribution/share-alike requirements can affect downstream releases. The preparation script requires the explicit --accept-source-licenses flag.

## Reproducible build steps

Install data preparation tools without replacing the environment's existing PyTorch:

    python -m pip install "huggingface_hub>=0.28" "pyarrow>=15.0" "zstandard>=0.22"

Prepare pinned data:

    python scripts/prepare_qmini_addsub_data.py --accept-source-licenses

Preparation uses a temporary sibling staging directory and publishes the final dataset folder only after Wikipedia, OASST1, Dolly, all three splits and the manifest have completed. A failed source download removes the partial staging output so a retry does not inherit half-written training files. The download cache must remain outside the prepared output folder; cached downloads can be reused after a failed attempt. The target output folder must be empty or new and is never overwritten.

Train the tokenizer using only training splits, then encode pretraining text:

    python scripts/train_qmini_addsub_tokenizer.py
    python scripts/encode_qmini_addsub_streams.py

Start a new model from random initialization:

    python -m qmini_addsub.train --train-bin data/qmini_addsub_v1/pretrain_train.u16 --valid-bin data/qmini_addsub_v1/pretrain_validation.u16 --tokenizer artifacts/qmini_addsub_v1/tokenizer.json --config artifacts/qmini_addsub_v1/model_config.json --output runs/qmini_addsub_v1_pretrain --device cpu --steps 2000

After a pretraining or SFT checkpoint exists, create a verified offline five-symbol weight archive:

    python -m qmini_addsub.export --checkpoint runs/qmini_addsub_v1_sft/sft-checkpoint-step-000000500.pt --output artifacts/qmini_addsub_v1/model.qma

The archive contains all quantized matrix/embedding/depthwise weights packed at 3 bits per symbol, checksums, a configuration/tensor index and a float32 auxiliary tensor sidecar. Load it for PyTorch reference parity/tamper checks with `load_addsub_export`. It is not yet a production integer-only/mobile runtime.

After a checkpoint exists, the PyTorch reference chat can be opened with `python -m qmini_addsub.generate --checkpoint <checkpoint.pt> --tokenizer artifacts/qmini_addsub_v1/tokenizer.json --interactive`. The reference generator applies a recent-token repetition penalty and stops before an exact four-token phrase would be repeated for a fifth time; this is a last-resort loop guard, not a substitute for good training. A stopped/short answer can still fail the held-out quality gate. This is a Colab-side reference chat path only; it is not presented as a phone runtime.

Instruction fine-tuning may begin only from a checkpoint produced by this Add/Sub v1 pretraining code:

    python -m qmini_addsub.sft --base-checkpoint runs/qmini_addsub_v1_pretrain/checkpoint-step-000002000.pt --tokenizer artifacts/qmini_addsub_v1/tokenizer.json --train-jsonl data/qmini_addsub_v1/sft_train.jsonl --valid-jsonl data/qmini_addsub_v1/sft_validation.jsonl --test-jsonl data/qmini_addsub_v1/sft_test.jsonl --output runs/qmini_addsub_v1_sft --device cpu

Paths/step counts above are examples; only run after checking data preparation, available storage/compute and validation. No training begins during CI or by importing this package.

## Held-out assistant evaluation

After SFT, run greedy generation only on the manifest-declared held-out test split:

    python -m qmini_addsub.evaluate --checkpoint runs/qmini_addsub_v1_sft/sft-checkpoint-step-000000500.pt --tokenizer artifacts/qmini_addsub_v1/tokenizer.json --test-jsonl data/qmini_addsub_v1/sft_test.jsonl --dataset-manifest data/qmini_addsub_v1/dataset_manifest.json --output runs/qmini_addsub_v1_eval

The evaluator verifies the test-file hash and tokenizer fingerprint, keeps prior turns as context, and writes every generated answer/reference pair plus exact-match and token-overlap metrics. These are lexical baselines only; they do not prove factuality, reasoning or semantic correctness. Never use the SFT training or validation split as the reported held-out result.

## Fail-closed Add/Sub assistant readiness gate

After held-out evaluation, run the readiness gate against the same SFT metrics and held-out report:

    python -m qmini_addsub.readiness --metrics runs/qmini_addsub_v1_sft/sft_metrics.jsonl --heldout-summary runs/qmini_addsub_v1_eval/heldout_summary.json --output runs/qmini_addsub_v1_eval/readiness.json

By default this requires at least two validation-response-loss measurements in the same SFT data/tokenizer run, at least 1% relative validation-loss improvement, at least 50 independent held-out conversations, zero generation failures, mean lexical token F1 of at least 0.15, at least 40% unique normalized answers, no single answer repeated on more than 35% of prompts, no more than 8% repeated four-word n-grams inside answers, and no four-word phrase repeated more than four times within any single answer. It also verifies the held-out test/tokenizer/data-manifest fingerprints against the SFT run. This catches both identical-answer collapse across prompts and obvious phrase loops within a generated answer. The thresholds can be explicitly changed on the command line, but must not be weakened merely to make a run pass.

A passing report is labelled `ready_for_review`, not human-level. The gate detects training stagnation, lineage mismatch, generation errors, repeated answers across prompts and repeated four-word phrases inside individual answers. It does not prove semantic understanding, factuality, reasoning, safety or general human-like conversation; open-ended English answers still need human review against independently verified references. A model that fails remains `not_ready` and must not be presented as a finished assistant.

## Inference arithmetic contract

The required weight codebook is {-1.5, -0.5, 0, +0.5, +1.5}. Five symbols require at least three bits; they cannot be encoded in fixed two-bit storage.

The real inference path must not use multiplication, division, matmul, dot products, convolution multiply, floating rescaling or fused multiply-add. Quantized activation/weight contributions are obtained from versioned lookup tables and accumulated by addition/subtraction. Zero maps to no contribution. This contract is not considered satisfied until the entire graph—including convolution, recurrent state, expert routing, nonlinearities, embedding/output projection and sampling—is audited.

See the full architecture and acceptance contract in docs/ADD_SUB_V1_SPEC.md.


## Data-cleaning contract

Pinned English sources pass through a fail-closed, versioned cleaner before the SFT split is written. Text is normalized to Unicode NFC, line endings and horizontal whitespace are normalized, and unsafe control characters are removed. Empty/short and overlong fields, email addresses, credential-like strings and explicitly labelled phone-contact strings are rejected with per-field reason counters in `dataset_manifest.json`.

OASST1 still requires `lang == "en"`, `ready_for_export`, positive review, `deleted is False`, `synthetic is False`, and rank-zero assistant responses. Dolly is pinned to a fixed dataset revision and is treated as an English source. This cleaner does not claim to infer language from arbitrary text: language provenance comes from those source contracts. Exact duplicate conversations are removed before stable whole-conversation train/validation/test splitting. Raw pinned downloads remain in the cache and are not overwritten.

Synthetic tests and benchmarks remain outside the production corpus. A cleaning pass never converts a synthetic sample into a human-authored one.


## Training speed measurement

The active Colab notebook selects up to four PyTorch CPU threads and passes that setting to both pretraining and SFT. Training metrics record measured step duration and effective tokens/second (or supervised assistant tokens/second); these are measurements from the actual runtime, not promised performance figures. The thread count is part of the run contract, so resuming with a different thread count fails closed instead of silently changing the run configuration. Keep batch size low enough for the Colab RAM limit; increasing learning rate is not a safe shortcut to faster learning.

## Scaled QAT and balanced tokenizer update

Dense Kaiming-normal master weights now use per-tensor scaled five-symbol STE (scale = 1.4 × mean absolute weight). Zero is exact and STE gradients remain live through zero master weights. Matrix weight decay is 0.01. The 1,024-token ByteLevel tokenizer samples up to 64 MiB from pretraining training text and includes all SFT training messages so conversational terms influence BPE merges. Validation and test are never used for tokenizer training.

FineWiki article text is paragraph-chunked without splitting page identities across data splits. SFT and generation use `You are Q-Mini.`; generation applies a configurable repetition penalty. New checkpoint IDs prevent resuming old optimizer states. Packed archive v2 saves tensor scales and retains v1 reading support. Old 710,676-parameter Drive files are preserved and need new training for the 760,340-parameter model. Phone deployment is explicitly outside the active Colab-only scope.


### Pretraining corpus selection

The default pinned pretraining source is `agentlans/HuggingFaceFW-finewiki-sample`, file `en.jsonl.zst`, source revision `e0bbfbb`. Its compressed file SHA-256 is checked before use. The file is a 1% random sample of cleaned English FineWiki articles from the August 2025 Wikimedia Enterprise snapshot. Rows must declare `in_language=en` and `wikiname=enwiki`; articles need a valid page ID, title and usable text. Page IDs and exact normalized article-body duplicates are removed before deterministic 90/5/5 train/validation/test assignment; splitting occurs before paragraph-aware chunking, preserving whole-article isolation. Near-duplicate paraphrases are not claimed to be detected. The source license is CC-BY-SA-4.0. The manifest records the pinned file hash, source revision, license, sample method, attribution, split hashes, article/chunk counts and rejected rows. The complete FineWiki English dataset is approximately 35.1 GB and is not downloaded by default.
