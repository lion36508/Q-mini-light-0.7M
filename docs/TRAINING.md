# Training and data protocol

Place permitted text data in `Google Drive/MyDrive/QMini/datasets/`. Supported formats are TXT, Markdown, JSON, JSONL/NDJSON, CSV and optional Parquet. Conversation rows with a `messages` list or common question/answer fields are converted to dedicated role control tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`) and an explicit `<|end|>` turn marker in newly trained tokenizers. The loader does not download pretrained models or remote corpora.

## Curriculum

1. Language foundation: broad, deduplicated, well-formed text with source provenance.
2. Structured competence: mathematics, explanations, logic, programming and technical documentation.
3. Instruction and dialogue: high-quality multi-turn conversations, task demonstrations and safe tool-use examples.
4. Mixed replay: retain general language alongside target skills to reduce catastrophic forgetting. Keep held-out validation documents isolated from training.

## Pipeline

Split documents deterministically first; train the local byte-level BPE tokenizer only on unique training documents (never validation documents); then store token IDs as uint16; run a tiny smoke training; train with AdamW, gradient accumulation and warmup/cosine scheduling; evaluate periodically; save model/optimizer/config/RNG checkpoints; export model and tokenizer together.

The trainer refuses to overwrite an existing checkpoint unless `--resume` is set. Token preparation refuses to overwrite existing train/valid streams. For version-3 prepared datasets, the trainer verifies tokenizer, train/validation stream and supplied mask SHA-256 fingerprints against dataset.json; auto/assistant loss modes require both response masks. Resume rejects a mismatched config, data, tokenizer or mask. Begin with batch size 1, sequence length 128 and gradient accumulation 1; scale only after measuring the actual accelerator memory.

Checkpoint trust: use `latest.pt` only from a training run you created or otherwise trust. Resume and model initialization use PyTorch's restricted `weights_only=True` loader; arbitrary pickle loading is not used. SFT `--init-from` additionally requires a Q-Mini `final_model/` manifest, matching architecture config, validated Q2 export, and matching hashes for the selected dense/packed weights. When `--tokenizer` is supplied, it must match the tokenizer saved with that checkpoint. Standalone weight files without this provenance are rejected. This binds the artifacts together but is not a cryptographic signature of who created them; do not trust files from an untrusted publisher. Keep a copy of a known-good checkpoint before resuming or fine-tuning.

The compact convolution-only qmini_base.json and historical qmini_research_134m.json profiles remain available for compatibility. The updated Colab pipeline uses qmini_research_152m.json, a 152,483,800-parameter hybrid CausalConv + gated diagonal SSM + sparse Top-1 MoE profile. The separate qmini_5m.json and qmini_5b.json profiles provide capacity estimates. The 5B profile is not distributed-training-ready: its FP32 parameter/gradient/AdamW state alone is about 82.1 GB before activations. More parameters, steps or refinement iterations do not automatically improve quality.


## Training budget and staged fine-tuning

The Colab setup installs the Q-Mini package with `--no-deps` so dependency resolution does not silently replace Colab's preinstalled CUDA-enabled PyTorch wheel. It separately installs the Python data/test dependencies and checks that the runtime PyTorch version is at least 2.10.0. The 50-step smoke proof may fall back to CPU for basic code-path validation, but the optional 5M, 152M and SFT training stages fail closed unless CUDA is available. The 152M and SFT stages also require at least 8 GiB of currently free GPU memory by default; this is a launch floor, not a guarantee against out-of-memory errors. Actual memory must still be monitored during the first microbatches.

The Colab notebook uses a configurable initial training-token budget based on 20 tokens per unique active Top-1 parameter. This is only a compute-budget heuristic; sparse-MoE scaling has to be measured. It reports how many passes the requested token budget implies over the available training stream, making corpus repetition visible.

For a separate instruction/SFT stage, use `--init-from /path/to/pretrain/final_model/model.pt`. The trainer loads only the model state dictionary and creates a new optimizer. The source path/hash is kept in the run manifest and verified when resuming that fine-tuning run. Use response-only masks for structured chat examples.

## 5B resource boundary

The 5B profile's FP32 weights, gradients and two AdamW moments require approximately 82.1 GB before activation memory, workspaces or fragmentation. The current trainer is single-process and lacks FSDP/ZeRO and expert/tensor parallelism. Do not treat the 5B file as a ready-to-run H100 training job.


## Near-duplicate isolation

Exact SHA-256 deduplication is always active. Optionally group near-duplicates using a disk-backed 64-bit SimHash over word trigrams. The threshold is the maximum Hamming distance from 1 to 3; zero disables SimHash grouping and preserves exact-only behavior. The entire similarity group is assigned to one split, and the dataset manifest records the threshold, group count and number of documents absorbed into groups.

Use the **same** holdout fraction and near-duplicate distance in both commands:

\`\`\`bash
python -m qmini.cli tokenizer --data-dir DATA --output tokenizer.json \\
  --vocab-size 32768 --valid-fraction 0.02 \\
  --near-duplicate-hamming-distance 3 \\
  --config-output config.json

python -m qmini.cli prepare --data-dir DATA --tokenizer tokenizer.json \\
  --output-dir prepared --valid-fraction 0.02 \\
  --near-duplicate-hamming-distance 3 --loss-mask auto
\`\`\`

This is an approximate similarity filter, not semantic plagiarism detection. Inspect the grouped-document counts and keep the validation corpus independently curated.


## Validation metric definition

Periodic \`eval_loss\` is the global token-weighted cross-entropy over all sampled validation targets:

\`eval_loss = sum(masked per-token NLL) / sum(supervised target tokens)\`.

For assistant-only datasets, context tokens do not contribute to the numerator or denominator. The \`eval_supervised_tokens\` field records the denominator. \`eval_perplexity\` is \`exp(eval_loss)\` when representable. This matches standalone \`eval-lm\` and avoids bias from an unweighted arithmetic mean of batch means with unequal response lengths.



## Colab learning proof and optional own-checkpoint SFT

\`colab/QMini_Training.ipynb\` now defaults to a **50-step small-model smoke run**. Its learning-proof gate requires at least 0.05 held-out cross-entropy improvement (before minus after) on the fixed evaluation seed and records the measured margin in \`learning_proof_*.json\` file in Drive. A long 152M research run is disabled until the user explicitly sets \`RUN_LONG_TRAINING = True\`; it has no arbitrary maximum optimizer-step cap. Set LONG_OPTIMIZER_STEPS_OVERRIDE to the total step target, or leave it None to use the documented token-budget heuristic. The target is the total optimizer-step count, not an extra count after resume. Increasing it reuses the same Drive run directory and checkpoint, then extends the saved cosine schedule. The step target is excluded from the run identity; source revision, implementation hash, data/tokenizer hashes, model config and optimizer settings remain part of the identity. The 5B configuration is not a Colab training target.

The optional SFT stage is disabled by default (\`RUN_OWN_CHECKPOINT_SFT = False\`). Put a separate licensed chat/instruction corpus under \`MyDrive/QMini/artifacts/sft_datasets/\`, with rows containing \`messages\` or supported instruction/answer fields, then set the flag to \`True\`. The notebook:
1. Uses the matching tokenizer/config from a Q-Mini-produced checkpoint.
2. Prepares a separate SFT dataset with assistant-only loss masks and a held-out split.
3. Initializes only from the Q-Mini model's verified Q2-v2 packed \`model.qmini.npz\` artifact via \`--init-from\`; no external model weights are accepted and the new optimizer is fresh. Current final-model exports are packed-only: dense model.pt snapshots are not accepted as SFT sources, including legacy files left in an older final-model directory.
4. Resumes to a new hash-keyed run directory on Drive.
5. Optionally compares the SFT checkpoint with its pre-SFT source on the separate \`sft-heldout-tasks.jsonl\` file, reporting task-pass-rate and per-language changes.

An NLL decrease or task-score increase on a small sample is evidence only for that run and dataset. Review sample counts and cases; do not generalize it to broad assistant capability.




## Reliable sampling with sparse response masks

When a prepared assistant-only mask contains very few positive target tokens, the trainer builds a disk-backed \`SupervisedWindowIndex\` once before model/optimizer allocation. It counts valid windows in bounded blocks and samples uniformly from all starts whose shifted label mask contains at least one supervised target. It preserves the exact alignment \`mask[start + 1:start + seq_len + 1]\`. It fails early only when the stream has no legal supervised window, rather than failing probabilistically after 128 random proposals.




Current final-model exports contain the verified Q2-v2 model.qmini.npz deployment artifact; they do not include or fingerprint a dense model.pt snapshot. A legacy model.pt that remains in an older output directory is deliberately not trusted as an SFT source. SFT initialization is .npz-only. The resumable latest.pt training checkpoint is a separate training-state artifact, not a final-model deployment or SFT source.



## Hypothetical distributed memory planning (not a distributed trainer)

The resource planner accepts --world-size N --sharding-scenario fsdp-full-shard or --sharding-scenario zero3 to divide the FP32 parameter/gradient/AdamW-state lower bound by a hypothetical number of ranks. The output explicitly reports distributed_training_implemented: false. This is an idealized arithmetic lower bound, not an FSDP/ZeRO measurement: it excludes all-gather/reshard peaks, activation memory, optimizer implementation details, communication buffers, fragmentation, rank imbalance and data-loader state. The current trainer still runs on one device. Do not use the planner's per-rank number as evidence that a multi-GPU 5B run is supported.


The CLI requires an explicit total optimizer-step target via --steps; it has no implicit 1,000-step default. For example, --steps 50000 means train until a total of 50,000 optimizer updates. Resume with --resume; when increasing the saved scheduler horizon, pass --extend-schedule. The Colab notebook performs that extension when its user-selected target is increased.

## Assistant SFT source and evaluation gates

The Colab SFT stage does not silently fine-tune the tiny smoke checkpoint. By default it selects the largest completed base_*/final_model artifact with a valid Q-Mini final-model manifest, matching tokenizer and verified packed-export hash, and at least 50 million parameters. A smaller experiment requires an explicit SFT_SOURCE_OVERRIDE. SFT is refused unless independent held-out task and multi-turn conversation sets both exist and neither is byte-identical to an SFT training file. Both held-out hashes are included in the SFT run identity; both baselines run before SFT and both post-SFT evaluations run afterward. This prevents an unmeasured SFT run, but does not itself guarantee improved dialogue quality.


## Empty-brain architecture preflight

Before attaching a corpus or starting an optimizer run, the model graph can be
checked using only synthetic token IDs and fresh random Q-Mini weights:

```bash
python -m qmini.cli audit --config configs/qmini_research_152m.json
python -m qmini.cli preflight \
  --config configs/qmini_research_152m.json \
  --device auto --sequence-length 8
```

`audit` is a config-only operation and never materializes model tensors, so it
is safe to use for the 5B configuration. `preflight` is a bounded dry-run: it
checks the model's actual parameter count, forward/logit shapes, finite outputs
and (by default) finite non-zero gradients. It refuses profiles above 250M
parameters unless the limit is explicitly raised. It does not open a dataset,
load pretrained weights, call an optimizer, perform an optimizer step or alter
the random-initialized weights.

The trainer independently checks measured free device/host memory before model
allocation. The default guard requires at least 2x the FP32
parameters+gradients+AdamW-state lower bound. This is a guardrail, not a full
activation/workspace forecast; the Colab 152M path retains its stricter 8 GiB
free-memory launch floor and first-step OOM monitoring.


## Generation and sampling

The generate CLI supports `--frequency-penalty` and `--presence-penalty`, each defaulting to zero and accepting values in [-2, 2]. Positive values reduce logits for tokens already generated in the current completion; the frequency penalty scales with the number of occurrences and the presence penalty applies once per seen token. These are sampling controls, not a claim of improved model capability. The CLI's opt-in `--serialized-prompt` bypasses literal control-token escaping and prints a warning; use it only for prompts produced by Q-Mini's trusted serializer.

Generation now prefills the prompt with a batched causal-convolution/SSM pass and carries its resulting cache into incremental token decoding. This removes the Python-level one-token forward loop during initial prompt processing. The implementation is covered by tokenwise-versus-prefill parity tests; no performance improvement is claimed until measured on a specified device.
