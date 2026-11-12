# Q-Mini algorithm replacement and training-correctness RFC

**Status:** Research proposal; no model code or weights are changed by this RFC.  
**Primary rule:** every candidate model starts from random initialization and is trained on Q-Mini's own licensed corpus. Do not call `from_pretrained()`, import downloaded checkpoints, or silently initialize from external model weights.

## 1. Decision to make

Q-Mini is intended to remain a text-only, attention-free, from-scratch language-model project. Its current experimental graph is:

- quaternary embedding and linear projections;
- causal depthwise convolution;
- a gated diagonal state-space recurrence in hybrid profiles;
- sparse Top-1 expert MLPs;
- a shared refinement block;
- an FP32-master / fake-quantized Q2-weight and INT8-activation path.

The custom recurrence already has tests comparing parallel scan with sequential recurrence, checking gradients and causality. The right next step is **not** to declare the math broken and replace everything at once. It is to compare this implementation against an independently implemented published sequence-model algorithm, under a controlled experiment, before scaling to billions of parameters.

## 2. Hard constraints

1. No pretrained model weights, adapters, LoRA weights, or external checkpoint initialization.
2. The model and tokenizer must start from random/local initialization and be trained on the project's own documented data.
3. Preserve the current no-Transformer-self-attention requirement for the primary branch. Do not add Q/K/V dot-product self-attention or softmax attention.
4. Keep the current Q-Mini implementation runnable as a baseline while candidate algorithms are evaluated.
5. Do not claim 5B training, 4GB-device inference, assistant quality, or benchmark superiority without reproducible measurements.

## 3. Recommended sequence-model candidates

### Candidate A — Mamba-2 / Structured State Space Duality (primary experiment)

Use Hugging Face Transformers' Mamba-2 implementation as an **architecture and code reference**, not as a downloaded model. It is a state-space sequence architecture and exposes recurrent/cache state for incremental generation. The official docs show configuration-based construction and explicitly state that a model initialized from a config does not load checkpoint weights.

Illustrative initialization only:

```python
from transformers import Mamba2Config, Mamba2ForCausalLM

config = Mamba2Config(
    vocab_size=local_tokenizer_vocab_size,
    hidden_size=small_experiment_width,
    num_hidden_layers=small_experiment_layers,
    state_size=small_experiment_state_size,
    # Set the remaining architecture values explicitly and record them.
)
model = Mamba2ForCausalLM(config)  # Random initialization; no checkpoint load.
```

Do **not** replace this with `Mamba2ForCausalLM.from_pretrained(...)`.

Why test it:
- it is an independently maintained SSM implementation;
- the documented cache includes SSM state and convolution state;
- it gives a useful reference for causal prefill, recurrent decode and state handling.

Cautions:
- pin a tested Transformers release and optional kernel dependencies;
- test the exact CPU/CUDA backends used by Q-Mini;
- do not assume an implementation that works on a workstation will run on ARM64;
- compare matched model sizes, token budgets, seeds and hardware.

Official references:
- https://huggingface.co/docs/transformers/main/model_doc/mamba2
- https://huggingface.co/docs/transformers/en/model_doc/mamba2

### Candidate B — RWKV-7 / Goose (secondary experiment)

RWKV-7 is an attention-free recurrent sequence-model family with dynamic state evolution. Its paper describes constant-size recurrent state, constant per-token inference work and parallelizable training.

Use the official architecture implementation and pin its exact source revision if evaluating it. Do not confuse the older built-in Hugging Face Transformers RWKV-4 implementation with RWKV-7. The Hugging Face RWKV-7 repositories contain pretrained checkpoints; those weights are **not** to be used. The code/paper may be studied or ported, but Q-Mini's experiment must create fresh parameters and train them from scratch.

References:
- https://arxiv.org/abs/2503.14456
- https://github.com/RWKV/RWKV-LM
- https://huggingface.co/docs/transformers/en/model_doc/rwkv

### Candidate C — quantization diagnostics must not change production Q2

The production matrix-weight format is fixed at exactly two bits per symbol with normalized codebook `{-1.0, -0.5, +0.5, +1.0}`. New Q2-v2 rows use scale `2 * mean(abs(original_row))`; see [Q2_QUANTIZATION_SPEC.md](Q2_QUANTIZATION_SPEC.md). This four-symbol codebook has no zero symbol: exact-zero entries in nonzero rows map to positive half-scale, while an all-zero row has scale zero and reconstructs exactly to zero.

FP16/BF16 references and INT8/INT4/ternary reconstruction helpers may be used only to measure approximation error on Q-Mini-owned checkpoints. They are not alternative production model formats and must not be selected by training, export, loading or inference. No third-party pretrained weights are permitted, and the no-self-attention architecture remains unchanged.

Recommended sequence:
1. establish a repeatable learning/evaluation baseline from Q-Mini random initialization;
2. measure per-layer Q2 reconstruction error and held-out loss against a higher-precision diagnostic reference;
3. keep Q2-v2 as the only production matrix-weight export/runtime format;
4. change QAT/threshold implementation only when tests preserve the exact four-symbol codebook, QAT/packer parity and held-out model quality.

## 4. Sparse experts: retain only after the dense path is proven

The current Top-1 load-balancing auxiliary-loss form is aligned with the Switch Transformer family. It is a reasonable baseline, but it does not prove that experts specialize or that routing remains balanced at larger scale.

Recommended experiments after the dense sequence model works:
- compare dense FFN against Top-1 MoE using the same tokenizer, training tokens, schedule and hardware;
- report per-layer expert token fractions, expert starvation, load variance, overflow and active-compute estimates;
- add a capacity factor / explicit overflow policy before claiming predictable memory and dispatch cost;
- consider router z-loss or logit regularization as a separately measured stability experiment;
- keep an unmodified reference implementation to verify routing outputs and auxiliary-loss values.

Mathematical special case: when `n_experts == 1`, `softmax([z]) == [1]` for every `z`. The router is redundant, receives no useful gradient, and should be bypassed in the dense reference. See issue #15.

Reference:
- https://jmlr.org/papers/v23/21-0998.html

## 5. Training and data algorithm recommendations

### Optimizer and numeric stability

Keep AdamW as the first controlled baseline; do not change architecture, optimizer, quantization and dataset at the same time. Add a separate parameter-group experiment that excludes bias and normalization/one-dimensional parameters from weight decay, and compare held-out loss rather than assuming it must win. Continue to log:
- finite loss and gradient checks;
- global gradient norm and clipping frequency;
- learning rate and warmup/cosine schedule;
- tokens seen and supervised target tokens;
- per-expert load and router auxiliary loss.

For a large model, BF16 mixed-precision compute with FP32 master parameters is a safer initial comparison than making a custom low-bit STE part of the first proof-of-learning run.

### Validation metrics

Periodic training validation now accumulates total masked NLL and total supervised targets, then divides once. The `eval_supervised_tokens` field records the denominator. The standalone language-model evaluator uses the same token-weighted definition. This resolves the original issue #14 finding; retain the unequal-mask-count regression test.

Also evaluate every supported refinement/thinking-step count separately; a single validation score at the default setting does not show that all step counts behave well.

### Dataset ingestion and split safety

Keep the local tokenizer-training step; Hugging Face Tokenizers supports training from iterators, so text can be supplied without loading the entire corpus into RAM:
- https://huggingface.co/docs/tokenizers/main/en/api/trainers

For very large remote or sharded corpora, Hugging Face Datasets' `IterableDataset` streaming, shard support and shuffle buffer can feed an offline preparation stage. Streaming is not random access; preserve deterministic train/validation grouping and immutable prepared token streams:
- https://huggingface.co/docs/datasets/en/stream

The current code has exact deduplication and optional disk-backed SimHash grouping. Keep the configured similarity threshold in both tokenizer fitting and token-stream preparation; add tests proving no duplicate group crosses the split.

### Multi-GPU training

The 5B configuration has 5,132,630,288 total parameters. FP32 parameters, FP32 gradients and two FP32 Adam moments alone are approximately 82.12 GB (76.48 GiB), before activations, workspaces, temporary buffers, allocator fragmentation and data pipeline memory. Top-1 routing reduces active expert compute per token; it does not remove inactive experts from the stored model or ordinary optimizer state.

Use Hugging Face Accelerate with PyTorch FSDP or DeepSpeed ZeRO only after the small single-device run is correct. These approaches shard parameters, gradients and optimizer state across devices; they do not make a 5B run free or remove the need for sufficient total GPU memory:
- https://huggingface.co/docs/accelerate/usage_guides/fsdp
- https://huggingface.co/docs/transformers/accelerate

The 4GB-device objective is an inference target, not a training-memory target.

## 6. Current repository audit snapshot

This is a code review snapshot, not a claim that every workflow has passed.

### Improvements visible in the current source

- Streaming inference now uses fixed-size convolution/SSM caches. The current model docstring says `max_context` limits batch/prompt prefill, while recurrent streaming decode can continue beyond it. This removes the old per-token full-context replay. Update issue #1's acceptance criteria to match this explicit policy and retain long-stream tests.
- `forward_step` now rejects grad-enabled inference calls.
- Dataset preparation has an incremental parser for top-level JSON arrays.
- Tokenizer and prepared-stream hashes are checked by the training pipeline.
- Checkpoint loading uses the restricted `weights_only=True` path for model/resume loading.
- CLI thinking-step validation is config-driven.
- The native C++ source is declared as package data.
- Benchmark tooling separates forward/prefill and autoregressive decode paths.
- Dataset manifests omit absolute local paths by default.
- Optional SimHash document grouping exists in the data pipeline.

The corresponding older issues may remain open even though code has changed. Reconcile each issue against the latest tests and workflow before closing it; do not infer a green CI run from the presence of a workflow file.

### Remaining concrete findings

1. `AdamW(model.parameters(), weight_decay=...)` applies decay to every parameter, including one-dimensional normalization scales and biases. Treat excluding those tensors as an experiment and report the result; it is not automatically a correctness failure.
2. Q2-v2 excludes a zero symbol: exact-zero entries in nonzero rows map to +0.5 × row scale. Keep this trade-off documented and require held-out quality measurements; do not widen or replace the production codebook.
3. The dense-vs-sparse comparison needs matched training runs. The current 5B and 853M configurations are specifications, not evidence of a trained assistant or a measured 4GB device result.
4. The latest CI run must finish successfully on the exact main commit before describing the repository as passing tests. ARM64 cross-compilation is not a real handset benchmark.

## 7. Staged implementation plan

### Stage 0 — correctness baseline
- Run and retain the current CI suite on the current main head.
- Issues #14 and #15 are now implemented on main; retain their regression tests and close only after CI passes on the code-bearing commit.
- Reconcile stale issues #1–#10 against actual implementation and tests.
- Add deterministic gradient, loss, checkpoint, data-split and cache tests.

### Stage 1 — tiny reference models
Train from random initialization:
- current Q-Mini baseline;
- a dense Mamba-2/SSD reference;
- optionally a RWKV-7 implementation in a fully isolated branch.

Start at a size that can be trained repeatedly and inspected. Keep the production matrix-weight contract fixed at Q2-v2; higher-precision tensors may be used only as a diagnostic reference for controlled comparisons.

### Stage 2 — scale the winning dense model
Scale in measured steps (for example, around 5M, 25M, 100–150M, then a larger dense reference). At every stage record tokens seen, compute, training stability, validation NLL, task scores, RAM/VRAM and throughput. Increase size only when the smaller run has a stable learning curve and reproducible quality.

### Stage 3 — sparse experts
Compare dense FFN and Top-1 MoE under matched compute and held-out evaluation. Keep all experts' storage and optimizer costs in the resource calculation.

### Stage 4 — quantization
Run reconstruction-error diagnostics on the project's own trained weights, with FP16/BF16 and optional INT8/INT4/ternary helpers as references only. Q2-v2 remains the sole production matrix-weight format; QAT changes must preserve its exact four-symbol codebook and pass task-level quality gates.

### Stage 5 — deployment
Build and test a complete inference runtime for the chosen architecture. Measure real RSS, cold start, prefill, decode tokens/sec, thermal behavior and energy on every claimed target device. Do not use a weight-file size or a planner estimate as proof of a 4GB-phone deployment.

## 8. Release gates

A candidate algorithm is accepted only when all of the following exist:
- source revision, dependency versions and complete model config;
- proof of random initialization and no external checkpoint loading;
- exact parameter count and reproducible seed;
- finite gradients and a successful small training run;
- token-weighted held-out NLL/perplexity;
- causal and full-vs-cached equivalence tests;
- dense-vs-MoE, quantized-vs-unquantized ablations where applicable;
- actual prefill/decode and peak-memory benchmarks;
- independently held-out math, multilingual, reasoning and code tests.

A model running without crashing is necessary, but it is not evidence that it has learned useful language behavior.
