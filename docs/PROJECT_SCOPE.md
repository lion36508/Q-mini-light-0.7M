# Q-Mini — Project Scope and Capability Specification

**Purpose:** record the intended product direction, non-negotiable architecture rules, capability goals, implementation status and acceptance criteria. This document is a scope contract, not a claim that every roadmap feature already exists.

## 1. Product objective

Build a from-scratch, text-only language model that can ultimately serve as a compact general-purpose assistant. The desired assistant should support natural conversation, instruction following, technical explanation, mathematics and logic, science/engineering questions, programming assistance and structured text tasks. It should progressively become efficient enough for local and offline inference.

The project is not an Android IDE, not an image model, not a multimodal model and not a wrapper around a third-party pretrained chatbot.

## 2. Hard architecture constraints

- No Transformer self-attention.
- No Q/K/V projections, dot-product attention, softmax attention or retrieval-attention.
- No image encoder, image CNN, vision adapter, VAE, GAN or video model branch.
- Legacy profiles use strictly causal left-padded 1D depthwise convolution. New hybrid profiles add a gated diagonal state-space recurrence with a differentiable parallel scan during training and an equivalent single-token recurrent update during streaming inference.
- Decoder-only next-token language modeling, initialized from random parameters.
- Sparse Top-1 expert routing in the current baseline.
- Production Q2-v2 normalized weight alphabet {-1, -0.5, +0.5, +1} with FP32 master parameters and STE; legacy v1 archives remain readable.
- Per-token AbsMax INT8 activation fake quantization with STE.
- RMSNorm and residual paths.
- Shared iterative refinement reuses one block for a configurable number of passes; it is not a symbolic proof engine.
- The new diagonal state-space layer is a compressed learned state, not a lossless transcript or exact retrieval memory. It is not an official Mamba or RWKV implementation.
- Use local, open-source-first tooling. No required paid API, hosted model or pretrained model weights.

The convolution mentioned here is a text-sequence operator; it does not imply image or visual processing.

## 3. Current baseline

| Item | Baseline |
|---|---:|
| Vocabulary target | 4,096; actual tokenizer size is saved in active config |
| Width | 256 |
| Expert hidden width | 384 |
| Main blocks | 6 |
| Experts per block | 4 |
| Active experts | 1 (Top-1) |
| Context limit | 512 tokens |
| Kernel width | 10 |
| Dilations | 1, 2, 4, 8, 16, 32 |
| Shared refinement | 1–4 repeated passes |
| Production matrix-weight values | Q2-v2: -1.0, -0.5, +0.5, +1.0 (2 bits/symbol); Q2-v1 is legacy-read-only |
| Activation path | Per-token INT8 fake quantization |

Parameter accounting must be computed from the instantiated model. The convolution-only baseline has a finite receptive field determined by its dilation schedule. Hybrid profiles add causal compressed recurrent state without self-attention.

## 4. Capability targets

### 4.1 Language and conversation
- Understand ordinary written prompts and respond coherently.
- Follow explicit instructions and preserve relevant conversation context.
- Explain, summarize, rewrite, classify and transform text.
- Maintain consistent multi-turn dialogue.
- Support multilingual training and evaluation; language quality must be measured independently for each language.

### 4.2 Reasoning and problem solving
- Break structured problems into manageable steps.
- Compare options and explain the basis of a conclusion.
- Solve logic and multi-step tasks using verified training examples.
- Express uncertainty and correct mistakes when evidence requires it.
- Use evaluation tasks designed to detect memorized answers and data leakage.

### 4.3 Mathematics, science and engineering
- Arithmetic, algebra, geometry and introductory statistics.
- Physics, chemistry and general scientific explanations.
- Computing, electronics, hardware, operating systems, networks and engineering fundamentals.
- Checkable answers and transparent scoring rather than unverified generated explanations.

### 4.4 Programming and technical work
- Explain code and algorithms.
- Generate code from explicit specifications.
- Diagnose and explain common programming errors.
- Propose fixes, create tests and review code.
- Evaluate generated programs by running tests in a constrained sandbox; do not treat plausible-looking code as proof of correctness.

### 4.5 Text and document tasks
- Summarization, extraction, classification, question answering and transformation.
- Conversation-formatted datasets and structured Q/A records.
- Source-aware answers when source text is supplied.

### 4.6 Persistent memory and task tools

The current repository now includes an optional local SQLite memory store with FTS5 full-text retrieval (and a substring-search fallback on SQLite builds without FTS5). The memory subsystem remains separate from model weights and the neural forward graph. Writes occur only after an explicit /remember or qmini memory add action. Users can search, list, export and delete records.

Model-directed task execution is still a future extension. Any such system must be separate, auditable and permission-controlled:

- Long-term memory is opt-in and user-visible.
- Users can inspect, correct, export and delete saved records.
- The model may propose a registered action, but a host policy validates the schema and permissions.
- Consequential actions require user confirmation.
- No unrestricted command execution, hidden file changes or silent network/account actions.

## 5. Training-data design

Maintain data categories with source provenance and licenses:

1. General language and reading comprehension.
2. Educational, scientific and technical explanation.
3. Mathematics and logic with checked answers.
4. Programming, algorithms, debugging and test-driven examples.
5. High-quality instruction following and multi-turn conversation.
6. Multilingual and Indic-language examples where data quality is sufficient.
7. Safety, uncertainty and factual correction examples.

The pipeline should:
- Deduplicate and normalize data.
- Remove personal/private information and respect licenses.
- Split by document or source before tokenization to avoid train/validation leakage.
- Keep test data separate from training and tuning.
- Track corpus versions, language mix and tokens seen.
- Reject malformed records and report skipped files/rows.
- Preserve original datasets and checkpoints; never overwrite without explicit intent.

## 6. Training and checkpoints

Current implementation includes next-token cross-entropy, assistant-mask token-weighting across gradient-accumulation microbatches, router load-balancing auxiliary loss, AdamW, gradient accumulation, warmup/cosine scheduling, optional non-reentrant activation checkpointing, available GPU mixed precision, evaluation and fingerprinted checkpoint/resume. A separate --init-from final_model/model.pt path starts a fresh instruction fine-tuning stage.

Checkpoint metadata should remain sufficient to reproduce a run: model state, optimizer state, configuration, step number and RNG state. Resumption must reject a mismatched architecture/configuration. Logs should include train loss, validation loss/perplexity, learning rate, tokens seen, step and expert-load statistics.

## 7. Inference and deployment

- PyTorch reference inference is the correctness baseline.
- Packed Q2-v2 export is an inference artifact; it is not evidence that the entire model graph runs natively or uses integer arithmetic end to end.
- The C++17 projection primitive is wired into `QLinear` and the tied output projection for CPU evaluation/inference when the shared library is present; the remainder of the model graph still uses PyTorch.
- The CPU projection kernel now has an AArch64/NEON integer-vector path and scalar fallback; CI cross-compiles the AArch64 shared library. A complete Android ARM64/Termux runtime remains a later deliverable and must also implement embedding, causal convolution, normalization, expert dispatch, output projection and tokenizer compatibility.
- Native output must be compared to the reference implementation with stated numerical tolerances.
- Device tests must report peak RAM, model size, cold-start time, tokens/second, thermal behavior and long-session stability.
- No APK or Android UI is implied by this model repository.

## 8. Acceptance criteria by phase

### Phase A — Foundation correctness
- Architecture tests prove no attention/QKV components exist.
- Causality tests prove future tokens do not change earlier logits.
- Q2-v2 four-symbol packing/unpacking, scale/zero-row, legacy-v1 compatibility and STE gradient tests pass.
- Router gradients and per-expert assignment counts are verified.
- Data, tokenizer, checkpoint/resume and export tests pass.
- CI is green on the current source revision.

### Phase B — Training reproducibility
- Smoke training completes end to end.
- A resumed run continues the same optimizer step and rejects config mismatches.
- Validation data is document-separated and never silently reused for training.
- Logs and checkpoints are persisted outside ephemeral training runtimes.

### Phase C — Capability improvement
- Instruction and conversation evaluations are held out.
- Math/logic scoring uses verifiable answers.
- Code-generation scoring uses executable sandboxed tests.
- Per-language performance is measured instead of inferred from tokenization support.
- Improvements are compared against a fixed baseline and data split.

### Phase D — Low-bit inference
- Packed archive round-trip is verified for every tensor.
- Quantized generation is compared with reference generation.
- Native projection and full-model paths have explicit parity tests.
- Memory and speed are measured on target hardware.

### Phase E — Assistant extensions
- Local memory tests cover persistence, duplicate handling, search, export and explicit deletion; richer import/backup and policy controls remain future work.
- Tool schemas, allow-lists, error handling and confirmation gates are tested.
- Tool actions cannot run merely because generated text contains a command.

### Phase F — ARM64/Termux runtime
- Full graph runs without Python/PyTorch if that remains the chosen deployment goal.
- Correct tokenizer/config/model bundle loads.
- Differential tests pass on supported devices, including device-specific NEON vs scalar parity.
- Performance benchmarks report actual token throughput, memory and thermal stability on supported phones.
- Real-device resource and stability reports are published.

## 9. Non-claims

Q-Mini must not be described as:
- Already trained or already a capable general-purpose assistant.
- A trained 5B model or a claim that the current single-process trainer can distribute a 5B run.
- A Transformer, RWKV implementation or official Microsoft BitNet implementation.
- An unrestricted autonomous agent. The optional local text memory is not a task-execution agent.
- A complete integer-only or full ARM64 runtime.
- A system that has passed benchmarks unless results and test conditions are recorded.

The project is successful when measured capabilities, reproducible training and a reliable local runtime catch up with the stated goals—not when the feature list becomes longer.


## Capacity profile

The compact 8M-class profile remains available for smoke tests and low-resource experiments. The repository also includes a 134M-class GPU research profile (width 640, hidden width 1,024, ten main blocks, eight experts, 8,192 target vocabulary and 2,048 context). Its full-vocabulary parameter count is verified by runtime audit. It is a larger training experiment, not a claim of general-purpose capability before training and evaluation.


## Capacity and execution status

The repository now contains explicit 5M, hybrid 152M, sparse 5B and single-expert dense-reference profiles. The 5B profile contains **5,132,630,288 total parameters**, of which **852,873,488 unique parameters are active per token** under the current Top-1 routing policy. Its Q2 + FP32 row-scale + FP16-vector deployment layout is estimated at **1,294,831,200 bytes** (about 1.295 GB decimal). The planner estimates a **1,409,085,024-byte** model-level CPU working set for batch 1, including streaming caches and simplified temporary buffers. This is not measured process RSS and excludes the Python/PyTorch runtime baseline, allocator fragmentation, OS memory, tokenizer and decompression overlap. **4 GB physical-device support is not yet proven.**

The 5B trainer is not yet distributed-ready: FP32 parameters, gradients and AdamW moments alone have an approximately 82.1 GB lower bound before activations; FSDP/ZeRO and expert/tensor parallelism remain unimplemented. A packed CPU inference runtime exists and avoids allocating the dense 5B FP32 matrices; QLinear and tied output projections can use the C++ INT8-by-Q2 kernel. Convolution, normalization, embeddings, SSM and expert dispatch still execute through PyTorch. Real device speed and peak memory remain unmeasured.

The hybrid diagonal SSM is an adaptive, input-dependent decay recurrent path with a parallel training scan and constant-size streaming state. Streaming can continue beyond max_context without replaying the prompt; max_context bounds batch/prompt prefill, not total recurrent stream duration. This compressed state is not lossless transcript memory. Model weights do not update autonomously during chat: persistent SQLite notes are external retrieval context, and actual weight learning requires an explicit fine-tuning run.

The SSM scan, causality, packed runtime, stream cache bounds, data split integrity and training smoke paths are unit-tested. Capability improvement must still be demonstrated using a trained checkpoint and held-out conversation, reasoning, coding and multilingual evaluations. The 5B weight file size is not a proxy for assistant quality.


The single-expert dense profile bypasses router and router normalization because a one-element softmax is identically 1 and its parameters have no useful gradient. Periodic validation loss is accumulated by total supervised target-token count, with that denominator recorded in training metrics.


New production archives are Q2-v2 with normalized codebook, doubled-mean row scaling and exact-zero row handling. The normative format specification is docs/Q2_QUANTIZATION_SPEC.md; ternary/INT4/INT8 weight quantizers remain diagnostics only.


The normative production codebook, row-scale formula, packed metadata, native ABI and legacy decoder are specified in `docs/Q2_QUANTIZATION_SPEC.md`. Ternary, INT4 and rowwise INT8 weight functions are diagnostics only; they are not production export or inference formats.
