# Q-Mini Light 0.7M — Attention-Free CausalConv + State-Space MoE Language Model

**Q-Mini is a from-scratch language-model research project aimed at becoming a compact, capable, offline-friendly text assistant.** It is not an Android app, a hosted-chatbot wrapper, or a fine-tune of downloaded pretrained weights. The model starts from random initialization. PyTorch supplies tensor operations, automatic differentiation and optimization; the tokenizer is trained locally from permitted corpus data.

> **Current status:** the repository contains an executable small-model baseline, training and data pipeline, reference text generation, packed-weight export utilities, a C++ projection primitive, tests and CI. It does **not** yet contain a trained, generally capable assistant, a complete packed-integer runtime, a finished Android app, persistent personal memory or unrestricted autonomous tool execution. This README separates implemented code from long-term goals.

[![Q-Mini CI](https://github.com/lion36508/Q-mini-light-0.7M/actions/workflows/tests.yml/badge.svg)](https://github.com/lion36508/Q-mini-light-0.7M/actions/workflows/tests.yml)

## Build the model first; train later

Open [Q-Mini Light 0.7M — Build Model First](https://colab.research.google.com/github/lion36508/Q-mini-light-0.7M/blob/main/colab/QMini_One_Click_Training.ipynb) in Google Colab and choose **Runtime → Run all**. This first notebook only prepares the model package: it mounts Drive, prepares the dataset/tokenizer, builds and verifies the initial Q2-v2 model, bundles the complete dormant training runtime and saves everything under `MyDrive/QMini-Light-0.7M/training_ready/`. It **does not start neural-model training**.

Before running, put reviewed corpus files in `MyDrive/QMini-Light-0.7M/datasets/`. The notebook stops if it finds no supported data. The Light model context is fixed at **512 tokens**. The initial package has zero optimizer steps and zero trained weight updates; it is a built, untrained model.

When you are ready to train, open [QMini Training Runner](https://colab.research.google.com/github/lion36508/Q-mini-light-0.7M/blob/main/colab/QMini_Training_Runner.ipynb) and explicitly set `START_TRAINING = True`. The runner loads the verified package and resumes from the newest valid checkpoint; it does not rebuild the initial model. A fresh Colab runtime may be used, but the model and training package remain saved in Drive. Training needs a compatible CUDA GPU with at least 8 GiB free memory; Colab hardware availability and session duration are not guaranteed.

## Build an untrained Q2-v2 initial brain

For the step-zero two-bit Q2-v2 build of the 0.7M Light model, use [the initial-brain Colab and CLI guide](docs/INITIAL_BRAIN_Q2_BUILD.md). Production matrix weights are exported as verified Q2-v2 (2 bits per symbol). The current QAT trainer still uses FP32 master parameters and optimizer state; this is a training-memory limitation, not a Q2 deployment artifact.

## Separate Light model build and training runner (advanced, two-notebook path)

The one-click notebook above is the recommended route. For manual control over package creation and training activation, use the two steps below.

Use [QMini Training-Ready Model Builder](https://colab.research.google.com/github/lion36508/Q-mini-light-0.7M/blob/main/colab/QMini_Training_Ready_Build.ipynb) to create the model package, tokenizer, prepared corpus and a **dormant bundled training runtime** in Google Drive. It never starts training.

Later, open [QMini Training Runner](https://colab.research.google.com/github/lion36508/Q-mini-light-0.7M/blob/main/colab/QMini_Training_Runner.ipynb) and explicitly set `START_TRAINING = True`. It resumes from the newest verified Drive checkpoint and updates a single progress record after each verified save. See [the lifecycle and recovery documentation](docs/COLAB_TRAINING_READY_BUILD.md).

## 1. Project vision

The long-term goal is to train Q-Mini into a useful general-purpose **text assistant** that can understand user instructions, maintain coherent conversations, explain difficult topics, solve structured problems, assist with programming and help users work through technical tasks.

Intended capability areas:

- **Conversation and language:** multi-turn dialogue, instruction following, context-aware answers, explanations, summaries, rewriting and question answering.
- **Reasoning and problem solving:** structured problem decomposition, logic, comparisons, planning and checking intermediate results.
- **Mathematics and science:** arithmetic, algebra, geometry, basic statistics, physics, chemistry, scientific concepts and engineering explanations.
- **Programming and software engineering:** code explanation, small program generation, debugging, algorithm review, test generation and development workflows.
- **Technical knowledge:** computers, operating systems, electronics, hardware, networks, software tools and engineering concepts.
- **Text and document work:** summarization, classification, extraction, transformation and answers grounded in supplied text.
- **Multilingual communication:** language coverage developed from verified training data, with English and Indian-language coverage as important evaluation targets. Fluency must be demonstrated by held-out tests, not assumed from tokenizer support.
- **Efficient local use:** progressively reduce inference cost and memory footprint so a trained model can run without a paid hosted inference API.

These are **training and product objectives**, not claims that the current random-initialized baseline already performs these tasks well.

## 2. Non-negotiable design constraints

1. **Text-only model.** Q-Mini is for text and language. Image understanding, image generation, vision encoders, video understanding and multimodal branches are outside this model's scope.
2. **No Transformer self-attention.** No Q/K/V projections, dot-product attention, softmax attention or retrieval-attention modules.
3. **Causal sequence processing.** A token at a future position must never influence an earlier position. The sequence mixer is a left-padded causal 1D depthwise convolution. This is a text-sequence operation, not an image CNN.
4. **From-scratch training.** Do not silently load or fine-tune third-party pretrained model weights. The tokenizer is trained locally from the selected corpus.
5. **Sparse expert computation.** Use learned Top-1 routing across the configured feed-forward experts in the current baseline; monitor expert usage rather than assuming specialization.
6. **Strict Q2 model weights.** Production matrix weights use exactly {-1.0, -0.5, +0.5, +1.0}, with two bits per weight. INT8 fake-quantized activations are separate. Clearly distinguish FP32 QAT master weights, Q2 packed storage and actual integer inference.
7. **Reproducible development.** Version architecture/configuration, test causal behavior, record data provenance, keep evaluation data held out and report actual parameter counts and runtime measurements.
8. **Free/open-source-first development.** Prefer open-source libraries and local execution. Do not make a paid hosted API or proprietary model service a required dependency.

## 3. Current architecture baseline

**Active Light profile:** 705,742 estimated parameters at a 4,096-token vocabulary. It retains causal depthwise convolution, the gated diagonal SSM, two-expert sparse Top-1 routing, shared refinement, Q2-v2 matrix quantization, INT8 activations, the training/checkpoint pipeline, evaluation and CI. The original 5B repository remains separate and is not selected or modified by the Light workflow.

The Light configuration reduces capacity while keeping the important model mechanisms. Larger configs in this repository are reference profiles only; the Light defaults select configs/qmini_light_0.7m.json.

| Component | Light profile |
|---|---:|
| Vocabulary target | 4,096 |
| Model width | 64 |
| Expert hidden width | 128 |
| Main causal blocks | 6 |
| Experts per block | 2 |
| Active experts per token | Top-1 |
| Maximum context | 512 tokens |
| Convolution kernel width | 10 |
| Convolution dilations | 1, 2, 4, 8, 16, 32 |
| Gated diagonal state-space | Enabled |
| Shared refinement | One block reused for 1–4 passes |
| Q2 normalized weight codebook | {-1.0, -0.5, +0.5, +1.0} |
| Activation quantization | Per-token AbsMax INT8 fake quantization |

### Forward path

~~~text
Token IDs
   -> Q2 token embedding
  -> Six Q-Mini causal blocks
       -> RMSNorm
       -> Quantized content/gate projection
       -> Left-padded depthwise causal convolution
       -> SiLU(content) * sigmoid(gate)
       -> Output projection + residual
       -> RMSNorm
       -> Sparse Top-1 expert router
       -> Selected expert MLP + residual
  -> Shared refinement block (1–4 iterations)
  -> Final RMSNorm
  -> Tied embedding projection
  -> Next-token logits
~~~

With kernel width 10 and the configured dilations, the six convolution blocks have a theoretical receptive field of 568 tokens. The Light profile uses a 512-token context, with the gated diagonal state-space path providing the recurrent long-range sequence state. Parameter count must always be read from the instantiated model using the sum of parameter tensor sizes; configuration changes alter the total.


### Q-Mini Light 0.7M training profile

The active Colab profile is `configs/qmini_light_0.7m.json`. At the configured 4,096-token vocabulary, the model has **705,742 estimated parameters** and retains the same architecture family as the main Q-Mini project.

| Setting | Light profile |
|---|---:|
| Vocabulary target | 4,096 |
| Model width | 64 |
| Expert hidden width | 128 |
| Main causal blocks | 6 |
| Experts per block | 2 |
| Active experts | Top-1 |
| Maximum context | 512 tokens |
| Convolution kernel | 10 |
| Dilations | 1, 2, 4, 8, 16, 32 |
| Shared refinement | 1–4 passes |
| Gated diagonal state-space | Enabled |
| Q2 normalized weight codebook | {-1.0, -0.5, +0.5, +1.0} |

The model is intentionally small for safer iteration and local/Colab experiments. The larger 5M, 134M, 152M and 5B configuration files remain available as copied reference profiles; the Light Colab workflow does not select them.

### Quantization — fixed production Q2-v2 format

- Production matrix weights use exactly the four normalized symbols {-1.0, -0.5, +0.5, +1.0}; each symbol is stored in exactly two bits.
- Each output row uses FP32 scale = 2 × mean(abs(original row)); effective weight = normalized symbol × row scale. A fully zero row has scale 0 and remains exactly zero. In a nonzero row, an exact-zero source entry maps to +0.5 × row scale because the production codebook has no zero symbol.
- QAT keeps FP32 master parameters and optimizer state with an STE. This is training state, not the deployed matrix-weight format. New packed exports store matrix symbols at two bits each, plus separately stored FP32 row scales and FP16 one-dimensional vectors.
- Activations use per-token AbsMax signed INT8 fake quantization. INT8 activations do not widen matrix weights.
- New archives use Q2-v2. Legacy Q2-v1 archives decode with their original symbol and scale semantics; contradictory Q2-v2 codebook metadata is rejected.
- The native C++ projection is a versioned INT8-by-Q2 kernel. It is a projection primitive, not a complete native model graph.
- Causal depthwise convolution, RMSNorm, embedding lookup and sparse dispatch still include PyTorch operations; the packed runtime is hybrid, not fully integer-only.

## 4. Implemented now vs. intended later

| Area | Implemented in this repository | Longer-term objective |
|---|---|---|
| Language model | From-scratch causal-convolution MoE baseline | Improve language quality through measured training |
| Conversation | Reference text generation and terminal multi-turn chat | Reliable, context-aware assistant dialogue |
| Reasoning | Shared latent refinement mechanism and next-token training | Demonstrated math, logic and multi-step problem-solving accuracy |
| Coding | General text-generation path can generate code | Dedicated code data, debugging evaluations and executable test-based scoring |
| Data | Local BPE tokenizer and text/chat data readers | Larger curated, deduplicated, licensed and balanced corpora |
| Training | AdamW, gradient accumulation, available mixed precision, assistant-only loss masks for structured chat/Q&A, evaluation and fingerprinted resumable checkpoints | Longer curriculum training and measured instruction tuning |
| Experts | Sparse Top-1 MoE and router/load statistics | Verified useful specialization without expert collapse |
| Quantization | Q2-v2 four-level matrix weights, INT8 activations and packed export | Measured Q2 quality, speed and memory on trained checkpoints/device |
| Memory | Opt-in local SQLite memory with FTS5 search, explicit save/recall/delete and JSON export | Improve retrieval and add user-controlled import/backup workflows |
| Tools/tasks | CLI for project operations; no model-controlled general tool executor | Permission-gated registered tools for carefully scoped tasks |
| Mobile | Standalone C++ projection primitive | Complete ARM64 runtime, integrated kernels and measured on-device inference |
| Multilingual | Tokenizer and text pipeline support Unicode text | Language-specific training and held-out language evaluations |

A feature in the right-hand column is a roadmap item, not an existing capability.

## 5. Training and learning plan

A chat-only corpus is insufficient for the complete target. Training data should be versioned by source, license, language, quality and task family.

### Suggested curriculum

1. **Language foundation:** clean, deduplicated, well-formed text with known provenance and permitted licenses.
2. **Technical and educational material:** mathematics, logic, science, programming, engineering, documentation and explanations.
3. **Instruction and conversation:** role-marked multi-turn examples, clear task instructions, context retention and grounded responses.
4. **Reasoning practice:** verified arithmetic, algebra, logic, structured problem solving and worked explanations with checked answers.
5. **Programming practice:** code explanation, algorithm design, debugging examples, test generation and code-review cases with verified expected outcomes.
6. **Safety and calibration:** uncertainty, correction, safe boundaries, refusal examples and instruction hierarchy.
7. **Mixed replay:** continue mixing core language examples with specialist skills to reduce catastrophic forgetting.

Keep validation documents separate from training documents. Do not treat unverified synthetic answers as ground truth. Respect dataset licenses and remove private or sensitive data.

### Training pipeline

- Discover supported local data files: TXT, Markdown, JSON, JSONL/NDJSON, CSV and optional Parquet.
- Split documents deterministically and de-duplicate exact documents using disk-backed indexes.
- Train the local byte-level BPE tokenizer on training documents only; validation documents are excluded from tokenizer vocabulary learning.
- Convert documents into role-marked text where appropriate and create uint16 token streams with aligned uint8 loss masks.
- In `auto` mode, plain text receives ordinary next-token loss while structured chat/Q&A applies supervised loss to assistant response spans and leaves user/system text as context.
- Run a tiny smoke training before the full run.
- Train next-token prediction with cross-entropy and router load-balancing loss.
- Use AdamW, gradient accumulation, warmup/cosine scheduling and available GPU mixed precision.
- Save model, optimizer, configuration and RNG state so a compatible run can resume.
- Evaluate on held-out data and retain JSONL training metrics.

## 6. Evaluation: capability must be measured

A bigger model, more experts or additional refinement steps do not automatically make a better assistant. Record:

- Validation cross-entropy and perplexity, tokens seen and training steps.
- Arithmetic/math exact-match and numeric-answer accuracy.
- Logic and multi-step task accuracy using held-out problems.
- Instruction adherence and multi-turn conversation consistency.
- Language-wise performance and invalid/repetitive-output rates.
- Code generation pass rate on sandboxed unit tests, not just subjective examples.
- Router token counts per expert and signs of starvation/collapse.
- Tokens per second, peak RAM/VRAM, cold-start time, model artifact size and thermal stability.
- Quantized-vs-reference numerical error and generation-quality change.

Use fixed datasets, prompts and scoring methods when comparing revisions. Report sample counts and limitations; do not claim benchmark results from training data or cherry-picked samples.

## 7. Memory and task execution

**Local memory is implemented as an optional subsystem outside the neural model.** It uses SQLite for persistence and FTS5 full-text search when the local SQLite build provides it; a literal-substring fallback is available when FTS5 is absent. Memory is never written automatically.

Start chat with `python -m qmini.cli chat --model-dir PATH --memory-db PATH_TO_LOCAL_SQLITE`. The chat supports `/remember TEXT`, `/recall QUERY`, `/memories` and `/forget ID`. Relevant saved notes may be surfaced as explicit memory context during a turn. The CLI also provides `python -m qmini.cli memory --db PATH add --text TEXT`, `search --query QUERY`, `forget --id ID` and `export --output BACKUP.json`. Records can be individually deleted or exported to JSON. SQLite database files are ignored by Git.

Model-directed task execution remains a future, permission-controlled extension. It should use typed tool schemas and a small allow-list; consequential actions must require confirmation. No unrestricted shell, filesystem, account or network actions should occur merely because the model generated a command.

## 8. Hardware and deployment direction

**Development and training:** GitHub is the source of truth for code, configuration, tests and documentation. Google Colab is the GPU training environment. Google Drive is the recommended location for datasets, tokenizer artifacts, token streams, checkpoints, logs and model exports.

**Local CPU inference:** the hybrid path uses the versioned C++17 packed-Q2-v2 × INT8 projection kernel when a compatible shared library is available; AArch64/NEON builds use vector widening multiply and integer accumulation, while other architectures use the scalar integer fallback. The convolution, RMSNorm, embedding lookup and sparse dispatch remain PyTorch. The benchmark script reports the selected projection backend; performance must be measured on each target phone.

**Android ARM64 / Termux:** the current native projection source has an AArch64/NEON compile path and CI cross-compiles it. A complete offline runtime still requires native embedding lookup, causal depthwise convolution, RMSNorm, sparse expert dispatch, output handling and tokenizer integration, plus differential tests and real-device benchmarks. CI cross-compilation is not an Android phone performance benchmark. This repository does not currently build an APK or complete Android runtime.

## 9. Roadmap

- **Phase A — Core correctness:** architecture invariants, causal tests, parameter audit, tokenizer/data validation, smoke training and safe checkpoint resume.
- **Phase B — Training quality:** curated language/technical corpora, longer runs, held-out validation, router diagnostics and repeatable checkpoints.
- **Phase C — Assistant competence:** instruction and multi-turn data, mathematics/logic, science and programming curricula, capability-specific evaluations and calibration.
- **Phase D — Efficient inference:** packed-format validation, quantized/reference comparisons, optimized kernels and measured CPU performance.
- **Phase E — Local assistant extensions:** optional user-controlled memory and permission-gated tools, only after core model behavior is reliable.
- **Phase F — ARM64 deployment:** complete native graph, differential tests, performance and memory measurements on supported phones.

Each phase should be accepted using tests and measurements, not by adding a feature name to the README.

## 10. Quick start

### Local environment

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python QMINI_MASTER.py verify
python -m qmini.cli audit --config configs/qmini_smoke.json
~~~

### Google Colab

1. Put permitted training files in Google Drive/MyDrive/QMini/datasets/.
2. Open colab/QMini_Training.ipynb and select an available GPU runtime.
3. Run the notebook from top to bottom. It mounts Drive, obtains the source, installs dependencies, trains the tokenizer, prepares data and runs a small smoke training before the base run.
4. Keep artifacts in Google Drive/MyDrive/QMini/artifacts/ so they survive Colab runtime resets.

The notebook and CLI contain the tokenizer, data preparation, training, resume, generation, chat and held-out evaluation commands. The training step count is the target total optimizer step, not an extra number of steps after resume.

## 11. Repository map

- QMINI_MASTER.py — single project entry point.
- qmini/model.py — causal-convolution MoE model.
- qmini/quantization.py — Q2-v2 four-level STE fake quantization and two-bit packing; alternate quantizers are diagnostic-only.
- qmini/tokenizer.py, qmini/data.py — local tokenizer and corpus pipeline.
- qmini/train.py — training, validation and resumable checkpoints.
- qmini/generate.py — cached autoregressive generation and terminal chat.
- qmini/evaluate.py — held-out language-model and prompt/reference evaluation.
- qmini/export.py — packed archive export and verification.
- qmini/memory.py — opt-in local SQLite/FTS5 memory with explicit CRUD and JSON export.
- qmini/native_kernel.cpp, qmini/native.py — packed INT8-by-Q2 CPU projection, AArch64/NEON path, scalar fallback and versioned Q2 ABI detection.
- scripts/benchmark.py — repeatable full-forward, streaming prefill and cached autoregressive decode benchmark.
- configs/ — compact base, 5M, historical 134M, hybrid 0.7M, 5B capacity and smoke configurations.
- tests/ — model, causality, quantization, data, training, export and native tests.
- docs/ — architecture, training, evaluation and project scope.
- colab/ — Colab training notebook.
- .github/workflows/ — automated CI.

## 12. Important limitations and data security

- A random initialization is not a useful assistant. Real capability requires enough high-quality training data, compute and independent evaluation.
- The current baseline is a compact research model, not a large foundation model or a claim of state-of-the-art performance.
- FP32 master weights and optimizer states are used in training.
- The packed archive now has a low-memory CPU inference path; it is a hybrid runtime, not a fully integer-only graph.
- The native kernel accelerates dense CPU linear projections; convolution, normalization, embedding lookup and MoE dispatch still use PyTorch.
- There are no bundled trained weights or private datasets.
- Local memory is stored in the path supplied by the user; protect that database as personal data and export/delete records deliberately.
- The GitHub repository is public. Never commit credentials, private datasets, personal information or private checkpoints. Keep such artifacts in private storage.

**Project principle:** build a truthful, testable, efficient text model first; add assistant capabilities only when the training data, implementation and evaluations support them.


## 13. Updated attention-free hybrid architecture

The new hybrid model path is:

```text
Token IDs -> Q2 Embedding
  -> [RMSNorm -> Gated Causal Depthwise Conv -> Residual
      -> RMSNorm -> Gated Diagonal State-Space Memory -> Residual
      -> RMSNorm -> Sparse Top-1 MoE -> Residual] x N
  -> Shared Refinement Block
  -> RMSNorm -> Tied Language-Model Head
```

No Transformer self-attention, Q/K/V projections or dot-product attention are introduced. The state-space layer maintains a learned fixed-width compressed state across prior tokens. It is not an exact transcript store or arbitrary content-addressable retrieval. Its parallel training scan and recurrent streaming update are tested against each other. This is Q-Mini's own diagonal SSM implementation, not an official Mamba or RWKV implementation.

| Profile | Total parameters | Unique Top-1 active parameters/token | Purpose |
|---|---:|---:|---|
| `qmini_light_0.7m.json` | **705,742** | **589,710** | Active 0.7M Light default |
| `qmini_5m.json` | 5,245,996 | 3,070,636 | Retained reference profile |
| `qmini_research_134m.json` | 134,426,200 | 33,372,632 | Historical convolution-only profile |
| `qmini_research_152m.json` | 152,483,800 | 51,430,232 | Larger hybrid reference profile |
| `qmini_5b.json` | 5,132,630,288 | 852,873,488 | Capacity design; not selected by Light workflows |

The CLI can inspect a config without allocating the model:

```bash
python -m qmini.cli plan --config configs/qmini_light_0.7m.json\npython -m qmini.cli plan --config configs/qmini_5m.json
python -m qmini.cli plan --config configs/qmini_light_0.7m.json
python -m qmini.cli plan --config configs/qmini_5b.json
```

FP32 parameter/gradient/AdamW-state lower bounds are approximately 11.3 MB for Light 0.7M, 83.9 MB for 5M, 2.44 GB for 152M and 82.1 GB for 5B. These figures exclude activation memory, CUDA workspaces, fragmentation and communication. The 5B profile is not a single-H100 training recipe: this repository still has no FSDP/ZeRO, expert parallelism or tensor parallelism.

## 14. Training budget and staged instruction fine-tuning

The Colab notebook uses a configurable initial token budget based on 20 tokens per unique active Top-1 parameter. This is an experimental compute-budget heuristic, not an established sparse-MoE scaling law. It prints the approximate number of passes over the available training-token stream so a small corpus cannot silently masquerade as billions of unique examples.

The training loop aggregates assistant-response loss by supervised target-token count across the complete gradient-accumulation batch. This prevents a short assistant answer and a long assistant answer from receiving equal microbatch weight solely because they happened to land in different microbatches.

A new instruction/SFT run should initialize from the packed `final_model/model.qmini.npz` using `--init-from`. The packed archive is streamed into trainable weights tensor-by-tensor; the fine-tuning stage gets a fresh optimizer. Legacy `model.pt` inputs remain readable when explicitly supplied, but new final-model exports no longer create them. The source model hash/path is saved in the run manifest and verified when resuming the SFT run. Use an assistant-only loss mask for structured conversations.

These additions improve architectural coverage and training correctness. They do not prove natural conversational ability. That still requires real training, held-out multi-turn evaluation and measured improvements.


## 15. 5B deployment size and CPU inference

The 5B profile contains 5,132,630,288 parameters. Its exact row-padded Q2 + FP32 row-scale + FP16 vector layout is estimated at **1,294,831,200 bytes** (about 1.295 GB decimal / 1.206 GiB), below the 1.5 GB model-weight artifact budget. This is a pre-training layout estimate; the actual exported archive is measured and written to \`model_manifest.json\`. Tokenizer and application files are additional.

New final-model folders now contain the verified `model.qmini.npz` deployment export and do not generate a dense FP32 `model.pt` snapshot, even for small models. Existing legacy `model.pt` files are left untouched for compatibility. The resumable `latest.pt` checkpoint remains outside `final_model/` and still contains FP32 QAT master parameters and AdamW state; BF16/FP16 autocast does not change that storage. Thus production matrix weights are strictly Q2-v2, but the current training process is not entirely two-bit. A later all-low-bit training path needs its own optimizer, gradient and quality validation. The default loader uses the packed CPU runtime without creating dense matrix tensors. New Q2-only manifests prevent any leftover legacy `model.pt` from overriding the packed model. Explicit CUDA/MPS requests are rejected when only a Q2 archive exists, because the current packed inference backend is CPU-only; it does not silently expand the model into dense FP32 weights. A compiled C++ kernel is recommended for speed:

\`\`\`bash
python -m qmini.native --build
python -m qmini.cli generate --model-dir /path/to/final_model --prompt "Explain photosynthesis."
\`\`\`

The PyTorch fallback still works without the C++ library, decoding one projection matrix at a time and the vocabulary head in bounded chunks, but it is slower. Packed exports are inference-only; \`--init-from model.qmini.npz\` supports a later training stage by dequantizing each tensor into a separately allocated trainable model. The resumable \`latest.pt\` training checkpoint remains outside the final deployment folder.

The 2-bit representation is aggressive quantization. Its file size does not establish model quality. Compare quantized output against held-out language, maths, logic, instruction-following and multi-turn tests. CPU execution is implemented, but per-device tokens/second, memory peak and thermal behavior require an actual device benchmark.

Q-Mini does not update its neural weights autonomously during chat. Optional SQLite notes are retrieved as external context; they are not neural long-term memory. To incorporate new examples into model weights, export/prepare a curated dataset and run a separate fine-tuning stage with `--init-from`, then evaluate the new checkpoint before deployment.


## 16. 4GB-device planning versus measured support

The 5B configuration is **5,132,630,288 total parameters** with **852,873,488 unique Top-1 active parameters per token**. A single-expert dense reference, \`configs/qmini_dense_853m.json\`, has **852,281,344 parameters**, giving a comparable active-parameter reference for controlled quality/latency comparisons. This does not prove that sparse MoE is better; both profiles need the same data, compute budget and held-out evaluation.

The current plan estimates:
- Packed 5B model-weight artifact: **1,294,831,200 bytes**.
- Persistent CPU stream caches, batch 1: **80,527,360 bytes**.
- Conservative model-level CPU working set: **1,409,085,024 bytes**.

The working-set estimate excludes the Python/PyTorch baseline, allocator fragmentation, OS memory, tokenizer, archive-decompression overlap, and any app process. It is not a measured peak-RSS number. Therefore Q-Mini has a low-file-size packed format and a CPU code path, but **4GB phone support is not yet proven**. Run the trained model benchmark on the exact device and record actual RSS, time-to-first-token, decode tokens/sec, thermals and quality.

Near-duplicate data leakage can be reduced with the disk-backed SimHash grouping option. Use the **same** value when training the tokenizer and preparing token streams:

\`\`\`bash
python -m qmini.cli tokenizer --help
python -m qmini.cli prepare --help
\`\`\`

The default \`--near-duplicate-hamming-distance 0\` keeps exact-deduplication behavior. Values 1–3 group similar word-trigram SimHashes before assigning whole groups to train/validation; the grouping threshold is recorded in \`dataset.json\`.


## Streaming context semantics

\`max_context\` is the maximum sequence length accepted by a single batch forward or initial prompt prefill. It is not a hard cap on the total number of tokens processed by an incremental stream. After prefill, the convolution caches remain fixed-size and the diagonal SSM retains a fixed-width learned state, so generation can continue without replaying the entire prompt after every token. This recurrent state is compressed memory, not exact access to every previous token. It therefore may preserve useful long-range information, but exact recall of arbitrary old details is not guaranteed.



## Colab training and SFT stages

The notebook now defaults to a small, reproducible learning-proof run: fixed-seed held-out NLL is measured before and after training. The 0.7M long run requires explicit \`RUN_LONG_TRAINING = True\` opt-in and has no arbitrary 2,000-step cap. Set LONG_OPTIMIZER_STEPS_OVERRIDE to choose the total optimizer-step target, or leave it None to use the documented token-budget heuristic. Increasing the target reuses the same Drive checkpoint and extends the saved learning-rate schedule. Optional assistant SFT is also disabled by default; when enabled, it uses a separate licensed dataset, response-only masks, a tokenizer-bound Q-Mini checkpoint and fresh optimizer state. The notebook never downloads pretrained weights. A small learning-proof or smoke run is not evidence of general assistant quality.

## Experimental quantization ablation

`scripts/quantization_ablation.py` compares Q2-v2 reconstruction against an FP16 reference and optional rowwise INT8, ternary and groupwise INT4 diagnostic references using only Q-Mini-owned weights/checkpoints. Those alternate formats are measurement baselines only; they are not production model, export or inference formats. The script measures weight reconstruction error, not held-out NLL, assistant quality, latency or peak RAM.

## Production Q2-v2 format contract

New packed model exports use **Q2-v2**: normalized matrix symbols {-1.0, -0.5, +0.5, +1.0} with codes 00, 01, 10, 11 respectively and an FP32 row scale equal to 2 * mean(abs(original_output_row)). An all-zero row uses scale zero and reconstructs exactly to zero; exact zeros within a non-zero row map to the positive half-symbol because the four-symbol production alphabet has no zero code. Activations remain INT8, while biases and one-dimensional metadata vectors remain FP16 in packed archives.

Existing Q2-v1 archives ({-2,-1,+1,+2} with legacy scales) remain readable. New exports never write v1. The native C++ projection uses a versioned Q2-v2 ABI and applies the same decode/scaling rule as the Python fallback. See docs/Q2_QUANTIZATION_SPEC.md for the normative contract and tests.


For a small Q-Mini-owned checkpoint, the quantization ablation can additionally measure held-out NLL/perplexity using `--eval-tokens PATH --eval-seq-len 128 --eval-batches 100 --eval-seed 42`. For response-only evaluation add `--eval-mask PATH`. All candidate formats use the same validation stream and INT8 activation path. The default 10M-parameter evaluation cap bounds memory use; weight-error-only reports remain available for larger checkpoints.



### Quantization ablation provenance and reporting

The quantization ablation tool now groups tensor reconstruction metrics by embedding, output projection, router, expert, convolution, state-space and other components. For each Q2 matrix it records the row-scale distribution (including zero-row count) and separate packed-symbol, FP32-scale and FP16-vector payload bytes. When the final model directory contains model.qmini.npz, it also records the exact archive byte size.

The quantization ablation needs the original trainable reference weights to compare Q2 against FP16, INT8 and diagnostic formats. For new runs, pass the run's own `latest.pt` using `--checkpoint`; the final `model.qmini.npz` is already quantized and cannot act as an unquantized reference. Legacy `model.pt` snapshots remain accepted only with a matching `model_manifest.json` and packed-export fingerprints.

Assistant fine-tuning is fail-closed: the notebook will not silently use the tiny smoke checkpoint as its source and requires independent held-out task and multi-turn conversation evaluation data before starting SFT.
