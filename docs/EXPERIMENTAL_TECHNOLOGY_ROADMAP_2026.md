# Q-Mini Experimental Technology Roadmap — 2026

**Status:** research and implementation plan; no new algorithm in this document is claimed to be implemented or validated.

## Product and safety constraints

- Keep exactly two model lines in this repository: the preserved legacy `qmini/` model and the independent `qmini_addsub/` model.
- The primary Add/Sub model is developed, trained, resumed and evaluated through Google Colab only. The active notebook uses CPU-only defaults and a separate Drive folder.
- Phone/Android/ARM deployment is out of scope. Historical phone prototypes are preserved separately in `experiments/phone/realme_c17/`; they are not imported by the model, Colab notebook or active Add/Sub CI.
- All active model experiments belong inside `qmini_addsub/`, behind versioned configs and checkpoint contracts. Do not modify legacy weights, checkpoints, tokenizer, trainer or runtime.
- Preserve the 760,340-parameter Add/Sub reference profile until a measured ablation justifies a new profile.
- Training may use floating-point arithmetic; the five-symbol QAT representation is an experimental training choice, not a claim of integer-only runtime.
- No external pretrained weights. English-first, provenance-checked data only.
- A research result on much larger models is a hypothesis for this 0.76M model, not evidence that the same gain transfers.

## What the current Add/Sub implementation already has

The reference architecture already includes causal depthwise convolution, an input-dependent gated diagonal state-space recurrence, sparse Top-1 SwiGLU experts, RMSNorm, small residual branch initialization, router load-balancing plus z-loss, and five-symbol QAT weights `{-1.5, -0.5, 0, +0.5, +1.5}`.

Do not add a second copy of these components. The five-symbol alphabet needs three fixed bits per weight; it is not a 1.58-bit ternary model. Current integer runtime coverage is limited to an additive projection primitive and bounded residual addition. Full convolution, recurrent state, gates, experts, normalization, activation LUTs and output decoding remain incomplete. No Add/Sub model has yet been trained on the reviewed English corpus.

## Colab learning algorithm — new experiment

The SFT stage now includes `qmini-self-paced-curriculum-v1`: uniform warm-up, per-example EMA response-loss difficulty ranking, gradual focus from the 55th to 80th difficulty percentile, and at least 35% uniform replay. Its state and schedule are checkpointed for exact resume. See [Learning Algorithm Design](LEARNING_ALGORITHM_DESIGN_2026.md).

This is inspired by SPaCe (Findings of ACL 2026), but is not a reproduction. It remains an experiment until compared against a uniform-sampling control under matched seeds, records, supervised tokens and optimizer steps. The sampler reads only train split losses and does not introduce model-generated training examples.

## Candidate technologies and fit

### 1. Surprise-gated external memory (highest product value)

**Research inspiration:** Google Titans and the MIRAS framework introduce neural long-term memory and test-time memorization, using a surprise signal to decide which incoming information should be retained. See https://research.google/pubs/titans-learning-to-memorize-at-test-time/ and the NeurIPS 2025 paper https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html.

**Q-Mini adaptation:** begin with a bounded, external memory sidecar rather than changing neural weights during a conversation. Store only explicitly useful, user-approved facts or recent task state. Score candidate memories by novelty, recency, confidence and relevance; retrieve a small top-k set. Provide clear inspect, edit, forget and reset operations. Keep per-user memory separate from model weights and training data.

**Why this fit:** it adds persistent context without increasing the 760K parameter count or requiring the model to relearn its weights for each user. It is safer and easier to validate than unguarded online gradient updates.

**Required tests:** memory leakage across users, false-memory insertion, contradiction handling, stale-fact expiry, prompt injection in retrieved text, reset/forget correctness, bounded storage and retrieval relevance. Never treat retrieved text as trusted instructions.

### 2. Selective-delta recurrent memory (architecture experiment)

**Research inspiration:** Mamba-3 (ICLR 2026) studies more expressive SSM recurrences, complex-valued state tracking and MIMO updates; Kimi Linear (October 2025) develops finer-grained per-channel forget gates through Kimi Delta Attention / Gated DeltaNet. References: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html and https://arxiv.org/abs/2510.26692.

**Q-Mini adaptation:** prototype a real-valued, per-channel selective-delta update as a drop-in replacement for the existing diagonal SSM block. The channel-wise forget gate should let different state channels retain or discard information at different rates; a delta correction should reduce overwriting when the new input is inconsistent with the current state. Keep the recurrence state fixed-size for token-by-token decoding.

Do not copy full Mamba-3 or Kimi Linear. Their published systems are tested at much larger scales and use more complex kernels. Do not introduce complex arithmetic in the first experiment: it complicates the Add/Sub-only runtime and is not justified until a real-valued variant shows a measurable gain.

**Required tests:** sequential recurrence vs parallel scan, finite gradients, no future-token leakage, batch-vs-streaming parity, state-cache size, long-range retrieval/state-tracking tasks, parameter count, CPU decode latency and activation-range calibration.

**Newer comparison worth testing:** Gated DeltaNet-2 (May 2026) separates channel-wise erase and write gates rather than forcing one gate to perform both jobs. Its paper reports results at 1.3B parameters trained on 100B tokens; this does not establish a gain at Q-Mini's 0.76M scale. Treat it as a low-rank/diagonal-inspired ablation, not a drop-in copy: preserve bounded state, measure added cache/state cost, and require identical data/token/prompt budgets. Reference: https://arxiv.org/abs/2605.22791.

### 3. Compact retrieval sidecar (high utility, no neural parameter growth)

Start with a deterministic BM25-style English retrieval index over user-approved local notes or reference documents. Return short passages with document identifiers and offsets so the assistant can ground answers and the UI can show sources. Keep retrieval optional and local-first. Do not add a separate large embedding model to the 0.7M core.

**Required tests:** answerable vs unanswerable queries, retrieval precision/recall, contradictory documents, source freshness, prompt injection resistance, storage size and CPU latency. The language model must be allowed to say that the evidence is insufficient.

### 4. Adaptive test-time compute (reason only when useful)

**Research inspiration:** ICLR 2025 work on compute-optimal test-time scaling finds that allocating inference compute according to problem difficulty can outperform uniform best-of-N budgets on selected reasoning tasks. See https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html.

**Q-Mini adaptation:** keep ordinary conversation on a single decode path. For selected tasks only, allow one bounded second pass or a small number of candidate solutions, then use a task-specific verifier: exact arithmetic checks for arithmetic, JSON parsing for structured output and sandboxed tests for code. Use calibrated uncertainty and a fixed compute budget. A low token probability alone is not a reliable correctness judge.

**Required tests:** compare single-pass vs adaptive pass on a fixed held-out set, record extra tokens/latency, verify that easy questions do not pay the extra cost, and reject any strategy that increases verbosity without improving correctness.

### 5. Optimizer experiments (training-only, not a default switch)

Recent work explores Muon-style orthogonalized momentum and adaptive variants such as NAMO/NAMO-D. One 2026 study evaluates those variants on GPT-2 pretraining; it does not prove they are better for this small CPU-trained, five-symbol-QAT model. Reference: https://arxiv.org/abs/2602.17080.

Keep AdamW as the control. Test a hybrid optimizer only on a short, reproducible run with identical data order, seed, token budget and schedule. Measure validation response loss, stability, wall-clock time and peak RAM. Promote it only after repeated wins. Do not increase the learning rate just to make training appear faster.

### 6. Quantization research (preserve the five-symbol baseline)

Microsoft BitNet demonstrates ternary 1.58-bit models and optimized CPU kernels, but its official model and reported speed results are at a vastly larger scale than Q-Mini. Reference: https://github.com/microsoft/BitNet.

Keep the required five-symbol Add/Sub format as the baseline. A ternary comparison may be run as a separately versioned quantization ablation only after the baseline has real trained weights. It must not silently change the five-symbol checkpoint format or be marketed as equivalent. Compare held-out model quality, packed size, quantization error, Colab CPU memory use and wall-clock training/evaluation cost.

## Recommended implementation order — Colab track

1. **Establish the real-data baseline:** complete one reproducible Colab CPU pretraining and SFT run; preserve data/tokenizer/run hashes and report held-out English loss, instruction performance, multi-turn continuity, repetition, factuality and reasoning slices.
2. **Ablate self-paced SFT:** compare the difficulty-aware sampler against `--curriculum-uniform-fraction 1.0` under matched seeds, data, supervised tokens and optimizer steps. Promote it only after repeatable held-out gains or matched quality at lower compute.
3. **Improve data efficiency:** assess source/topic balance, duplication and error buckets. A future topic-aware curriculum may be added only with a permanent uniform coverage floor.
4. **Run a selective-delta SSM ablation:** compare a small channel-wise erase/write-gate variant inspired by Gated DeltaNet-2 against the current diagonal SSM; replace the SSM for the experiment, do not stack another block on top. Keep the reference config untouched and compare identical data/tokens.
5. **Add bounded adaptive compute or audited memory:** only after the base model passes real task tests. Any added memory must be controlled, source-aware and not treated as trusted instructions.
6. **Compare optimizer and quantization variants:** keep AdamW and five-symbol QAT as controls. Test alternatives with matched seeds, tokens, step time and validation trajectories; do not change defaults based on research headlines.
7. **Phone deployment stays parked:** all phone-specific source and notes remain in `experiments/phone/realme_c17/`; they do not affect this Colab-only learning roadmap.

## Promotion gates

An experiment is not accepted because its paper is new or its unit tests pass. It must beat the current Add/Sub baseline on a pre-registered held-out evaluation, preserve multi-turn and language quality, report sample counts and confidence intervals, and disclose changes in parameter count, training tokens, optimizer steps and Colab wall-clock time. A candidate that improves validation loss but causes repetitive answers, worse factuality or loss of generality is rejected.

Until the first real-data Add/Sub run completes, architecture and curriculum changes should be treated as hypotheses. The existing CI smoke model is not a natural-language quality benchmark, and passing CI does not establish assistant capability.

## References

- Google Research, *Titans: Learning to Memorize at Test Time* (2025): https://research.google/pubs/titans-learning-to-memorize-at-test-time/
- NeurIPS 2025, *Titans: Learning to Memorize at Test Time*: https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html
- ICLR 2026, *Mamba-3: Improved Sequence Modeling using State Space Principles*: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html
- Kimi Team, *Kimi Linear: An Expressive, Efficient Attention Architecture* (2025): https://arxiv.org/abs/2510.26692
- NVIDIA et al., *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention* (2026): https://arxiv.org/abs/2605.22791
- ICLR 2025, *Scaling LLM Test-Time Compute Optimally*: https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html
- Zhang et al., *Adam Improves Muon* (2026): https://arxiv.org/abs/2602.17080
- Microsoft, BitNet inference framework: https://github.com/microsoft/BitNet
- Do et al., *SPaCe: Unlocking Sample-Efficient LLM Training With Self-Pace Curriculum Learning* (Findings ACL 2026): https://aclanthology.org/2026.findings-acl.171/
- Liu et al., *Ladders of Thought: A Self-Evolving Curriculum* (2026): https://arxiv.org/abs/2609.25643
- Akhtar et al., *When Does Muon Actually Win?* (2026): https://arxiv.org/abs/2610.21555
