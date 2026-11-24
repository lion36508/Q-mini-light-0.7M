# Q-Mini Add/Sub Learning Algorithm Design — Colab Track

**Scope:** this document defines the active Google Colab training system. Phone/Android/ARM deployment is explicitly excluded and preserved separately under `experiments/phone/realme_c17/`.

**Status:** the self-paced assistant-SFT sampler is implemented in `qmini_addsub/curriculum.py` and wired into SFT checkpoint format v8. The complete learning recipe is experimental; it has not yet been proven to improve a real-data run.

## 1. Learning is more than parameter count

Adding parameters or increasing the number of training steps does not guarantee better learning. Model capability depends on the architecture, the data distribution, optimization, memory/state updates, feedback quality, and how training allocates limited compute. Biological examples also cannot be reduced to neuron count: connectivity, specialized circuits, development, sensory inputs, learning dynamics, and evolutionary pressures all matter. The useful ML analogy is to improve the learning system, not simply count parameters.

A 760,340-parameter model still has a strict capacity limit. A better curriculum can help it use its capacity more efficiently, but it cannot guarantee human-level general intelligence or compensate for missing knowledge, insufficient data, or weak evaluation.

## 2. Existing Add/Sub architecture — preserve as the control

The baseline contains causal depthwise convolution, gated diagonal state-space recurrence, sparse Top-1 SwiGLU experts, RMSNorm, learned residual branch gates, router load-balancing and router z-loss, plus a five-symbol QAT weight alphabet. Do not stack another attention/SSM/expert system on top just because a newer paper exists. First measure the current model on real, held-out English data.

## 3. Stage A — broad language pretraining

Pretraining remains broad next-token learning over the reviewed English corpus. Sampling random windows across the training token stream is intentionally retained as the baseline so that rare language patterns and factual coverage are not accidentally dropped by a difficulty filter.

Required controls:
- Train/validation/test split by complete source document before chunking.
- Source, revision, license and SHA-256 recorded in a manifest.
- Train-only data used for gradient updates; validation is for checkpoint selection; test remains untouched until final evaluation.
- No model-generated text is labelled as human-authored.
- Preserve the existing pretraining optimizer and run contract as the baseline until controlled optimizer experiments show otherwise.

Potential future experiment: cluster article chunks by topic and difficulty, then schedule a bounded curriculum while retaining uniform document replay. Do not implement this before the first measured baseline because it risks narrowing factual coverage.

## 4. Stage B — self-paced assistant SFT

Implemented algorithm: `qmini-self-paced-curriculum-v2` (fixed-point replay quota).

### Per-example difficulty measurement

For each sampled conversation, compute the mean cross-entropy over assistant response tokens only. User prompt tokens and padding are masked from both the training target and the difficulty score. Keep an exponential moving average (EMA) per training conversation with decay 0.80.

### Warm-up

Start with uniform sampling for `max(1, min(100, floor(0.05 × planned_steps)))` optimizer steps. This gives the sampler an initial loss estimate rather than declaring examples easy or hard before the model has seen them.

### Adaptive focus

After warm-up, rank only records with an observed EMA response loss; records not yet sampled are never assigned a fabricated median difficulty. Gradually move the target difficulty from the 55th percentile to the 80th percentile over the planned SFT run. Use a smooth Gaussian focus band with bandwidth 0.18; this favours moderately difficult examples without repeatedly selecting only the highest-loss/noisiest records. Equal-loss ties receive average ranks so source-file order cannot act as a false difficulty signal.

### Uniform replay

A fractional quota is carried across optimizer steps so the requested 35% uniform replay is preserved over the run even when batch size is one; integer sample counts are allocated deterministically. Focused examples are drawn without replacement from records with measured EMA loss. If too few measured records are available, the remaining batch slots fall back to uniform sampling. Set `--curriculum-uniform-fraction 1.0` for a pure uniform-sampling control. The Colab notebook assigns distinct SFT/checkpoint/evaluation/export directories to the self-paced and uniform variants. Compare runs with identical base checkpoint, training records, seed, optimizer steps and supervised token budgets.

### Exact resume

Checkpoint SFT v7 stores:
- per-record EMA response losses and update counts;
- current curriculum step, schedule and hyperparameters;
- Python/Torch RNG and optimizer states;
- dataset, tokenizer, manifest and base-checkpoint hashes.

Changing the curriculum settings or the data creates a different run contract and cannot silently resume the old optimizer/sampler state. The sampler reads only training losses; it does not inspect test targets or use held-out examples for sampling.

### Why this is not reinforcement learning

This implementation does not update weights during generation, does not generate synthetic training records, and does not train on its own unverified answers. It changes which already-approved human-authored training examples are sampled. That avoids a feedback loop in which the model reinforces its own mistakes.

## 5. Research inspiration and adaptation limits

### SPaCe — self-paced curriculum learning

SPaCe (Findings of ACL 2026) clusters examples and adaptively allocates samples using learning progress. Its published results are research evidence for the value of data scheduling, not proof that the same gains transfer to Q-Mini. Q-Mini's current implementation is a simpler per-example response-loss sampler with a uniform replay floor, not a full reproduction of SPaCe.

Reference: https://aclanthology.org/2026.findings-acl.171/

### Ladders of Thought

LoT (September 2026) studies progressively simplified reasoning traces and a self-evolving curriculum on 1–8B models. It suggests testing progressive difficulty, but Q-Mini should first evaluate whether a much smaller, human-data-only curriculum helps. Do not import synthetic reasoning traces or teacher outputs into the approved corpus without separate provenance and review.

Reference: https://arxiv.org/abs/2609.25643

### Muon versus AdamW

Do not replace AdamW by reputation. A matched-budget October 2026 benchmark on 17.7M/33.6M parameter language models reported lower validation loss for Muon in its data-matched comparisons, but AdamW was faster per wall-clock budget for language in that implementation. Those models, hardware and kernels are not Q-Mini. Keep AdamW as the control and test any alternative with matched seeds, tokens, optimizer steps and wall-clock measurements.

Reference: https://arxiv.org/abs/2610.21555

### Test-time training and long-term memory

Titans and test-time-training ideas are interesting future research, but online weight updates during user conversations can cause catastrophic forgetting, privacy risks and hard-to-reproduce behavior. For the current Colab phase, keep learning in controlled offline training and checkpointed SFT. A bounded, auditable external memory can be researched separately after the base assistant quality is measured.

## 6. Evaluation protocol

Every curriculum experiment must be paired with a uniform-sampling control. Use the same:
- exact base checkpoint and model configuration;
- English training records, validation and held-out test hashes;
- tokenizer and sequence formatting;
- random seed set, optimizer settings, batch size, step count and supervised token budget;
- evaluation prompts, decoding settings and repetition checks.

Report response loss, held-out exact/semantic task scores, multi-turn continuity, factuality checks, reasoning-task accuracy, blank/identical/repeated-output rates, sample counts, wall-clock time and confidence intervals across multiple seeds where practical.

Promotion rule: the curriculum is the new default only if it repeatedly improves held-out assistant behavior or reaches comparable quality with measurably fewer supervised tokens/steps, without worsening factuality, generality, or repetition. Lower training loss by itself is not enough.

## 7. Out of scope for this phase

- Realme/Android deployment or phone-specific optimization.
- Native C++/ARM/NEON runtime integration.
- Full integer-only inference graph.
- Online learning from user conversations.
- Self-generated synthetic assistant corpus.
- Replacing AdamW or changing the model architecture without compute-matched ablations.

All phone-specific work remains separate under `experiments/phone/realme_c17/` and must not affect the Colab model or its learning configuration.
