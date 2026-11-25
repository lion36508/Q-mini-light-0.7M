# Evaluation protocol

Training loss and passing software tests are not proof that a model can answer real questions. Keep held-out data separate from training and tokenizer fitting.

## Legacy Q-Mini Light

Use the same trained checkpoint, tokenizer and fixed held-out token stream for before/after comparisons. Report token-weighted validation cross-entropy/perplexity, supervised-target count, task-level accuracy and multi-turn behavior. Compare Q2 reconstruction against its reference only on the same checkpoint.

## Q-Mini Add/Sub

The current evaluator verifies the dataset manifest, tokenizer hash and held-out SFT split before generation. It records every prediction/reference pair, generation failures, exact match, token overlap and output diversity. These are lexical diagnostics; they do not establish semantic correctness, factuality, reasoning, instruction adherence or safety.

A readiness report is a software/lexical gate, not a model certification. Review actual answers and independently score math, science, logic, coding and multi-turn continuity before making capability claims.

## Performance reports

Record device model, operating system, runtime, checkpoint/tokenizer hashes, batch and context lengths, prefill/decode tokens per second, peak RAM/VRAM, cold start and thermal behavior. Benchmarks using random token IDs or random weights must be labelled as synthetic microbenchmarks. Cross-compiling ARM64 code is not a real-device test.

Do not compare unrelated checkpoints, data mixtures, token budgets or hardware. Do not publish a speed, memory or quality improvement without the matching raw report and test conditions.

## Release gate

A release candidate needs a reproducible trained checkpoint, verified tokenizer and data lineage, independent held-out results, no unresolved critical test failures and documented limitations. General assistant quality cannot be inferred from parameter count, training loss, a synthetic learning proof or output diversity alone.
