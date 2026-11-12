# Evaluation protocol

Training loss is a learning signal, not proof that Q-Mini can answer real user questions. Keep a versioned, held-out evaluation set that is never mixed into training or tokenizer fitting.

## Language-model evaluation

The standalone evaluator reports token-weighted cross-entropy and perplexity, supervised target count, input throughput, peak CUDA allocation and per-block expert token loads.

~~~bash
python -m qmini.cli eval-lm \
  --model-dir /path/to/final_model \
  --tokens /path/to/held-out/valid.bin \
  --mask /path/to/held-out/valid.mask.bin \
  --batch-size 2 --seq-len 512 --batches 100 --seed 42 \
  --output /path/to/reports/language-eval.json
~~~

Use evaluation streams produced by the same tokenizer and configuration as the model. Keep the document split and tokenizer training split identical. Compare runs with the same data hashes, seed, batch/sequence settings and model configuration.

## Task evaluation

Prepare a UTF-8 JSONL file with one JSON object per prompt. Each row must contain a non-empty prompt and one of reference, expected or answer; category is optional.

~~~json
{"category":"math","prompt":"What is 6 times 7?","reference":"42"}
{"category":"science","prompt":"What gas do plants absorb?","reference":"Carbon dioxide"}
{"category":"coding","prompt":"What does a Python list store?","reference":"An ordered collection of items"}
~~~

Run deterministic greedy evaluation.

~~~bash
python -m qmini.cli eval-tasks \
  --model-dir /path/to/final_model \
  --dataset /path/to/evaluation-tasks.jsonl \
  --max-new-tokens 96 --output /path/to/reports/task-eval.json
~~~

The report contains exact-match and lexical token F1 overall and by category, plus every prompt/reference/prediction. These are transparent string-comparison diagnostics. They do not establish semantic correctness, factuality, instruction adherence or code correctness. Mathematics requires verified numeric scoring; code generation requires execution in a constrained test sandbox; reasoning requires independently checked multi-step tasks.

## Required capability reporting

Track:
- Validation cross-entropy and perplexity, tokens seen and training steps.
- Arithmetic/math exact-match and numeric-answer accuracy.
- Logic and multi-step task accuracy using held-out problems.
- Instruction adherence and multi-turn conversation consistency.
- Language-wise performance and invalid/repetitive-output rates.
- Code generation pass rate on sandboxed unit tests, not subjective examples.
- Router token counts per expert and signs of starvation/collapse.
- Tokens per second, peak RAM/VRAM, cold-start time, model artifact size and thermal stability.
- Quantized-vs-reference numerical error and generation-quality change.

Use fixed datasets, prompts and scoring methods when comparing revisions. Report sample counts and limitations; do not claim benchmark results from training data or cherry-picked samples.


## Hybrid architecture tests

The gated diagonal state-space path is accepted only when the parallel scan agrees with the sequential recurrence, gradients are finite, future-token changes cannot affect earlier logits, and incremental streaming logits match batch logits within the configured prefill length. Longer streaming tests must additionally demonstrate fixed-size convolution/SSM caches and no prompt replay after max_context. These tests establish implementation correctness, not conversational competence.

A capability report must additionally include fixed held-out tests for multi-turn continuity, instruction adherence, Assamese/Hindi/English performance, arithmetic and independently verified maths, science explanation, logic/state tracking, code-generation unit tests in a constrained runner, long-context recall and repetition/termination failures. Exact match and lexical token F1 are diagnostic metrics, not semantic reasoning scores.


## Quantized deployment and sparse/dense comparison

For the 5B design, compare the sparse profile against \`configs/qmini_dense_853m.json\` under a controlled compute budget. Keep tokenizer, train/validation documents, tokens seen, optimizer schedule and evaluation prompts identical. Report:
- Validation NLL/perplexity and response-mask target count.
- Multi-turn consistency, factuality checks, maths/logic accuracy, and sandboxed coding pass rate.
- Per-expert load distribution, starvation/collapse, active compute and total resident model size.
- Q2 packed versus dense/fake-quant output differences, including capability regression by task.
- Prefill throughput, autoregressive decode latency/tokens/sec, peak process RSS, cold start and thermal behavior.

The planner's 1.409 GB CPU working-set estimate for batch 1 is not a measured process RSS and does not establish a 4 GB-phone target. A release claim requires measurements on the actual device with the trained packed archive and tokenizer loaded.


## Metric aggregation

Report token-weighted negative log-likelihood, not an arithmetic mean of batch-level means. For masked conversation data, sum each supervised response token's cross-entropy and divide by the total supervised response-token count across the entire evaluation stream. Log that count alongside validation loss and perplexity. The periodic training evaluator and standalone \`eval-lm\` use this same definition.



## Structured task-file scoring

The JSONL task evaluator accepts an optional \`language\`, \`scorer\` and \`tolerance\` per example. Supported scorer values are:
- \`lexical\`: normalized Unicode exact match;
- \`numeric\`: exactly one finite numeric answer, compared with absolute tolerance;
- \`json\`: parsed JSON values must be structurally equal;
- \`contains\`: normalized reference text must occur in the prediction.

Reports include per-category and per-language task-pass rates, Wilson 95% confidence intervals, lexical diagnostics, and case-level scoring errors. These are useful objective checks for well-specified tasks; they do not by themselves establish factuality, safety, general reasoning or code quality. Code-generation claims still require a separate constrained test harness with independently reviewed test cases.




## Weight-format reconstruction experiment

Run \`python scripts/quantization_ablation.py --model-dir /path/to/final_model --output reports/quantization.json\` for a small final model, or \`python scripts/quantization_ablation.py --checkpoint /path/to/run/latest.pt --output reports/quantization.json\` for a training checkpoint. It compares Q2-v2 reconstruction against an FP16 reference and optional rowwise INT8, zero-preserving ternary and groupwise INT4 diagnostic references. These alternate formats are measurement baselines only, not production model/export/runtime formats. The report contains overall and per-tensor relative L2 error, cosine similarity, maximum absolute error and estimated weight payload bytes.

This script measures reconstruction error only. It does not measure model NLL, assistant capability, latency or peak RAM, and it does not change the strict Q2-v2 production exporter. No wider or alternate format may replace Q2 as the deployed model. Quality/runtime comparisons require the same Q-Mini-trained checkpoint and a fixed held-out evaluation set.




## Multi-turn conversation evaluation

Use \`eval-conversations\` with a JSONL file containing one conversation per line. Each record has an optional \`id\`, \`language\`, \`category\`, \`system\` prompt, and a non-empty \`turns\` array. Every turn contains \`user\`, \`reference\` (or \`expected\`/\`answer\`), and optional \`scorer\`, \`tolerance\`, \`language\` and \`category\`.

Example:

\`\`\`json
{
  "id": "continuity-001",
  "language": "en",
  "category": "multi_turn_continuity",
  "system": "Answer accurately and concisely.",
  "turns": [
    {"user": "What is six times seven?", "reference": "42", "scorer": "numeric"},
    {"user": "Repeat the number you just gave.", "reference": "42", "scorer": "numeric"}
  ]
}
\`\`\`

Run:

\`\`\`bash
python -m qmini.cli eval-conversations \\
  --model-dir /path/to/final_model \\
  --dataset evaluation/conversations.jsonl \\
  --max-new-tokens 128 \\
  --output reports/conversation-evaluation.json
\`\`\`

Generated assistant replies are fed back into the prompt for the next turn. The evaluator preserves system/role/end markers and removes only complete oldest turns when the prompt reaches \`max_context\`. If the current user turn cannot fit by itself, evaluation fails explicitly rather than silently truncating it. The report contains per-turn predictions, task-specific pass rate, Wilson intervals, per-category/per-language results and the history-token count before each turn.

The benchmark set must remain held out from tokenizer fitting, pretraining and SFT. Use numeric/JSON scorers for objectively structured answers where possible; lexical exact match and token F1 remain diagnostics. Human review and externally checked answers are still needed for open-ended factuality, science, reasoning and code behavior.


## Q2-v2 production format and legacy archive compatibility

All new exports use Q2-v2 normalized symbols {-1,-0.5,+0.5,+1}, with FP32 row scale 2 * mean(abs(original row)); an all-zero matrix row has scale zero. The packed runtime, Python fallback, depthwise convolution and native C++ ABI carry an explicit format version. Legacy Q2-v1 exports continue to decode under their original {-2,-1,+1,+2} table. The exact normative codebook, metadata and scale invariants are in docs/Q2_QUANTIZATION_SPEC.md.

Ternary/INT4/INT8 weight quantizers remain experimental ablation methods and are not production exporter formats.


When running quantization ablations, optionally pass `--eval-tokens` (and `--eval-mask` for assistant-only targets) plus a fixed sequence length, batch count and seed. The tool evaluates FP16, INT8, Q2-v2, ternary and INT4 candidate weights against the exact same held-out token stream while keeping the INT8 activation path fixed. The default evaluation cap is 10M parameters to bound host-memory use from candidate state copies; raise it only on a machine with enough RAM. The report records the evaluation stream/mask hashes, token-weighted NLL and perplexity per format. This is still a language-model metric, not a semantic assistant-quality score.



## Quantization report provenance

The ablation report groups per-tensor reconstruction metrics into embedding, output-projection, router, expert, convolution, state-space and other components. It records Q2 row-scale distributions, including the number of zero-scale rows, and splits estimated payload into packed matrix-symbol/FP16 matrix bytes, scales and vectors. If the exact packed export is present beside the source model, its file size is reported separately.

A final-model model.pt can be used as the higher-precision reference only when its model_manifest.json matches the config/parameter count and its dense and packed export SHA-256 values. A Q-Mini-owned latest.pt training checkpoint remains a supported alternative. These integrity checks prevent a stale FP16 file from being compared with a different packed release.
