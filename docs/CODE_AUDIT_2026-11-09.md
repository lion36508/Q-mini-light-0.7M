# Q-Mini source-code audit — 2026-11-09

## Scope and honesty

This began as a **static source and test-coverage audit** at commit `4aa23e8ed024045988a571eab555c375ac72c1c4`; the follow-up status below is updated against the later successful CI commit `852a8ccc619533983c222bffee60e35feace9c87`. The review covered the model, configuration, data ingestion, tokenizer, training loop, generation/chat, packed export/runtime, quantization, native projection kernel, evaluation, CLI, benchmark, project docs, and the corresponding test files.

The source was reviewed remotely through GitHub. The full test suite was **not executed in this audit session**, and no training run or hardware benchmark was performed. Findings below distinguish confirmed code defects from engineering gaps and research risks. A static review cannot prove that every possible defect has been found.

**Training ownership:** training remains the user's task. This audit does not start training, download a pretrained checkpoint, or change the project's random-initialization-only rule.

## Executive verdict

Q-Mini is not throwaway code. It already contains a real, non-trivial research implementation: an attention-free causal-convolution language model; an optional gated diagonal state-space path; sparse Top-1 expert routing; a shared iterative refinement block; a tokenizer and document-split pipeline; response-loss masks; a single-device training/checkpoint path; packed Q2 export; a CPU C++ projection primitive; streaming decode; evaluation tools; and meaningful unit tests.

However, it is still a **research prototype**, not yet a demonstrated high-quality general assistant or a proven 5B/4GB-phone model. Code execution, a low-bit archive, a large parameter count, and a falling training loss are not by themselves evidence of assistant capability.

## 1. Architecture review

Current hybrid architecture, in order:

1. Token embedding (tied to the language-model output projection).
2. Main blocks with RMSNorm, gated causal depthwise convolution, optional gated diagonal SSM, Top-1 router and expert MLP.
3. One shared Q-Mini block reused for the configured refinement/thinking passes.
4. Final RMSNorm and tied vocabulary projection.

There is no Transformer self-attention, Q/K/V projection or attention module in the implementation. The current hybrid profile is a custom causal-convolution + diagonal-recurrence + sparse-MoE design. It is **not** an implementation of Mamba-2 or RWKV-7.

### Strengths found in source/tests

- The convolution path is left-padded and has a token-by-token cache path.
- The SSM scan has tests against the sequential recurrence, gradient checks, causality checks and batch-versus-streaming parity tests.
- Streaming state is fixed-size and the tests exercise generation beyond the prefill context limit.
- Top-1 expert routing has a differentiable selected-gate path and a balance auxiliary loss; the one-expert case correctly bypasses the degenerate one-element router.
- Training includes masked response loss, gradient accumulation, clipping, mixed-precision selection, checkpoint/resume contracts and dataset/tokenizer fingerprints.
- Checkpoint loading uses restricted `weights_only=True` loading for model/checkpoint paths.
- The packed loader avoids materializing all dense FP32 matrices at inference and validates tensor names, shapes, dtypes and scales.
- The native C++ primitive has scalar and ARM NEON code paths, and CI compiles the primitive and cross-compiles an ARM64 shared library.

These are real engineering foundations, not evidence that the design is already optimal.

## 2. Confirmed defects reported as focused issues

### #25 — RESOLVED: JSON arrays with legal leading whitespace
File: `qmini/data.py`, `_iter_json_file()` and `_iter_json_array()`.

Format detection skips leading whitespace, then seeks back to the start. The streaming array parser immediately reads one character and requires it to be `[`. Valid files beginning with spaces/newlines therefore fail. The report asks for bounded whitespace skipping and regression cases for spaces, tabs, blank lines and BOM.

https://github.com/lion36508/Q-Mini/issues/25

### #26 — RESOLVED: long interactive-chat history now preserves role/control markers
File: `qmini/generate.py`, `interactive_chat()`.

The trim branch slices an arbitrary token suffix and calls `tokenizer.decode()` without explicitly preserving special tokens. The Hugging Face Tokenizers API documents `skip_special_tokens=True` as the default for `tokenizers.Tokenizer.decode()`; Q-Mini's role and end markers are special tokens. The result can remove message boundaries, and the arbitrary slice can begin mid-turn. The issue requests turn-aware truncation and a regression test that continues chatting after truncation.

Reference for the Tokenizers decode default: https://huggingface.co/docs/tokenizers/python/latest/api/reference.html

https://github.com/lion36508/Q-Mini/issues/26

### #27 — RESOLVED: tokenizer's zero validation-fraction contract is consistent
Files: `qmini/tokenizer.py` and `qmini/data.py`.

`train_tokenizer()` accepts `validation_fraction=0`, but delegates to `iter_training_text_records()`, which rejects `valid_fraction=0`. The CLI also accepts the float and fails later in the helper. The issue requests a single documented contract: either support tokenizer fitting with no holdout, or reject zero at the public API/CLI boundary.

https://github.com/lion36508/Q-Mini/issues/27

### #28 — RESOLVED: sparse response-mask windows are indexed and sampled safely
File: `qmini/train.py`, `_sample_batch()`.

The loader checks that a mask has some supervised tokens somewhere in the full stream. The sampler then makes at most 128 uniformly random proposals per sample and raises if all those windows miss the supervised positions. With extremely sparse response-only labels, the dataset can be globally valid while a training/evaluation batch fails probabilistically. The issue requests a supervised-position-aware or deterministic fallback sampler and tests for sparse/boundary masks.

https://github.com/lion36508/Q-Mini/issues/28

These four reports describe code-level defects or robustness failures; they are not claims that normal default-path training always fails.

## 3. Major architecture, quality and deployment risks

The following are important, but should not be mislabeled as proven bugs:

### A. Long-range information and model quality
The convolution path has a finite receptive field. The diagonal SSM carries a learned compressed state, not exact access to arbitrary earlier tokens. The code correctly documents this boundary. Large context values and repeated refinement do not guarantee retrieval or reasoning. Long-context and real task results are required.

### B. Sparse MoE quality and efficiency
Top-1 routing reduces active expert compute, but all experts still contribute to total model storage and optimizer state. Python-level per-expert dispatch may become a bottleneck. Router balance loss is a useful signal but does not prove specialization or prevent starvation on real corpora. The current benchmark/test scale is not a 5B expert-dispatch validation.

### C. Quantization quality
Production matrix weights now use Q2-v2 normalized symbols `{-1.0, -0.5, +0.5, +1.0}` with row scale `2 * mean(abs(original_output_row))`; new archives are `qmini-packed-v2`, while legacy `qmini-packed-v1` retains its original `{-2,-1,+1,+2}` decoder. The Q2-v2 alphabet has no zero symbol for individual entries in a nonzero row; fully zero rows use scale zero. Codebook, metadata, Python/packed/depthwise and native C++ format-version paths have regression coverage. This establishes format consistency, not quality preservation: the tiny 33,606-parameter smoke ablation reported relative L2 reconstruction error 0.270430 for Q2 versus 0.000264 for FP16, with only 16 synthetic held-out targets. Compare real Q-Mini-trained checkpoints on larger held-out NLL and task sets before claiming that Q2 preserves assistant quality.

### D. Native runtime boundary
The C++ primitive accelerates linear projections. Embedding lookup, causal convolution, RMSNorm, router/expert dispatch, state management, tokenizer and generation still involve Python/PyTorch. This is a hybrid packed runtime, not a complete native or integer-only graph. Real target-device memory, latency, thermal and energy measurements are still needed.

### E. 5B training is not implemented by the current trainer
The current trainer selects one CUDA device when available, otherwise CPU. It does not implement distributed process setup, FSDP/ZeRO sharding, or expert parallelism. The 5B configuration's FP32 weights, gradients and two Adam moments alone are roughly 82.1 GB before activations/workspaces. A 5B config file is not a 5B training system.

### F. Assistant capability is unproven
The current task evaluator reports normalized exact match and lexical token F1; these are diagnostics, not semantic correctness. The docs already call out the need for verified math scoring, constrained code execution, multilingual evaluation, multi-turn consistency and other capability measures.

Existing tracked work for these broader gaps:
- #12 — 5B architecture and real 4GB inference proof.
- #13 — optional random-initialized Mamba-2/SSD reference experiment, with no pretrained weights.
- #18 — distributed training infrastructure.
- #19 — evaluation across every supported refinement count.
- #20 — long-sequence/mixed-precision SSM numerical stress tests.
- #21 — reproducible Q-Mini-owned pretraining to SFT workflow.
- #22 — Q2 versus ternary/INT4/FP16 ablations.
- #23 — assistant capability benchmarks.
- #8 — CLOSED: Linux CPU smoke-model prefill/decode instrumentation is implemented and exercised in CI; it is not a 5B/phone benchmark.
- #19 — CLOSED: all configured refinement depths are measured with deterministic validation windows and per-depth metrics.

## 4. Existing areas that looked sound in this static pass

The following were checked and did not produce a new confirmed defect in this pass:

- Parameter-count formulas are reconciled against actual instantiated parameters in CI for the named profiles.
- The parallel SSM prefix composition matches the sequential recurrence in the unit test.
- The stream-state carries the refinement-step count forward; generation does not accidentally change the number of passes on subsequent tokens.
- Training validation aggregates masked NLL by supervised-token count.
- Prepared dataset manifests verify stream and tokenizer hashes; assistant-target masks are required for auto/assistant prepared datasets.
- The JSON top-level array parser is chunked and validates trailing data; issue #25 is a specific leading-whitespace regression, not a claim that the earlier bounded-array work is absent.
- The C++ linear kernel validates shapes/scales and uses 64-bit scalar accumulation; CI compiles the CPU implementation and cross-compiles ARM64.

This section is a record of reviewed paths, not a formal proof of correctness.

## 5. Priority order

1. Fix #25 and #26 first: they affect data ingestion and user-visible multi-turn conversation structure.
2. Resolve #27 and #28: make public API contracts and sparse response-mask sampling reliable.
3. Complete #19 and #20 before trusting results across refinement counts or mixed precision.
4. Complete #21 and #23 before calling a checkpoint a conversational assistant.
5. Complete #22 before claiming the Q2 format preserves model quality.
6. Treat #18 as a prerequisite to any serious 5B training attempt.
7. Treat #12 and #8 as release gates for the stated 5B/4GB target.

## 6. Scope explicitly not performed

- No model training or fine-tuning was started.
- No external pretrained weights were downloaded or loaded.
- No source-code bug fix was merged as part of this audit; the four findings were recorded as issues for controlled fixes and tests.
- The full pytest suite was not run in this session. Review of source and test files is not the same as executing the tests.
- No GPU, Colab, ARM phone, memory, speed, energy or thermal benchmark was performed.
- This report does not claim that every possible defect has been discovered.

## Follow-up verification on main

The later CI run #323 succeeded on commit `852a8ccc619533983c222bffee60e35feace9c87`: **143 tests passed and 3 were skipped** (two CUDA-only numerical stress cases on CPU CI, one NEON-only selection case on x86_64). CI also built the C++ projection kernel, cross-compiled ARM64 NEON, installed the package wheel, trained a tiny 33,606-parameter smoke checkpoint, measured prefill and cached decode, and exercised the Q2/INT8/ternary/INT4 ablation plus task and multi-turn evaluation tools.

Issues #8, #19, #25, #26, #27, #28 and #29 were closed only after this passing run and their regression coverage. Issues #12, #13, #17, #18, #20, #21, #22 and #23 remain open because they still require material work or target-hardware/quality evidence. The latest follow-up enhancement to the quantization report (component-level metrics, row-scale summaries and exact weight-payload breakdown) is being tested separately; do not treat it as covered by run #323 until its own commit has a green workflow.

## Audit conclusion

Q-Mini has a credible experimental code foundation and is not “useless code.” It needs targeted correctness fixes, numerical/quality ablations, stronger evaluation and a genuinely scalable training/runtime path before it can be described as a powerful general assistant. Keep the from-scratch random initialization requirement unchanged. Make every improvement test-backed, benchmarked and traceable to an issue.
## 7. Follow-up audit and CI evidence — 2026-11-09 15:56 UTC

This follow-up supersedes the earlier "not executed" test limitation above for the exact revisions and checks listed here. It does not convert a static audit into proof that every possible defect has been found.

### Latest verified revision

- `main`: `90d2c4ddbf886a0ce47318621e22a623e4622a45`
- CI run [#358](https://github.com/lion36508/Q-Mini/actions/runs/42649249377): completed successfully.
- Test result: **170 passed, 3 skipped**.
- Skips: two CUDA-only numerical tests because the CI runner has no CUDA GPU, and one runtime NEON test because the runner is x86_64. The ARM64 NEON shared library was separately cross-compiled successfully.
- Also passed: Python syntax validation for all eight Colab code cells, C++ scalar kernel build, ARM64 cross-compile, package wheel install/native-kernel check, reproducible CPU native/fallback benchmarks, tiny random-initialized checkpoint training and packed decode, quantization and structured-conversation smoke evaluation, audit and CLI checks.
- CI run [#356](https://github.com/lion36508/Q-Mini/actions/runs/42648154487) on the action-upgrade commit also passed: 167 tests passed, 3 skipped.

### Recent failed-run review

The 100 most recent GitHub Actions runs were inspected. Eleven had `conclusion=failure`; cancelled runs were counted separately and not misclassified as test failures. The failures fell into these groups:

- `#298`, `#305`, `#314`, `#321`, `#322`: chat-history trimming/context-limit behavior, evaluation handling when earlier turns are evicted, and old test assumptions about message-boundary formatting. Current test coverage exercises the repaired whole-turn trimming path.
- `#313`: a quantization ablation metric became non-finite on a zero/near-zero reference tensor. The current suite includes quantization edge-case tests and passes.
- `#325`, `#326`, `#340`: missing imports in the quantization/ablation test or diagnostic path (`numpy`/`json`). These were fixed and covered by later CI.
- `#345`: quantization manifest/config normalization caused three manifest-validation tests to fail. The config normalization and regression assertions were repaired in later commits.
- `#351`: one Colab notebook test compared a literal escaped newline rather than checking the two code fragments separately. The test assertion was corrected in a later commit.

The current full suite passing is evidence that the checked-in regressions have been repaired; it is not evidence that every runtime combination or dataset has been tested. Runs `#354`, `#355`, and `#357` were cancelled during overlapping branch updates under the workflow's `cancel-in-progress` concurrency policy; they are not additional failing test runs.

### Changes made in this follow-up

1. CI actions upgraded to Node-24-compatible majors (`checkout@v7`, `setup-python@v7`, `upload-artifact@v7`). Issue #32 was closed after run #356 passed.
2. `train_model(..., init_from=...)` now requires a Q-Mini `final_model/` manifest and matching `config.json`; verifies the recorded architecture and parameter count; verifies the Q2 packed archive fingerprint and contents; verifies the selected dense snapshot hash/size or that the selected `.npz` is exactly the manifest-listed packed export; and checks the source tokenizer hash when a tokenizer is supplied.
3. Standalone `.pt`/`.npz` weight files without Q-Mini final-model provenance are rejected for SFT initialization. Regression tests cover valid own-checkpoint dense and packed SFT initialization, orphan/external artifacts, tampered packed artifacts and malformed non-object manifests.

This provenance contract binds the model files together and reduces accidental use of third-party weights. It is not a digital signature and cannot prove the identity of the person who created a deliberately forged manifest. Do not treat files from an untrusted publisher as authenticated.

### Remaining open engineering blockers

These are incomplete features or evidence gaps, not presently demonstrated unit-test failures:

- **#34 (P0 data):** source/license provenance, release-mode license gates, data quality/language/source concentration reports and PII/secrets screening are not complete. The repository does exact and optional near-duplicate grouping and deterministic document-level train/validation splitting, but that is not a complete data governance system.
- **#17, #36:** staged long-run Colab training, robust cloud artifact storage and cost/interrupt recovery still need an end-to-end interrupted-run demonstration.
- **#18, #33:** real FSDP/ZeRO sharded training and sharded resume are not implemented. `qmini.distributed` currently supplies planning and environment preflight only.
- **#21, #35:** the SFT source is now restricted to Q-Mini-owned artifacts, but a reproducible full pretrain-to-SFT curriculum and measured held-out capability improvement have not yet been demonstrated.
- **#20:** CUDA BF16/FP16 and long-sequence stress tests remain unverified on actual GPU hardware; CPU CI cannot substitute for them.
- **#22:** Q2-v2 format consistency is tested, but quality preservation must still be measured on a genuinely trained Q-Mini checkpoint and larger independent holdouts.
- **#23:** semantic correctness, factuality, robust multilingual performance and constrained code execution are not established by current lexical/numeric/JSON smoke evaluators.
- **#12, #13:** the actual 5B target and 4GB-phone runtime are not built/proven; the optional random-initialized Mamba-2/SSD comparison has not been completed. Neither should be described as a current capability.

### Scope and limitations

No large-scale pretraining or SFT was started in this audit. No external pretrained weights were downloaded or loaded. No user dataset was available for license, privacy, language-balance or leakage audit. No CUDA GPU or physical ARM phone was benchmarked. The three CI skips are therefore still real coverage gaps. The current verified result is a passing CPU CI suite and improved checkpoint provenance—not a claim that Q-Mini is already a capable general-purpose assistant or that every open roadmap issue is resolved.

## 8. Follow-up source review — 2026-11-11

### Reviewed main revision

- Public repository: `lion36508/Q-Mini`.
- Branch reviewed: `main`.
- Exact base HEAD: `2f995ef642700f6444020b316d2e8da0d042bb5d` (2026-11-10 23:26:21 UTC).
- The preceding audit evidence above stopped at `90d2c4ddbf886a0ce47318621e22a623e4622a45`; it is historical evidence, not the current HEAD.
- The source tree still contains configurations and implementation code, not a trained 5M/152M/5B model release or persisted real corpus-based training run. The Colab notebook's code cells have no saved outputs. Its 5M intermediate and 152M long-training flags remain disabled by default.

### Additional defects and risks confirmed at the reviewed HEAD

1. The SFT verifier accepted a dense `.pt` path only when its manifest declared a dense snapshot and matching fingerprint, but the current final export writer always writes `dense_training_snapshot_included=false` with null dense fingerprint fields. A stale legacy `model.pt` could be preserved without becoming a valid SFT source. The safer contract is Q2-v2 packed `.npz`-only SFT; legacy dense files must not be mistaken for verified SFT checkpoints.
2. The SFT verifier checked packed artifact size/hash and parameter count but did not independently require `format=qmini-packed-v2` and `q2_version=2`, unlike the initial-brain verifier.
3. A dense-only directory with a non-v2 manifest failed with a generic missing-verified-weights message, which did not explain that the manifest did not fingerprint the legacy dense file.
4. The generate CLI did not expose the existing trusted serialized-prompt mode. Such mode must be explicit and warn that only programmatically serialized prompts are trusted.
5. The default Colab learning-proof gate accepted any positive held-out NLL movement, regardless of magnitude, and loaded the expected checkpoint without a clear missing-file diagnostic.
6. Colab setup used pip `--upgrade` for utility/tokenizer/data packages, which could needlessly replace already-satisfactory runtime packages. The cell-3 architecture preflight source also needed newline restoration so its code was executable rather than appended to its leading comment.
7. The CI precision scan classifies documented FP32 QAT master/optimizer paths under `training_parameter_or_optimizer`. This category is a review inventory, not a valid fail-closed CI category while that training representation is intentional.
8. The packed-export verifier/reconstructor assumed well-formed JSON object metadata and archive members; malformed member structures could leak raw `KeyError` or `AttributeError` exceptions.

### Changes prepared on the separate improvement branch

The fixes and regression coverage are being assembled on `gpt/model-improvements`, based on the main SHA above; no changes are made to `main` by this branch work.

- SFT source policy is narrowed to the manifest-listed Q2-v2 `.npz` export. The misleading `legacy_dense_snapshot_preserved` manifest field is removed; any old dense file is left untouched but is not trusted as an SFT source.
- SFT validation now checks packed format, Q2 version and parameter count.
- Dense-only legacy bundle errors identify an unfingerprinted manifest.
- The generate CLI adds an opt-in `--serialized-prompt` with a warning and frequency/presence repetition penalties.
- Colab's small learning-proof gate requires at least 0.05 held-out cross-entropy improvement; it reports the measured margin and checks for `latest.pt` before loading it. The existing smoke-step, 5M opt-in and `RUN_LONG_TRAINING=False` defaults are preserved.
- Colab setup no longer uses `pip --upgrade` for the listed runtime dependencies, and the architecture-preflight cell is made executable.
- The CI precision audit is explicitly informational while FP32 QAT master parameters and optimizer state remain part of training.
- Export verification/reconstruction normalizes malformed archive member errors to `ValueError`.
- Generation prompt prefill uses a batched causal-convolution/SSM pass to produce decode caches; MoE dispatch groups token indices after one assignment sort instead of scanning every token once per expert.
- Tests are added for packed-only SFT, Q2-v2 SFT format checks, repetition penalties, CLI warning behavior, batch-prefill parity, grouped MoE parity, malformed archives, and notebook guards.

### Scope deliberately not claimed

- No training or fine-tuning was started.
- No trained weights, checkpoints, dataset, benchmark numbers, or Colab outputs were created or inferred.
- The prefill and MoE code changes are implementation-level optimizations; they are not accompanied by a measured throughput claim until a reproducible benchmark is run.
- No 5M run design, default flag, Q2 codebook, attention-free architecture, or 1B-related file/folder was intentionally changed.
- A public CI pass does not substitute for real Colab execution, a licensed corpus review, CUDA numerical stress, or target-phone benchmarking.


## 9. Main advanced after the 2026-11-11 review

After the source review at `2f995ef642700f6444020b316d2e8da0d042bb5d`, main advanced to `d75d7157311fb3ea6828b31476e2884ecac42061` on 2026-11-11. That upstream commit adds a first release-mode dataset provenance gate:

- `docs/DATASET_PROVENANCE.md` documents the `qmini-source-provenance-v1` metadata contract.
- `qmini/data.py` now records source file hashes and provenance fields, supports explicit release mode, and screens for common PII/secret patterns.
- `qmini/tokenizer.py` binds tokenizer fitting to the dataset provenance hash and release-mode flag.
- `qmini/cli.py` exposes `--source-metadata` and `--release-mode` on tokenizer/prepare commands.
- `qmini/train.py` verifies tokenizer/dataset provenance and binds the dataset manifest into the resume contract.
- `tests/test_data.py` and `tests/test_train.py` add corresponding provenance and resume-contract coverage.

This is a meaningful improvement to data governance, but it remains a first fail-closed screening gate, not complete legal verification or a comprehensive PII scanner. A real corpus still needs manual licensing and privacy review, independent test data and measured language/source quality. No training corpus or training run is implied by the code's presence.

The improvement branch was rebased onto this new main commit rather than deleting or reverting the upstream provenance work. The files `qmini/data.py`, `qmini/tokenizer.py`, `tests/test_data.py`, and `docs/DATASET_PROVENANCE.md` are retained from main unchanged by the improvement set.
