# Q-Mini Add/Sub Phone Runtime Readiness

**Target device:** Realme C73 5G, model RMX3945 (the user's reported device identifier).
**Status:** target profile and test protocol only. No completed Add/Sub Android build or physical-device speed measurement exists yet.

## Verified target hardware profile

Realme India's published C73 5G specification lists:
- MediaTek Dimensity 6300, 6 nm.
- CPU: 8 cores, up to 2.4 GHz; 2x Cortex-A76 + 6x Cortex-A55.
- Physical RAM configuration listed as 4 GB. Dynamic RAM expansion is virtual capacity and must not be counted as physical RAM available to a native inference process.
- Android 15 / realme UI 6.0 in the published spec; the user's current device environment reports Android 16, so the runtime must detect the actual OS/API level at run time.

Official specification: https://www.realme.com/in/realme-c73-5g/specs

## What can and cannot be claimed today

The Add/Sub model's current reference architecture has 760,340 parameters and a five-symbol weight alphabet. Five symbols require 3 bits per weight in fixed-width packing. The existing .qma archive is a weight container with a float32 auxiliary sidecar; it is not a mobile inference engine.

The Python/PyTorch reference generator is not the phone runtime. The current integer prototype covers only an additive projection primitive and bounded integer residual addition. Full integer convolution, SSM recurrence/state, gate functions, expert routing/dispatch, normalization, activation/state lookup tables, output projection and sampling are still required. Existing ARM64 CI only cross-compiles the legacy qmini/native_kernel.cpp; it does not prove the Add/Sub model runs on ARM Android.

Do not delete or overwrite the legacy Q2-v2 model/checkpoints while the new Add/Sub model is not trained and not deployable. The Add/Sub line is the primary development target; the legacy line is retained as a read-only compatibility/reference baseline until the new model passes the replacement gate.

## Required mobile runtime design

- Native C++17 core, built for Android arm64-v8a, with a scalar reference implementation and ARM64 NEON implementation. Dispatch must be feature-safe and fall back to scalar code.
- No Python, NumPy, PyTorch, BLAS, GPU requirement, or network dependency in the deployed inference process.
- Preserve the exact five-symbol alphabet {-1.5, -0.5, 0, +0.5, +1.5}; code 2 is exact zero and contributes exactly nothing.
- All projection accumulation must use precomputed contribution lookup, comparisons/index selection, integer addition/subtraction, and defined saturating/bounded behavior. No multiplication, division, matrix multiply, dot product, floating-point rescale, or hidden multiply in normalization/gating.
- Version all lookup tables and activation scales. Export must include tokenizer/config/runtime contract hashes and a complete tensor manifest.
- Keep streaming convolution and SSM caches fixed-size; do not replay the whole prompt for each new token.
- Android UI/storage integration must be a thin layer. Model files are read-only at inference and logs must not save private prompts by default.

## Release-gate tests

### Numerical correctness
- Exhaustively test each int8 activation value against each of the five weight symbols.
- Test zero-symbol skip, negative values, maximum accumulator bounds, overflow rejection/saturation and deterministic rounding.
- Compare scalar and NEON outputs exactly on identical inputs.
- Compare the complete integer graph against the floating-point reference within per-layer calibrated tolerances.
- Test all tensor shapes, padding, empty inputs, malformed archives, corrupt hashes and unsupported versions.

### Model behavior
- Batch prefill and streaming decode must agree within the documented quantization tolerance.
- Future-token perturbation must not change earlier logits.
- Long conversations must keep bounded cache size and must not replay the full prompt.
- Run held-out English tasks, instructions, multi-turn conversation, arithmetic, retrieval/state tracking and repetition-collapse tests on the same checkpoint and tokenizer.
- The release gate must fail on blank, identical, repetitive, truncated or generation-error outputs.

### On-device performance (measure, do not estimate)
- Record device model, SoC, Android/API level, app build hash, model and tokenizer hashes, thermal state and power mode.
- Measure cold start, time to first token, prefill at 64/256/512 tokens, decode for 128 generated tokens, p50/p95 token latency, tokens/second, peak process PSS/RSS, model-file size and battery/thermal change.
- Run at least five repetitions after a documented warm-up. Repeat sustained generation for 10 minutes and report any thermal slowdown.
- Compare scalar vs NEON and the Python reference only as a correctness/control baseline, not as competing production backends.

## Provisional engineering targets (not measured results)

These are initial goals for the 4 GB physical-RAM target, not claims about current performance:
- Total model + tokenizer + LUT files: <= 8 MiB.
- Incremental inference-process PSS attributable to the model/runtime: <= 256 MiB.
- Sustained generation: >= 10 tokens/second after warm-up on the target device.
- Median decode latency: <= 100 ms/token; p95 <= 200 ms/token.
- Cold model load: <= 2 seconds.
- Sustained 10-minute run: no crash, unbounded memory growth, or >20% throughput loss after thermal stabilization.

If the hardware or Android build cannot meet a target, report the measured result and adjust the product promise; do not hide the result or optimize a synthetic microbenchmark while end-to-end latency gets worse.

## Replacement gate: new Add/Sub becomes the sole default

The user-facing default may switch from legacy Q2-v2 to Add/Sub only after all of these are true:
1. A real-data English pretraining and assistant SFT run completes with immutable data/tokenizer/config/checkpoint hashes.
2. Held-out readiness gates pass and a human-reviewed test suite shows no collapse or critical regressions.
3. The full integer-only graph passes operator audit and parity tests; no float32 auxiliary inference path remains.
4. ARM64 scalar + NEON builds pass, and the complete model is measured on the actual Realme C73 5G.
5. The new release can be loaded, chatted with, resumed/exported and rolled back without deleting legacy artifacts.

Until then, mark Add/Sub as the primary experimental target, but not a finished phone assistant. Preserve the old model as a compatibility baseline; remove it from the default training route only after replacement passes.