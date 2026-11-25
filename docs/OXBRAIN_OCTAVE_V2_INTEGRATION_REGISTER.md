# OxBrain-Octave v2: project research integration register

**Status:** Experimental integration plan on `research/wavebit-hsra-integration`.
**Authority:** Q-Mini's project-owned experiments and their reproducible artifacts are valid engineering inputs. A public paper is not a prerequisite for preserving or integrating a user-developed method.
**Safety rule:** Keep the original zero-free legacy model unchanged. New techniques are opt-in in the zero-inclusive Add/Sub track until accuracy, compatibility and resource tests pass.

This register separates the *origin of an idea* from the *evidence available in this checkout*. A method can be original to the project and still need its implementation, reference vectors or benchmark artifacts brought into this branch. Lack of a public paper does not make it false; an unrepeatable speed or quality claim is not treated as a measured result.

## Integration layers

| Research component | Correct integration layer | Q-Mini integration decision |
|---|---|---|
| Mirror-63 / Mirror-13 lookup tables | Native integer coefficient/activation decoder | Preserve as project-origin kernel work. Port the exact table-generation rule, signedness, overflow bounds and test vectors. Keep scalar reference and NEON implementation numerically identical. |
| Deferred Split-Accumulator | Native dot-product accumulation | Preserve the project's method and internal test vectors. Add a scalar reference, explicit accumulator-width proof, split/merge parity tests and overflow tests before enabling it in the production dispatch path. |
| WaveBit / H-BitLinear / Hadamard rotation | Training quantizer + packed projection | Existing WaveBit/Octave path and Hadamard activation quantizer remain. Wire the native path end-to-end and compare INT4/INT8 activation modes with identical checkpoints and held-out English quality. |
| Octave-Shift x2/x4 with zero-skip | Weight codebook + serializer + native kernel | Existing five-base-symbol plus shared boost-bit format remains distinct from BitNet ternary. Preserve v1/v2 ABI and packed format compatibility. |
| BitNet-style ternary / 1.58-bit weights | Separate quantizer ablation | Do not silently reinterpret Octave codes. A separately named codec/profile is required so checkpoints and loaders cannot mix codebooks. |
| Monarch structured matrices | Projection implementation | Add as an optional projection backend only after a tested factorization, inverse/packing contract and crossover benchmark exist. Keep the direct projection fallback. |
| tANS / Huffman entropy coding | Offline model archive | Add as an archive codec after measuring actual symbol distributions. Runtime decoding must be streaming/bounded-memory and verify hashes before model load. |
| Loss-free dynamic MoE bias | Router selection + training-step hook | Integrated as an opt-in Add/Sub config. The bias changes top-1 choice, not the original softmax gate weight; it is updated once after each optimizer step in both pretraining and SFT. |
| Gated diagonal SSM | Sequence model | Existing baseline remains available and attention-free. |
| Mamba-2 / Gated DeltaNet / 1:7 hybrid | Optional sequence-model profile | Integrate as a separately named architecture profile with fresh random initialization. It must not silently replace the current recurrence or change old checkpoint interpretation. Attention ratio variants remain experimental profiles, not the default. |
| Three-bit KV cache | Attention-profile runtime only | The present baseline has no attention KV cache. Implement the cache only for a profile that actually has attention; do not add a dead cache to the current SSM-only graph. |
| ReplaySSM | Recurrent decode backend | Add behind a decode interface after state snapshot/restore, deterministic replay and cache invalidation contracts are tested. |
| STree / EAGLE-3 / DFlash / DSpark | Speculative decode backends | Add as separate draft/verify strategies sharing the same verifier API. Keep greedy/sampling baseline and fall back automatically when the draft is slower or rejected too often. |
| mHC / nGPT / Huginn recurrence ideas | Training/residual stability | Add optional normalization/recurrence variants with activation-norm, gradient-norm and long-unroll tests. Do not assume a name guarantees stability. |
| Snell-style test-time compute allocation | Inference compute controller | Connect to an explicit refinement/step budget and a held-out stopping-policy evaluation. Do not advertise a benchmark multiplier as an expected model gain. |
| PTRM | Reasoning/evaluation harness | Keep as a task-specific verifier/rollout experiment for puzzle-style tasks. Report its benchmark separately from general assistant conversation quality. |

## Implementation order

1. **Compatibility foundation:** preserve baseline and old checkpoint loading; make every experimental switch explicit in the config and export manifest.
2. **Integer kernel path:** project-owned Mirror LUT and Deferred Split-Accumulator reference implementations first, then NEON implementation and bit-exact/parity tests. Benchmark on real ARM hardware before enabling automatic dispatch.
3. **Quantized projection path:** complete WaveBit/Hadamard/Octave end-to-end inference wiring; add a distinct ternary ablation and structured Monarch projection backend without conflating formats.
4. **MoE and recurrent training:** loss-free router bias is now integrated behind a default-off flag. Add the other router/stability variants with per-layer load and norm diagnostics.
5. **Sequence model alternatives:** implement Mamba-2/Gated DeltaNet as named, fresh-initialized alternatives; compare matched tokens, parameters, context and hardware. Keep the current attention-free baseline.
6. **Decode alternatives:** add a state snapshot/restore contract, then ReplaySSM and speculative draft/verify adapters. KV cache belongs only to an attention-enabled profile.
7. **Archive and reasoning:** add entropy-coded export and task-specific PTRM evaluation independently of the neural weight format.

## Acceptance gates

Every experiment records source revision, config, seed, tokenizer/data hashes, checkpoint hash and exact device/runtime. Minimum checks:

- deterministic scalar reference and native parity, including zero, negative values, tail lengths and accumulator overflow;
- forward/streaming parity and state rollback parity where relevant;
- exact config/state-dict/export/load round-trip, including compatibility with a baseline checkpoint;
- held-out English token loss and a separate held-out human conversation set;
- peak RAM, serialized bytes, prefill/decode tokens per second, energy per generated token and thermal stability;
- matched baseline comparisons. A speedup is accepted only when end-to-end latency improves without unacceptable quality, memory or energy regressions.

No research component is rejected solely because it lacks a public citation. At the same time, no speedup, reasoning multiplier or device capability is inferred from an algorithm's name. Project-owned implementations and measurements are to be imported and reproduced, not erased or renamed.
