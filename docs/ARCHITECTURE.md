# Q-Mini architecture lock

## Forward graph

```text
Token IDs
   -> Q2 token embedding
  -> [QMiniBlock x 6]
       -> RMSNorm
       -> quantized content/gate projection
       -> left-padded depthwise causal convolution
       -> SiLU(content) * sigmoid(gate)
       -> output projection + residual
       -> RMSNorm
       -> sparse Top-1 router
       -> selected expert MLP only
       -> residual
  -> one shared QMiniBlock repeated 1..4 times
  -> final RMSNorm
  -> tied embedding projection
  -> next-token logits
```

No Transformer attention, Q/K/V or dot-product attention modules are present. Legacy profiles use left-padded causal depthwise convolution only. New hybrid profiles combine that convolution with a gated diagonal state-space recurrence. Router and expert operations remain token-local.

## Causality and memory

Each convolution is padded only on the left by `dilation * (kernel_size - 1)`. No future token can affect an earlier output. The convolution path has a finite receptive field of `1 + (kernel_size - 1) * sum(dilations)`. Hybrid profiles additionally maintain a fixed-width learned recurrent state, updated as `s[t] = a[t] * s[t-1] + (1-a[t]) * u[t]`. Training uses a differentiable parallel associative scan; token-by-token generation uses the equivalent recurrent update. This state compresses prior information and cannot guarantee exact retrieval of arbitrary earlier text.

`max_context` bounds the length of one batch forward or initial prompt prefill. It does not cap the total length of a streaming decode. Streaming keeps the fixed-size convolution caches and one diagonal SSM state per active layer; it never replays the prompt after each generated token. Batch/stream parity is tested within the configured prefill length. Beyond that length, the model has recurrent compressed-history semantics, not an exact rolling-window transcript: old information may influence the state, but exact arbitrary-token recall is not guaranteed.

## Text and control-token boundary

The tokenizer reserves `<pad>`, `<bos>`, `<eos>`, `<unk>` and the chat role/end markers as structural token IDs. Literal occurrences of those spellings inside untrusted dataset message bodies, interactive user text, recalled notes or stored assistant text are escaped with a zero-width separator after the opening `<`. This keeps quoted text from forging role boundaries, terminating a message early or injecting special token IDs into response-only loss masks. Only the chat serializer itself writes the real structural markers.

## Quantization

Production matrix weights use exactly the Q2-v2 normalized codebook {-1.0, -0.5, +0.5, +1.0}, with two bits per symbol. Each output row has FP32 scale = 2 × mean(abs(original row)); decoded effective weight = normalized symbol × row scale. Four symbols are packed per byte. A fully zero row has scale 0 and remains exact zero; an exact-zero source entry in a nonzero row maps to +0.5 × row scale because there is no zero symbol.

QAT uses FP32 master parameters and optimizer state with an STE; those are training state, not deployed matrix weights. Activations use signed INT8 and do not widen matrix weights. New exports use Q2-v2; legacy Q2-v1 artifacts retain their original decoder. The normative contract is [Q2_QUANTIZATION_SPEC.md](Q2_QUANTIZATION_SPEC.md).

The packed loader installs matrix codes and scales without materializing every dense FP32 matrix. CPU QLinear projections and the tied vocabulary projection use the versioned C++17 INT8-by-Q2 kernel when the compatible library is present. Otherwise, PyTorch fallback decodes one matrix at a time and the tied vocabulary projection in bounded row chunks. The packed runtime is inference-only; training uses FP32 QAT master parameters.

## Sparse experts and refinement

Each token is sent to one expert per block. Router auxiliary loss discourages imbalance but does not guarantee meaningful expert specialization. One additional block is shared and reused from one to four times. This is learned hidden-state refinement, not symbolic theorem proving or a guarantee of correct reasoning.

The authoritative parameter count is computed from `sum(p.numel() for p in model.parameters())` on the instantiated model.

## Native runtime boundary

The native kernel covers dense linear projections only. AArch64/NEON builds use vector widening multiply and pairwise integer accumulation for the INT8-by-Q2-v2 dot product; other targets use the scalar integer fallback. Causal depthwise convolution, RMSNorm, token embedding lookup, sparse dispatch and state management remain PyTorch operations. This is a packed low-memory hybrid runtime, not a fully integer-only or fully native ARM64 graph. CI cross-compiles the C++ primitive for AArch64, but only a real Android device benchmark can establish phone speed, peak resident RAM, thermal stability and Android compatibility.


## Local memory boundary

The qmini/memory.py module provides an optional local SQLite store. FTS5 indexes note content and tags when available; otherwise search falls back to literal substring matching. Writes happen only through explicit user commands. This subsystem is not a neural memory layer, does not alter model weights and does not grant arbitrary tool execution.

Model weights are immutable during inference. Q-Mini does not perform autonomous online gradient updates or silently rewrite its own parameters after a chat. Persistent notes are retrieved as external context. Actual learning from new curated examples requires a separate supervised fine-tuning run, initialized with `--init-from`, a fresh optimizer and an explicit evaluation cycle.


## Higher-capacity profile

The compact baseline is preserved in configs/qmini_base.json. For GPU training, configs/qmini_research_134m.json supplies width 640, expert hidden width 1,024, ten causal blocks, eight Top-1 experts per block, an 8,192-token target vocabulary and a 2,048-token configured context. The dilation schedule is 1, 2, 4, 8, 16, 32, 64, 128, 256 and 512.

For the full 8,192-token vocabulary, the instantiated shape contains 134,426,200 parameters, including one shared refinement block. The CI audit computes this value from actual PyTorch parameter tensors. If tokenizer training produces fewer than 8,192 entries, the active config and model parameter count are adjusted to that actual vocabulary.


## Hybrid profile and resource limits

`configs/qmini_research_152m.json` contains 152,483,800 parameters at its full 8,192-token vocabulary. The state-space block is inserted in every main block and in the shared refinement block. Top-1 routing means only one expert per block processes a token, but all experts remain part of the stored model and optimizer state.

`configs/qmini_5m.json` contains 5,245,996 parameters. `configs/qmini_5b.json` contains 5,132,630,288 parameters. Use `python -m qmini.cli plan --config configs/qmini_5b.json` to inspect it without allocating a 5B model.

The 5B FP32 parameter, gradient and two AdamW moment arrays require about 82.1 GB before activations and runtime overhead. FSDP/ZeRO and expert/tensor parallel execution are not implemented, so the 5B config is not yet an executable H100 distributed-training setup.


## 5B deployment footprint and CPU runtime

For \`configs/qmini_5b.json\` (5,132,630,288 parameters), the row-padded Q2 weight payload, FP32 row scales, FP16 vectors and a metadata reserve are estimated at **1,294,831,200 bytes** (about 1.295 GB decimal / 1.206 GiB). This is below the 1,500,000,000-byte model-weight artifact budget. The estimate is based on exact tensor shapes, not a measured trained checkpoint; tokenizer and application files are separate.

Large training runs keep the resumable \`latest.pt\` checkpoint in the run directory. The final deployment folder omits the redundant dense FP32 \`model.pt\` when it would exceed 256 MiB and retains \`model.qmini.npz\`, configuration and tokenizer. \`qmini generate\` and \`qmini chat\` prefer the packed CPU runtime automatically. A packed archive can be supplied to \`--init-from\` for a later training stage; it is dequantized one tensor at a time into the separately allocated trainable model.

Run \`python -m qmini.cli plan --config configs/qmini_5b.json\` to inspect the estimated artifact size. Run \`python -m qmini.cli verify-export --path final_model/model.qmini.npz\` to validate an archive without expanding the complete FP32 state. CPU compatibility is implemented, but speed on any specific handset is not claimed until benchmarked on that device.


## 5B sparse-versus-dense compute reference and RAM estimate

The sparse configuration has 5,132,630,288 total parameters, 852,873,488 unique Top-1 active parameters per token and 16 experts per block. All experts' packed weights must remain present in the model artifact, even though only one expert is selected for each token in a given block.

\`configs/qmini_dense_853m.json\` is a single-expert dense reference with 852,281,344 parameters. It omits the one-expert router and router normalization because a one-element softmax is always 1 and produces no useful router gradient. It is close to the sparse profile's unique active parameter count, but it is not yet a scientifically controlled comparison: both variants must be trained on the same corpus, tokenizer, tokens seen, schedule and random seeds, then compared on the same held-out tasks and hardware.

For batch 1 with the maximum configured thinking passes, the config planner estimates 80,527,360 bytes of persistent convolution/SSM/rolling-token state, and 1,409,085,024 bytes for packed weights + those caches + simplified activation/fallback buffers. This is an engineering estimate, not measured process RSS. Python/PyTorch runtime memory, allocator fragmentation, OS usage, tokenizer storage and archive decompression overlap are excluded. A 4 GB physical-RAM deployment claim must wait for measured peak RSS on the target handset.


The normative production matrix-weight contract and versioned legacy decoder are specified in docs/Q2_QUANTIZATION_SPEC.md.
