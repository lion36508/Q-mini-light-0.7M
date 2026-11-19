# Q-Mini Add/Sub v1 — Architecture Contract

**Status:** design contract only. No model implementation, checkpoint or training run is claimed.

## 1. Product-line separation

| Legacy line | New line |
|---|---|
| Python package: `qmini/` | Package boundary: `qmini_addsub/` |
| Q2-v2 artifact | New artifact format: `qmini-addsub-v1` |
| 4,096 vocabulary legacy profile | Independently versioned config/tokenizer |
| Existing Q-Mini Light checkpoints | New checkpoints only |
| Existing Drive folder | Separate `QMini-Light-AddSub-v1/` Drive folder |

The old model remains runnable and its artifacts remain untouched. The new code must not import legacy model/runtime classes or load legacy weights. Stable ideas such as data manifests, checksums, tests and resumable-save principles may be reused, but architecture and artifact loaders stay separate.

## 2. Required weight alphabet

The new model's discrete weight alphabet is exactly:

`{-1.5, -0.5, 0, +0.5, +1.5}`

Zero is a real symbol. A zero weight means no connection contribution and must be skipped exactly; it must not be approximated as a small positive or negative weight.

**Storage fact:** five distinct symbols cannot fit in a fixed 2-bit code (2 bits provide only four codewords). The initial packed format therefore requires at least 3 bits per weight, with three unused 3-bit codes reserved for future versioning or rejected as invalid. Do not market this alphabet as 2-bit.

## 3. Inference arithmetic rule

The production inference path must use only:

- integer addition
- integer subtraction
- comparisons and index selection
- constant-time table lookup for coefficient contributions and nonlinear functions

No runtime multiplication, division, floating-point rescaling, matrix multiplication, dot product, tensor `@`, `torch.matmul`, `F.linear`, convolution multiply, or fused multiply-add is permitted.

A coefficient lookup maps the activation integer and one of the five weight symbols to a precomputed integer contribution. The inference accumulator then combines contributions only with add/subtract. The zero symbol maps to a skipped contribution. Activation functions, normalization and other nonlinear transforms require fixed lookup tables or an alternative explicitly approved add/sub-only implementation. They must not hide multiplication in a scale, gate or normalization step.

This requirement covers the *entire deployed graph*, not only the projection kernel.

## 4. Training policy

Training precision is an independent research choice. This contract does **not** require a staged FP16/BF16 phase, does not prescribe an FP16-to-low-bit schedule, and does not authorize any change to the legacy model's trainer.

The new model is initialized independently and trained only using the project's approved, provenance-checked English corpus. Do not load external pretrained checkpoints, adapters, LoRA weights or old Q-Mini weights.

## 5. Required package layout

```text
qmini/                         # existing legacy Q-Mini Light; preserve
qmini_addsub/
    README.md
    config.py                  # independent model configuration
    model.py                   # add/sub graph only
    lookup_tables.py           # immutable, versioned contribution/activation tables
    arithmetic.py              # bounded integer add/sub accumulation
    tokenizer.py               # independent tokenizer contract/version
    data.py                    # provenance and document-level split
    train.py                   # new training implementation
    checkpoint.py              # new checkpoint format; never reads qmini files
    export.py                  # qmini-addsub-v1 packing and validation
    runtime.py                 # deployed add/sub-only runtime
    tests/
```

This is the target layout, not a claim that all these files already exist. The legacy package must not import from `qmini_addsub`, and the new package must not import model/runtime classes from `qmini`.

## 6. Artifact compatibility

The manifest must include:

- `format = qmini-addsub-v1`
- exact alphabet and code mapping, including zero
- bits per weight and rejected/reserved codewords
- accumulator width, signedness and saturation/overflow policy
- lookup-table hashes and versions
- tokenizer hash and vocabulary size
- config hash and actual parameter count
- source dataset manifest hash and training-run identity
- proof/test report identifying the audited inference runtime build

The loader must reject Q2-v1, Q2-v2, dense PyTorch checkpoints and any unknown artifact format.

## 7. Acceptance tests

1. Round-trip all five symbols, including exact zero, across pack/unpack.
2. Invalid 3-bit codes are rejected; no symbol silently maps to another.
3. Exhaustive small-domain parity against a simple add/sub reference interpreter.
4. Tests prove zero weights add no contribution.
5. Static and compiled-binary audit rejects multiply/divide/matmul/FMA instructions in the deployed inference graph.
6. Test all nonlinear and normalization paths; no hidden multiply is allowed.
7. Same seed/config/data produces deterministic initialization and reproducible export.
8. Save/resume parity and artifact tamper detection pass.
9. Held-out English NLL and task evaluation are reported separately from synthetic smoke tests.
10. Real target-device RAM, speed, thermal and energy measurements are published before phone-support claims.

## 8. Explicit non-goals for v1

- Do not merge with legacy Q2-v2 weights or checkpoints.
- Do not overwrite or migrate the old Drive dataset/checkpoint folders.
- Do not re-enable TinyStories or teacher-generated synthetic samples as production training data.
- Do not claim 2-bit storage for five symbols.
- Do not claim an add/sub-only runtime until the complete deployed graph and compiled instructions pass the acceptance tests.
