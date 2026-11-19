# Q-Mini Add/Sub v1

This folder is the independent home for the next Q-Mini model line.

**Implementation status:** architecture contract only; runtime/trainer/checkpoint code has not yet been added.

Rules:

1. Do not import model, quantization, export or runtime code from the legacy `qmini/` package.
2. Do not read or overwrite legacy Q2-v2 checkpoints, tokenizer files or Drive artifacts.
3. Use the five-symbol alphabet `{-1.5, -0.5, 0, +0.5, +1.5}`; zero is a genuine no-connection symbol.
4. Five symbols require at least three bits in fixed-width packing.
5. Deployed inference may combine values only with integer addition/subtraction; coefficient and nonlinear lookup tables are allowed, runtime multiplication is not.
6. Do not add an FP16/BF16 warm-up training stage by default. Training precision must be decided independently and documented.
7. Train an independent random initialization on approved, provenance-checked English data.

See [the full Add/Sub v1 architecture contract](../docs/ADD_SUB_V1_SPEC.md).

The legacy Q-Mini Light model remains under `qmini/` and is not replaced by this package.
