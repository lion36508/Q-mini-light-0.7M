from __future__ import annotations

import re
from pathlib import Path


_SUPPORTED_PREFIXES = {"checkpoint", "sft-checkpoint"}


def checkpoint_step(path: str | Path, prefix: str) -> int:
    """Return the numeric optimizer step encoded in a Q-Mini checkpoint filename."""
    if prefix not in _SUPPORTED_PREFIXES:
        raise ValueError(f"Unsupported checkpoint prefix: {prefix!r}.")
    name = Path(path).name
    match = re.fullmatch(rf"{re.escape(prefix)}-step-(\\d+)\\.pt", name)
    if match is None:
        raise ValueError(
            f"Malformed {prefix} checkpoint filename {name!r}; "
            f"expected {prefix}-step-<integer>.pt."
        )
    return int(match.group(1))


def latest_checkpoint(
    directory: str | Path,
    prefix: str,
) -> Path | None:
    """Select the checkpoint with the largest numeric step, or None when absent.

    Ambiguous names and duplicate numeric steps fail closed instead of guessing.
    The caller remains responsible for the trainer's full run-contract validation.
    """
    if prefix not in _SUPPORTED_PREFIXES:
        raise ValueError(f"Unsupported checkpoint prefix: {prefix!r}.")
    folder = Path(directory)
    if not folder.is_dir():
        return None

    candidates = sorted(folder.glob(f"{prefix}-step-*.pt"))
    if not candidates:
        return None

    by_step: dict[int, Path] = {}
    for candidate in candidates:
        step = checkpoint_step(candidate, prefix)
        if step in by_step:
            raise ValueError(
                f"Ambiguous {prefix} checkpoints for step {step}: "
                f"{by_step[step].name} and {candidate.name}."
            )
        by_step[step] = candidate
    return by_step[max(by_step)]
