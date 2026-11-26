from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _step_from_name(path: Path, prefix: str) -> int:
    match = re.fullmatch(re.escape(prefix) + r"-step-(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"malformed checkpoint filename: {path.name}")
    step = int(match.group(1))
    if step < 1:
        raise ValueError(f"checkpoint step must be positive: {path.name}")
    return step


def latest_checkpoint(
    folder: str | Path, prefix: str
) -> tuple[Path | None, int | None]:
    """Return the checkpoint with the greatest numeric step, never lexicographic order."""
    if not isinstance(prefix, str) or not prefix or Path(prefix).name != prefix:
        raise ValueError("checkpoint prefix must be a non-empty filename component")
    directory = Path(folder)
    candidates = list(directory.glob(f"{prefix}-step-*.pt"))
    if not candidates:
        return None, None

    by_step: dict[int, Path] = {}
    for path in candidates:
        if not path.is_file():
            raise ValueError(f"checkpoint candidate is not a regular file: {path}")
        step = _step_from_name(path, prefix)
        if step in by_step:
            raise ValueError(f"duplicate numeric checkpoint step {step}: {by_step[step]} and {path}")
        by_step[step] = path

    step = max(by_step)
    return by_step[step], step


def resolve_checkpoint(
    folder: str | Path,
    prefix: str,
    metrics_filename: str,
    explicit: str | Path | None = None,
) -> tuple[Path | None, int | None]:
    """Select an explicit or newest checkpoint and reject unsafe resume states."""
    directory = Path(folder)
    latest_path, latest_step = latest_checkpoint(directory, prefix)

    if explicit is not None:
        selected = Path(explicit)
        if not selected.is_file():
            raise FileNotFoundError(f"requested checkpoint does not exist: {selected}")
        selected_step = _step_from_name(selected, prefix)
        if latest_step is not None and latest_step > selected_step:
            raise ValueError(
                f"newer checkpoint step {latest_step} exists; refusing to resume from older "
                f"step {selected_step}"
            )
        return selected, selected_step

    if latest_path is None and (directory / metrics_filename).is_file():
        raise FileNotFoundError(
            f"metrics file exists but no checkpoint matching {prefix}-step-*.pt was found "
            f"in {directory}; refusing to start over a potentially incomplete run"
        )
    return latest_path, latest_step



def verify_completed_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    prefix: str,
    metrics_path: str | Path,
    expected_checkpoint: dict[str, Any],
    expected_metrics: dict[str, Any],
    expected_run_contract: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    """Read-only verification before a notebook skips a completed training stage."""
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{run_name} checkpoint must contain a mapping.")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be a non-empty string.")

    path = Path(checkpoint_path)
    filename_step = _step_from_name(path, prefix)
    checkpoint_step = checkpoint.get("step")
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 1
    ):
        raise ValueError(f"{run_name} checkpoint has an invalid optimizer step.")
    if checkpoint_step != filename_step:
        raise ValueError(
            f"{run_name} checkpoint filename step {filename_step} does not match "
            f"its stored optimizer step {checkpoint_step}."
        )

    mismatches = [
        key for key, expected in expected_checkpoint.items()
        if checkpoint.get(key) != expected
    ]
    saved_contract = checkpoint.get("run_contract")
    if not isinstance(saved_contract, dict):
        mismatches.append("run_contract")
        saved_contract = {}
    static_saved_contract = {
        key: value for key, value in saved_contract.items()
        if key != "planned_steps"
    }
    static_expected_contract = {
        key: value for key, value in expected_run_contract.items()
        if key != "planned_steps"
    }
    if static_saved_contract != static_expected_contract:
        mismatches.append("run_contract")
    if mismatches:
        raise ValueError(
            f"{run_name} completed checkpoint is incompatible with the current "
            "dataset, tokenizer, model or training configuration: "
            + ", ".join(sorted(set(mismatches)))
        )

    schedule_horizon = checkpoint.get(
        "schedule_total_steps",
        saved_contract.get("planned_steps", checkpoint.get("planned_steps")),
    )
    if (
        isinstance(schedule_horizon, bool)
        or not isinstance(schedule_horizon, int)
        or schedule_horizon < checkpoint_step
    ):
        raise ValueError(f"{run_name} checkpoint has an invalid schedule horizon.")

    metric_file = Path(metrics_path)
    if not metric_file.is_file():
        raise FileNotFoundError(
            f"{run_name} checkpoint exists without its metrics file: {metric_file}"
        )

    rows: list[dict[str, Any]] = []
    previous_step = 0
    previous_horizon = 0
    for line_number, line in enumerate(
        metric_file.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{run_name} metrics file has invalid JSON at line {line_number}."
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"{run_name} metrics row {line_number} must be a JSON object."
            )

        row_step = row.get("step")
        if (
            isinstance(row_step, bool)
            or not isinstance(row_step, int)
            or row_step < 1
            or row_step <= previous_step
        ):
            raise ValueError(
                f"{run_name} metrics steps must be positive and strictly increasing."
            )
        if row_step > checkpoint_step:
            raise ValueError(
                f"{run_name} metrics contain step {row_step} newer than selected "
                f"checkpoint step {checkpoint_step}; refusing to report completion."
            )
        for key, expected in expected_metrics.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"{run_name} metrics lineage mismatch for {key!r} at step {row_step}."
                )

        row_contract = row.get("run_contract")
        if not isinstance(row_contract, dict):
            raise ValueError(
                f"{run_name} metrics row {row_step} is missing run_contract."
            )
        row_static_contract = {
            key: value for key, value in row_contract.items()
            if key != "planned_steps"
        }
        if row_static_contract != static_expected_contract:
            raise ValueError(
                f"{run_name} metrics run_contract mismatch at step {row_step}."
            )

        row_horizon = row.get("schedule_total_steps")
        if (
            isinstance(row_horizon, bool)
            or not isinstance(row_horizon, int)
            or row_horizon < row_step
            or row_horizon < previous_horizon
        ):
            raise ValueError(
                f"{run_name} metrics contain an invalid or decreasing schedule horizon."
            )
        previous_step = row_step
        previous_horizon = row_horizon
        rows.append(row)

    if not rows or rows[-1]["step"] != checkpoint_step:
        last_step = rows[-1]["step"] if rows else None
        raise ValueError(
            f"{run_name} metrics do not validate selected checkpoint step "
            f"{checkpoint_step} (latest metrics step: {last_step})."
        )
    if rows[-1]["schedule_total_steps"] != schedule_horizon:
        raise ValueError(
            f"{run_name} final metrics schedule does not match the selected checkpoint."
        )

    return {
        "status": "validated",
        "checkpoint": str(path),
        "step": checkpoint_step,
        "schedule_total_steps": schedule_horizon,
        "metrics": str(metric_file),
    }
