from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read evaluation report: {target}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Evaluation report must be a JSON object: {target}")
    return value


def _read_metrics(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Training metrics not found: {target}")
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid training metrics JSON at {target}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Training metrics entry at {target}:{line_number} must be an object."
                )
            records.append(record)
    if not records:
        raise ValueError("Training metrics file is empty.")
    return records


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _check(name: str, passed: bool, observed: Any, required: Any, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
        "detail": detail,
    }


def assess_release_readiness(
    metrics: list[dict[str, Any]],
    task_report: dict[str, Any],
    conversation_report: dict[str, Any],
    *,
    min_validation_improvement: float = 0.01,
    min_task_examples: int = 20,
    min_conversation_turns: int = 20,
    min_task_pass_rate: float = 0.25,
    min_conversation_pass_rate: float = 0.25,
    min_unique_answer_fraction: float = 0.50,
    max_largest_identical_answer_fraction: float = 0.35,
) -> dict[str, Any]:
    """Fail closed unless training and held-out task/chat quality checks pass.

    Passing is a release floor, not proof of human-level intelligence or factuality.
    """
    limits = {
        "min_validation_improvement": min_validation_improvement,
        "min_task_pass_rate": min_task_pass_rate,
        "min_conversation_pass_rate": min_conversation_pass_rate,
        "min_unique_answer_fraction": min_unique_answer_fraction,
        "max_largest_identical_answer_fraction": max_largest_identical_answer_fraction,
    }
    for name, value in limits.items():
        number = _number(value, name)
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    for name, value in (
        ("min_task_examples", min_task_examples),
        ("min_conversation_turns", min_conversation_turns),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    checks: list[dict[str, Any]] = []

    # Validate the complete metrics chronology and finite training signals.
    previous_step = 0
    for index, record in enumerate(metrics, 1):
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= previous_step:
            raise ValueError("Training metric steps must be positive and strictly increasing.")
        previous_step = step
        for key in ("train_loss", "train_cross_entropy", "gradient_norm", "learning_rate"):
            _number(record.get(key), f"metrics[{index}].{key}")

    latest = metrics[-1]
    phase_start = latest.get("schedule_start_step", 0)
    if isinstance(phase_start, bool) or not isinstance(phase_start, int) or phase_start < 0:
        raise ValueError("Latest schedule_start_step must be a non-negative integer.")
    phase_metrics = [
        record for record in metrics
        if record.get("schedule_start_step", 0) == phase_start
        and "eval_loss" in record
        and isinstance(record.get("validation_sha256"), str)
    ]
    eval_losses = [
        _number(record["eval_loss"], "validation eval_loss")
        for record in phase_metrics
    ]
    validation_hashes = {
        record.get("validation_sha256")
        for record in phase_metrics
    }
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in validation_hashes
    ):
        raise ValueError("Validation dataset fingerprint must be a lowercase SHA-256.")
    if len(validation_hashes) > 1:
        raise ValueError("Validation dataset fingerprint changed within the active training phase.")
    if len(eval_losses) < 2:
        checks.append(_check(
            "validation_learning",
            False,
            {"evaluation_points": len(eval_losses), "phase_start_step": phase_start},
            {"minimum_evaluation_points": 2, "minimum_relative_improvement": min_validation_improvement},
            "At least two held-out validation evaluations in the active data/schedule phase are required.",
        ))
    else:
        first_loss, last_loss = eval_losses[0], eval_losses[-1]
        improvement = (first_loss - last_loss) / max(abs(first_loss), 1e-12)
        checks.append(_check(
            "validation_learning",
            improvement >= min_validation_improvement,
            {
                "first_eval_loss": first_loss,
                "last_eval_loss": last_loss,
                "relative_improvement": improvement,
                "evaluation_points": len(eval_losses),
                "phase_start_step": phase_start,
            },
            {"minimum_relative_improvement": min_validation_improvement, "minimum_evaluation_points": 2},
            "Validation cross-entropy must improve in the active dataset phase; training loss alone is insufficient.",
        ))

    # A release evaluation must be based on genuine, independently held-out files.
    task_count = task_report.get("examples")
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count < 0:
        raise ValueError("Task evaluation report has an invalid examples count.")
    conversation_count = conversation_report.get("turns")
    if (
        isinstance(conversation_count, bool)
        or not isinstance(conversation_count, int)
        or conversation_count < 0
    ):
        raise ValueError("Conversation evaluation report has an invalid turns count.")
    if task_report.get("format") != "qmini-task-evaluation-v2":
        raise ValueError("Task report must use qmini-task-evaluation-v2.")
    if conversation_report.get("format") != "qmini-conversation-evaluation-v1":
        raise ValueError("Conversation report must use qmini-conversation-evaluation-v1.")
    task_hash = task_report.get("dataset_sha256")
    conversation_hash = conversation_report.get("dataset_sha256")
    for label, value in (("Task", task_hash), ("Conversation", conversation_hash)):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"{label} evaluation report is missing a valid dataset SHA-256.")
    checks.append(_check(
        "independent_held_out_sets",
        task_hash != conversation_hash,
        {"task_dataset_sha256": task_hash, "conversation_dataset_sha256": conversation_hash},
        "Different task and conversation dataset SHA-256 values",
        "Task and multi-turn conversation evaluations must use separate held-out files.",
    ))

    task_rate = _number(task_report.get("task_pass_rate"), "task_pass_rate")
    conversation_rate = _number(
        conversation_report.get("task_pass_rate"), "conversation task_pass_rate"
    )
    checks.append(_check(
        "task_accuracy",
        task_count >= min_task_examples and task_rate >= min_task_pass_rate,
        {"examples": task_count, "task_pass_rate": task_rate},
        {"minimum_examples": min_task_examples, "minimum_pass_rate": min_task_pass_rate},
        "Task accuracy must clear the configured floor on a held-out test set.",
    ))
    checks.append(_check(
        "conversation_accuracy",
        conversation_count >= min_conversation_turns
        and conversation_rate >= min_conversation_pass_rate,
        {"turns": conversation_count, "task_pass_rate": conversation_rate},
        {"minimum_turns": min_conversation_turns, "minimum_pass_rate": min_conversation_pass_rate},
        "Multi-turn conversation accuracy must clear the configured floor.",
    ))

    for label, report in (("task", task_report), ("conversation", conversation_report)):
        diversity = report.get("output_diversity")
        if not isinstance(diversity, dict):
            raise ValueError(
                f"{label} evaluation report has no output_diversity diagnostics; "
                "regenerate it with the current evaluator."
            )
        examples = diversity.get("examples")
        unique_count = diversity.get("unique_normalized_answers")
        largest_count = diversity.get("largest_identical_answer_count")
        unique_fraction = _number(
            diversity.get("unique_answer_fraction"),
            f"{label} unique_answer_fraction",
        )
        largest_fraction = _number(
            diversity.get("largest_identical_answer_fraction"),
            f"{label} largest_identical_answer_fraction",
        )
        empty_count = diversity.get("empty_answer_count")
        all_identical = diversity.get("all_outputs_identical")
        expected_count = task_count if label == "task" else conversation_count
        if (
            isinstance(examples, bool) or not isinstance(examples, int) or examples < 1
            or examples != expected_count
            or isinstance(unique_count, bool) or not isinstance(unique_count, int) or not 1 <= unique_count <= examples
            or isinstance(largest_count, bool) or not isinstance(largest_count, int) or not 1 <= largest_count <= examples
            or isinstance(empty_count, bool) or not isinstance(empty_count, int) or not 0 <= empty_count <= examples
            or not isinstance(all_identical, bool)
            or not 0.0 <= unique_fraction <= 1.0
            or not 0.0 <= largest_fraction <= 1.0
            or not math.isclose(unique_fraction, unique_count / examples, rel_tol=0.0, abs_tol=1e-9)
            or not math.isclose(largest_fraction, largest_count / examples, rel_tol=0.0, abs_tol=1e-9)
            or all_identical != (unique_count == 1)
        ):
            raise ValueError(f"{label} output_diversity fields are malformed or inconsistent.")
        enough_examples = examples >= (
            min_task_examples if label == "task" else min_conversation_turns
        )
        passed = (
            enough_examples
            and not all_identical
            and empty_count == 0
            and unique_fraction >= min_unique_answer_fraction
            and largest_fraction <= max_largest_identical_answer_fraction
        )
        checks.append(_check(
            f"{label}_repetition_collapse",
            passed,
            {
                "examples": examples,
                "unique_answer_fraction": unique_fraction,
                "largest_identical_answer_fraction": largest_fraction,
                "empty_answer_count": empty_count,
                "all_outputs_identical": all_identical,
            },
            {
                "minimum_examples": min_task_examples if label == "task" else min_conversation_turns,
                "minimum_unique_answer_fraction": min_unique_answer_fraction,
                "maximum_largest_identical_answer_fraction": max_largest_identical_answer_fraction,
                "empty_answer_count": 0,
                "all_outputs_identical": False,
            },
            "Identical/empty-answer collapse blocks readiness even when loss falls.",
        ))

    ready = all(item["passed"] for item in checks)
    return {
        "format": "qmini-release-readiness-v1",
        "status": "ready_for_review" if ready else "not_ready",
        "ready": ready,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "limitations": [
            "Passing this gate is not proof of human-level conversation, factuality, reasoning, or safety.",
            "Open-ended replies still require human review and independently verified references.",
            "The default thresholds are conservative engineering floors and should be calibrated to a fixed, representative English benchmark.",
        ],
    }


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed held-out quality gate for a Q-Mini training run."
    )
    parser.add_argument("--metrics", required=True, help="Training metrics.jsonl")
    parser.add_argument("--tasks-report", required=True, help="Current eval-tasks JSON report")
    parser.add_argument("--conversations-report", required=True, help="Current eval-conversations JSON report")
    parser.add_argument("--output", required=True, help="Readiness JSON output path")
    parser.add_argument("--min-validation-improvement", type=float, default=0.01)
    parser.add_argument("--min-task-examples", type=int, default=20)
    parser.add_argument("--min-conversation-turns", type=int, default=20)
    parser.add_argument("--min-task-pass-rate", type=float, default=0.25)
    parser.add_argument("--min-conversation-pass-rate", type=float, default=0.25)
    parser.add_argument("--min-unique-answer-fraction", type=float, default=0.50)
    parser.add_argument("--max-largest-identical-answer-fraction", type=float, default=0.35)
    args = parser.parse_args(argv)

    report = assess_release_readiness(
        _read_metrics(args.metrics),
        _read_json(args.tasks_report),
        _read_json(args.conversations_report),
        min_validation_improvement=args.min_validation_improvement,
        min_task_examples=args.min_task_examples,
        min_conversation_turns=args.min_conversation_turns,
        min_task_pass_rate=args.min_task_pass_rate,
        min_conversation_pass_rate=args.min_conversation_pass_rate,
        min_unique_answer_fraction=args.min_unique_answer_fraction,
        max_largest_identical_answer_fraction=args.max_largest_identical_answer_fraction,
    )
    _atomic_json_save(report, Path(args.output))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Q-Mini readiness: {report['status']}")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
