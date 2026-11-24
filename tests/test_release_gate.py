import pytest

from qmini.release_gate import assess_release_readiness


def _metrics(*, improvement=True):
    losses = [4.0, 3.8] if improvement else [4.0, 4.2]
    return [
        {
            "step": 100,
            "schedule_start_step": 0,
            "train_loss": 4.5,
            "train_cross_entropy": 4.4,
            "gradient_norm": 0.8,
            "learning_rate": 0.0001,
        },
        {
            "step": 200,
            "schedule_start_step": 0,
            "train_loss": 4.1,
            "train_cross_entropy": 4.0,
            "gradient_norm": 0.6,
            "learning_rate": 0.00005,
            "eval_loss": losses[0],
            "validation_sha256": "c" * 64,
        },
        {
            "step": 300,
            "schedule_start_step": 0,
            "train_loss": 3.9,
            "train_cross_entropy": 3.8,
            "gradient_norm": 0.5,
            "learning_rate": 0.00001,
            "eval_loss": losses[1],
            "validation_sha256": "c" * 64,
        },
    ]


def _diversity(examples=20):
    return {
        "examples": examples,
        "unique_normalized_answers": 15,
        "unique_answer_fraction": 15 / examples,
        "largest_identical_answer_count": 2,
        "largest_identical_answer_fraction": 2 / examples,
        "empty_answer_count": 0,
        "all_outputs_identical": False,
    }


def _reports():
    return (
        {
            "format": "qmini-task-evaluation-v2",
            "examples": 20,
            "task_pass_rate": 0.6,
            "dataset_sha256": "a" * 64,
            "output_diversity": _diversity(),
        },
        {
            "format": "qmini-conversation-evaluation-v1",
            "turns": 20,
            "task_pass_rate": 0.55,
            "dataset_sha256": "b" * 64,
            "output_diversity": _diversity(),
        },
    )


def test_release_gate_accepts_learning_and_independent_diverse_heldout_outputs():
    task, conversation = _reports()
    report = assess_release_readiness(_metrics(), task, conversation)
    assert report["ready"] is True
    assert report["status"] == "ready_for_review"
    assert report["checks_passed"] == report["checks_total"]


def test_release_gate_rejects_no_validation_learning():
    task, conversation = _reports()
    report = assess_release_readiness(
        _metrics(improvement=False), task, conversation
    )
    check = next(item for item in report["checks"] if item["name"] == "validation_learning")
    assert check["passed"] is False
    assert report["ready"] is False


@pytest.mark.parametrize("report_kind", ["task", "conversation"])
def test_release_gate_rejects_collapsed_outputs_even_when_accuracy_is_high(report_kind):
    task, conversation = _reports()
    collapsed = {
        "examples": 20,
        "unique_normalized_answers": 1,
        "unique_answer_fraction": 0.05,
        "largest_identical_answer_count": 20,
        "largest_identical_answer_fraction": 1.0,
        "empty_answer_count": 0,
        "all_outputs_identical": True,
    }
    if report_kind == "task":
        task["output_diversity"] = collapsed
    else:
        conversation["output_diversity"] = collapsed
    report = assess_release_readiness(_metrics(), task, conversation)
    failed_name = f"{report_kind}_repetition_collapse"
    assert next(item for item in report["checks"] if item["name"] == failed_name)["passed"] is False
    assert report["ready"] is False


def test_release_gate_rejects_training_set_reused_for_both_heldout_reports():
    task, conversation = _reports()
    conversation["dataset_sha256"] = task["dataset_sha256"]
    report = assess_release_readiness(_metrics(), task, conversation)
    check = next(item for item in report["checks"] if item["name"] == "independent_held_out_sets")
    assert check["passed"] is False
    assert report["ready"] is False


def test_release_gate_rejects_insufficient_heldout_sample_count():
    task, conversation = _reports()
    task["examples"] = 4
    task["output_diversity"] = {
        "examples": 4,
        "unique_normalized_answers": 3,
        "unique_answer_fraction": 0.75,
        "largest_identical_answer_count": 2,
        "largest_identical_answer_fraction": 0.5,
        "empty_answer_count": 0,
        "all_outputs_identical": False,
    }
    report = assess_release_readiness(_metrics(), task, conversation)
    check = next(item for item in report["checks"] if item["name"] == "task_accuracy")
    assert check["passed"] is False
    assert report["ready"] is False


def test_release_gate_rejects_inconsistent_diversity_report():
    task, conversation = _reports()
    task["output_diversity"]["unique_normalized_answers"] = 1
    with pytest.raises(ValueError, match="malformed or inconsistent"):
        assess_release_readiness(_metrics(), task, conversation)


def test_release_gate_ignores_legacy_eval_rows_without_dataset_hash_until_new_evals_exist():
    task, conversation = _reports()
    legacy = {
        "step": 50,
        "schedule_start_step": 0,
        "train_loss": 5.0,
        "train_cross_entropy": 5.0,
        "gradient_norm": 1.0,
        "learning_rate": 0.0001,
        "eval_loss": 5.0,
    }
    report = assess_release_readiness(
        [legacy, *_metrics()], task, conversation
    )
    assert report["ready"] is True


def test_release_gate_requires_two_new_evals_for_the_active_phase():
    task, conversation = _reports()
    metrics = _metrics()
    metrics[-1]["schedule_start_step"] = 300
    report = assess_release_readiness(metrics, task, conversation)
    check = next(item for item in report["checks"] if item["name"] == "validation_learning")
    assert check["passed"] is False
    assert report["ready"] is False
