from __future__ import annotations

from scripts.prepare_qmini_addsub_data import _select_oasst_conversation_path


def _message(message_id: str, parent_id: str | None, role: str) -> dict:
    return {
        "message_id": message_id,
        "parent_id": parent_id,
        "role": role,
        "text": f"Text for {message_id}.",
    }


def _ids(path):
    return None if path is None else [item["message_id"] for item in path]


def test_oasst_tree_selects_longest_complete_branch_independent_of_row_order():
    root = _message("u0", None, "prompter")
    first_answer = _message("a0", "u0", "assistant")
    short_user = _message("u-short", "a0", "prompter")
    short_answer = _message("a-short", "u-short", "assistant")
    long_user = _message("u-long", "a0", "prompter")
    long_answer = _message("a-long", "u-long", "assistant")
    final_user = _message("u-final", "a-long", "prompter")
    final_answer = _message("a-final", "u-final", "assistant")
    messages = [
        root, first_answer, short_user, short_answer,
        long_user, long_answer, final_user, final_answer,
    ]

    expected = ["u0", "a0", "u-long", "a-long", "u-final", "a-final"]
    path, reason, branching = _select_oasst_conversation_path(messages)
    assert _ids(path) == expected
    assert reason is None
    assert branching is True

    reversed_path, reversed_reason, reversed_branching = _select_oasst_conversation_path(
        list(reversed(messages))
    )
    assert _ids(reversed_path) == expected
    assert reversed_reason is None
    assert reversed_branching is True


def test_oasst_tree_equal_length_branches_use_stable_message_id_tie_break():
    root = _message("u0", None, "prompter")
    later_id = _message("a-z", "u0", "assistant")
    earlier_id = _message("a-a", "u0", "assistant")

    for messages in ([root, later_id, earlier_id], [earlier_id, root, later_id]):
        path, reason, branching = _select_oasst_conversation_path(list(messages))
        assert _ids(path) == ["u0", "a-a"]
        assert reason is None
        assert branching is True


def test_oasst_tree_rejects_duplicate_message_ids_and_multiple_roots():
    duplicate = [
        _message("u0", None, "prompter"),
        _message("a0", "u0", "assistant"),
        _message("a0", "u0", "assistant"),
    ]
    path, reason, _ = _select_oasst_conversation_path(duplicate)
    assert path is None
    assert reason == "duplicate_message_id"

    multiple_roots = [
        _message("u0", None, "prompter"),
        _message("u1", None, "prompter"),
        _message("a1", "u1", "assistant"),
    ]
    path, reason, _ = _select_oasst_conversation_path(multiple_roots)
    assert path is None
    assert reason == "invalid_root_count"


def test_oasst_tree_drops_orphaned_or_incomplete_paths_instead_of_fabricating_turns():
    orphaned = [
        _message("u0", None, "prompter"),
        _message("a0", "missing-parent", "assistant"),
    ]
    path, reason, branching = _select_oasst_conversation_path(orphaned)
    assert path is None
    assert reason == "orphaned_or_cyclic_tree"
    assert branching is False

    incomplete = [
        _message("u0", None, "prompter"),
        _message("a0", "u0", "assistant"),
        _message("u1", "a0", "prompter"),
    ]
    path, reason, _ = _select_oasst_conversation_path(incomplete)
    assert path is None
    assert reason == "no_complete_assistant_path"


def test_prepare_data_module_import_does_not_require_optional_network_parquet_packages():
    # Importing the module exposes pure data-cleaning/tree helpers. The optional
    # Hugging Face and Parquet packages are required only when a source download
    # or Parquet read is actually requested.
    import scripts.prepare_qmini_addsub_data as prepare

    assert callable(prepare.load_oasst_conversations)
    assert callable(prepare.load_dolly_conversations)



def test_oasst_tree_dynamic_program_selects_longest_suffix_across_many_branches():
    root = _message("u-root", None, "prompter")
    messages = [root]
    expected_path = ["u-root"]

    # Add many assistant branches with different lengths. The best branch is
    # deliberately not first in input order, and the path is long enough to
    # exercise the non-recursive post-order traversal.
    for branch_index in range(96):
        assistant_id = f"a-{branch_index:03d}"
        messages.append(_message(assistant_id, "u-root", "assistant"))
        parent_id = assistant_id
        branch_length = (branch_index % 19) + 1
        for depth in range(branch_length):
            user_id = f"u-{branch_index:03d}-{depth:02d}"
            assistant_reply_id = f"r-{branch_index:03d}-{depth:02d}"
            messages.append(_message(user_id, parent_id, "prompter"))
            messages.append(_message(assistant_reply_id, user_id, "assistant"))
            parent_id = assistant_reply_id

        if branch_index == 18:
            expected_path = ["u-root", assistant_id]
            for depth in range(branch_length):
                expected_path.extend([
                    f"u-{branch_index:03d}-{depth:02d}",
                    f"r-{branch_index:03d}-{depth:02d}",
                ])

    path, reason, branching = _select_oasst_conversation_path(messages)

    assert _ids(path) == expected_path
    assert reason is None
    assert branching is True


def test_oasst_tree_dynamic_program_uses_lexicographic_tie_break_at_first_divergence():
    root = _message("u0", None, "prompter")
    messages = [root]
    for branch_id in ("z-branch", "a-branch", "m-branch"):
        messages.extend([
            _message(branch_id, "u0", "assistant"),
            _message(f"u-{branch_id}", branch_id, "prompter"),
            _message(f"r-{branch_id}", f"u-{branch_id}", "assistant"),
        ])

    path, reason, branching = _select_oasst_conversation_path(list(reversed(messages)))
    assert _ids(path) == ["u0", "a-branch", "u-a-branch", "r-a-branch"]
    assert reason is None
    assert branching is True
