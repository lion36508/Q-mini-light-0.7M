from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


OASST_REQUIRED_FIELDS = {
    "message_id",
    "parent_id",
    "text",
    "role",
    "lang",
    "review_result",
    "deleted",
    "rank",
    "synthetic",
    "message_tree_id",
    "tree_state",
}

EMAIL_RE = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_clean_english_text(value: Any, min_chars: int = 2) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if len(text) < min_chars or not any(char.isalpha() for char in text):
        return False
    if EMAIL_RE.search(text):
        return False
    return True


def accept_oasst1_message(row: dict[str, Any]) -> tuple[bool, str]:
    """Fail-closed source filter: reviewed human English OASST1 only."""
    missing = OASST_REQUIRED_FIELDS - set(row)
    if missing:
        return False, "missing_fields"
    if row["lang"] != "en":
        return False, "non_english"
    if row["tree_state"] != "ready_for_export":
        return False, "not_ready"
    if row["review_result"] is not True:
        return False, "unreviewed"
    if row["deleted"] is not False:
        return False, "deleted"
    if row["synthetic"] is not False:
        return False, "synthetic"
    role = row["role"]
    if role == "assistant":
        rank = row["rank"]
        if rank is None:
            return False, "missing_rank"
        try:
            if int(rank) != 0:
                return False, "non_top_rank"
        except (TypeError, ValueError):
            return False, "invalid_rank"
    elif role != "prompter":
        return False, "other_role"
    if not is_clean_english_text(row["text"]):
        return False, "empty_or_sensitive"
    return True, "accepted"


def stable_split(group_id: str, seed: str = "qmini-addsub-v1") -> str:
    """Deterministically split whole documents/trees without row leakage."""
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("A non-empty group ID is required for leakage-safe splitting.")
    digest = hashlib.sha256((seed + ":" + group_id).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> tuple[int, str]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count, sha256_file(destination)
