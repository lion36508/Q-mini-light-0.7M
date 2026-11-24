from pathlib import Path

import pytest

from qmini_addsub.data import stable_split
from scripts.prepare_qmini_addsub_data import _article_text_chunks, _write_finewiki_rows


def _rows_for_all_splits():
    wanted = {"train": 0, "validation": 0, "test": 0}
    rows = []
    candidate = 1
    while any(count < 4 for count in wanted.values()):
        page_id = str(candidate)
        split = stable_split(page_id, seed="qmini-finewiki-en-2025-08")
        if wanted[split] < 4:
            wanted[split] += 1
            body = (
                f"This article explains the subject for page {page_id}. "
                "It contains verified-style encyclopedic prose with enough words "
                "to survive text quality filtering. " * 4
            )
            rows.append({
                "page_id": page_id,
                "title": f"FineWiki Test Article {page_id}",
                "text": body,
                "in_language": "en",
                "wikiname": "enwiki",
            })
        candidate += 1
    return rows


def test_finewiki_chunker_preserves_content_and_respects_boundaries():
    body = ("First paragraph with useful factual text. " * 18) + "\n\n"
    body += ("Second paragraph with different factual text. " * 18) + "\n\n"
    body += ("Third paragraph with a long tail. " * 40)
    chunks = _article_text_chunks("Chunking Test", body, max_chars=512)
    assert len(chunks) >= 3
    assert all(chunk.startswith("# Chunking Test") for chunk in chunks)
    joined = " ".join(chunks)
    for marker in ("First paragraph", "Second paragraph", "Third paragraph", "long tail"):
        assert marker in joined
    assert all(len(chunk) <= 560 for chunk in chunks)


def test_finewiki_split_is_by_whole_article_and_records_hashes(tmp_path):
    rows = _rows_for_all_splits()
    rows.append(dict(rows[0]))  # Duplicate page ID must not create a second article.
    rows.append({
        "page_id": "not-english",
        "title": "Ignored",
        "text": "Not English " * 80,
        "in_language": "de",
        "wikiname": "dewiki",
    })

    report = _write_finewiki_rows(rows, tmp_path)
    assert set(report["splits"]) == {"train", "validation", "test"}
    assert report["dropped"]["duplicate_page_id"] == 1
    assert report["dropped"]["not_english_enwiki"] == 1

    split_text = {
        name: (tmp_path / f"pretrain_{name}.txt").read_text(encoding="utf-8")
        for name in ("train", "validation", "test")
    }
    for name, entry in report["splits"].items():
        assert entry["articles"] == 4
        assert entry["chunks"] >= 4
        assert entry["bytes"] > 0
        assert len(entry["sha256"]) == 64
        own_titles = [
            row["title"] for row in rows
            if stable_split(row["page_id"], seed="qmini-finewiki-en-2025-08") == name
            and row["in_language"] == "en"
        ]
        for title in own_titles:
            assert title in split_text[name]
            for other_name, other_text in split_text.items():
                if other_name != name:
                    assert title not in other_text

    with pytest.raises(FileExistsError, match="fresh versioned data folder"):
        _write_finewiki_rows(_rows_for_all_splits(), tmp_path)


def test_finewiki_rejects_empty_split_instead_of_silently_training(tmp_path):
    train_ids = []
    for page_id in range(1, 1000):
        if stable_split(str(page_id), seed="qmini-finewiki-en-2025-08") == "train":
            train_ids.append(str(page_id))
            if len(train_ids) == 5:
                break
    rows = [{
        "page_id": page_id,
        "title": f"Train-only {page_id}",
        "text": "Useful explanatory prose. " * 20,
        "in_language": "en",
        "wikiname": "enwiki",
    } for page_id in train_ids]
    with pytest.raises(ValueError, match="split is empty"):
        _write_finewiki_rows(rows, tmp_path)
