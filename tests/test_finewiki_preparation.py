import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import zstandard

from qmini_addsub.data import stable_split
from scripts import prepare_qmini_addsub_data as prepare_module
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
    duplicate_content = dict(rows[1])
    duplicate_content["page_id"] = "duplicate-content-id"
    duplicate_content["title"] = "Duplicate content under another page ID"
    rows.append(duplicate_content)
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
    assert report["dropped"]["duplicate_article_content"] == 1
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
            row["title"] for row in rows[:12]
            if stable_split(row["page_id"], seed="qmini-finewiki-en-2025-08") == name
            and row["in_language"] == "en"
        ]
        for title in own_titles:
            title_heading = f"# {title}\\n"
            assert title_heading in split_text[name]
            for other_name, other_text in split_text.items():
                if other_name != name:
                    assert title_heading not in other_text

    with pytest.raises(FileExistsError, match="fresh versioned data folder"):
        _write_finewiki_rows(_rows_for_all_splits(), tmp_path)



def test_pinned_finewiki_download_streams_jsonl_after_hash_verification(tmp_path, monkeypatch):
    rows = _rows_for_all_splits()
    compressed = zstandard.ZstdCompressor().compress(
        ("\n".join(json.dumps(row) for row in rows) + "\n").encode("utf-8")
    )
    archive = tmp_path / "fixture.jsonl.zst"
    archive.write_bytes(compressed)
    expected_sha = hashlib.sha256(compressed).hexdigest()
    monkeypatch.setattr(prepare_module, "FINEWIKI_SHA256", expected_sha)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **kwargs: str(archive)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    report = prepare_module.prepare_finewiki_sample(
        tmp_path / "cache", tmp_path / "prepared"
    )
    assert report["sha256"] == expected_sha
    assert report["license"] == "CC-BY-SA-4.0"
    assert report["synthetic_training_records_included"] == 0
    assert sum(item["articles"] for item in report["splits"].values()) == len(rows)


def test_pinned_finewiki_rejects_hash_mismatch_before_writing_splits(tmp_path, monkeypatch):
    archive = tmp_path / "wrong.jsonl.zst"
    archive.write_bytes(zstandard.ZstdCompressor().compress(b"{}\n"))
    monkeypatch.setattr(prepare_module, "FINEWIKI_SHA256", "0" * 64)

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **kwargs: str(archive)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="failed pinned SHA-256"):
        prepare_module.prepare_finewiki_sample(tmp_path / "cache", output)
    assert not list(output.glob("pretrain_*.txt")) if output.exists() else True

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
