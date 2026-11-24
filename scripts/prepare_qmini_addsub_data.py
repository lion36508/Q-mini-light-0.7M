from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import tempfile
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


from qmini_addsub.data import (
    accept_oasst1_message,
    normalize_training_text,
    sha256_file,
    stable_split,
    write_jsonl,
)


FINEWIKI_REPOSITORY = "agentlans/HuggingFaceFW-finewiki-sample"
FINEWIKI_REVISION = "e0bbfbb"
FINEWIKI_FILE = "en.jsonl.zst"
FINEWIKI_SHA256 = "57fc6edbc8472c5424b421a222860a1853274601f3b6e7ac2e5542af5956d04e"
FINEWIKI_LICENSE = "CC-BY-SA-4.0"
FINEWIKI_SNAPSHOT = "2025-08"
OASST_REPOSITORY = "OpenAssistant/oasst1"
OASST_REVISION = "fdf72ae0827c1cda404ff25b6603abec9e3399b"
OASST_FILE = "data/train-00000-of-00001-b42a775f407cee45.parquet"
OASST_SHA256 = "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e"
OASST_SIZE = 39_516_251
DOLLY_REPOSITORY = "databricks/databricks-dolly-15k"
DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DOLLY_FILE = "databricks-dolly-15k.jsonl"


def _article_text_chunks(title: str, body: str, max_chars: int = 28_000) -> list[str]:
    """Split one page into bounded, paragraph-aware chunks without losing long sections."""
    if max_chars < 512:
        raise ValueError("max_chars must be at least 512.")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    def flush() -> None:
        nonlocal current, current_size
        if current:
            chunks.append(f"# {title}\n\n" + "\n\n".join(current))
            current = []
            current_size = 0

    for original in paragraphs:
        paragraph = original
        while len(paragraph) > max_chars:
            cut = paragraph.rfind(" ", 0, max_chars)
            if cut < max_chars // 2:
                cut = max_chars
            piece, paragraph = paragraph[:cut].strip(), paragraph[cut:].strip()
            if piece:
                if current:
                    flush()
                chunks.append(f"# {title} (continued)\n\n{piece}")
        if not paragraph:
            continue
        extra = len(paragraph) + (2 if current else 0)
        if current and current_size + extra > max_chars:
            flush()
        current.append(paragraph)
        current_size += len(paragraph) + (2 if len(current) > 1 else 0)
    flush()
    return chunks


def _write_finewiki_rows(rows, output: Path) -> dict:
    """Clean and article-split an iterable of raw FineWiki rows."""
    output.mkdir(parents=True, exist_ok=True)
    split_paths = {
        name: output / f"pretrain_{name}.txt"
        for name in ("train", "validation", "test")
    }
    existing = [str(path) for path in split_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "FineWiki output already exists; use a fresh versioned data folder: "
            + ", ".join(existing)
        )

    counts = {
        name: {"articles": 0, "chunks": 0, "bytes": 0}
        for name in split_paths
    }
    dropped: dict[str, int] = defaultdict(int)
    seen_ids: set[str] = set()
    seen_content_hashes: set[str] = set()
    handles = {
        name: path.open("w", encoding="utf-8", newline="\n")
        for name, path in split_paths.items()
    }
    try:
        for row_number, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                dropped["not_object"] += 1
                continue
            if row.get("in_language") != "en" or row.get("wikiname") != "enwiki":
                dropped["not_english_enwiki"] += 1
                continue

            raw_id = row.get("page_id", row.get("id"))
            title, title_reason = normalize_training_text(
                row.get("title"), min_chars=1, max_chars=512
            )
            body = row.get("text")
            if raw_id is None or not str(raw_id).strip():
                dropped["missing_page_id"] += 1
                continue
            page_key = str(raw_id).strip()
            if title_reason != "accepted" or not isinstance(body, str) or len(body.strip()) < 200:
                dropped["empty_or_invalid_article"] += 1
                continue
            if page_key in seen_ids:
                dropped["duplicate_page_id"] += 1
                continue
            normalized_body = unicodedata.normalize(
                "NFC", body.replace("\r\n", "\n").replace("\r", "\n")
            )
            content_fingerprint = hashlib.sha256(
                " ".join(normalized_body.split()).encode("utf-8")
            ).hexdigest()
            if content_fingerprint in seen_content_hashes:
                seen_ids.add(page_key)
                dropped["duplicate_article_content"] += 1
                continue
            seen_ids.add(page_key)
            seen_content_hashes.add(content_fingerprint)

            split = stable_split(
                page_key, seed="qmini-finewiki-en-2025-08"
            )
            accepted_chunks = 0
            for chunk in _article_text_chunks(title, body):
                cleaned, reason = normalize_training_text(
                    chunk, min_chars=200, max_chars=32768
                )
                if reason != "accepted":
                    dropped[f"chunk_{reason}"] += 1
                    continue
                handles[split].write(cleaned + "\n\n")
                counts[split]["chunks"] += 1
                accepted_chunks += 1
            if accepted_chunks:
                counts[split]["articles"] += 1
            else:
                dropped["article_without_usable_chunks"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    for split, path in split_paths.items():
        counts[split]["bytes"] = path.stat().st_size
        counts[split]["sha256"] = sha256_file(path)
        if counts[split]["articles"] == 0 or counts[split]["chunks"] == 0:
            raise ValueError(f"FineWiki {split} split is empty.")

    return {
        "splits": counts,
        "dropped": dict(sorted(dropped.items())),
        "split_rule": (
            "stable SHA-256 page-ID split: 90% train, 5% validation, 5% test; "
            "all chunks from one Wikipedia page remain in one split"
        ),
    }


def prepare_finewiki_sample(cache: Path, output: Path) -> dict:
    """Prepare a hash-pinned 1% random English FineWiki sample, split by whole page."""
    from huggingface_hub import hf_hub_download

    compressed_path = Path(hf_hub_download(
        repo_id=FINEWIKI_REPOSITORY,
        filename=FINEWIKI_FILE,
        repo_type="dataset",
        revision=FINEWIKI_REVISION,
        cache_dir=str(cache / "huggingface"),
    ))
    actual_sha = sha256_file(compressed_path)
    if actual_sha != FINEWIKI_SHA256:
        raise ValueError(
            "FineWiki English sample failed pinned SHA-256 verification; "
            "refusing to train on an unverified source."
        )

    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError(
            "The Add/Sub data extra requires zstandard. Install with "
            "python -m pip install 'zstandard>=0.22'."
        ) from exc

    def iter_rows():
        with compressed_path.open("rb") as source:
            reader = zstandard.ZstdDecompressor().stream_reader(source)
            with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                for line_number, line in enumerate(text_stream, 1):
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"FineWiki JSONL line {line_number} is invalid."
                        ) from exc

    prepared = _write_finewiki_rows(iter_rows(), output)
    return {
        "repository": FINEWIKI_REPOSITORY,
        "revision": FINEWIKI_REVISION,
        "file": FINEWIKI_FILE,
        "sha256": actual_sha,
        "license": FINEWIKI_LICENSE,
        "upstream_dataset": "HuggingFaceFW/finewiki",
        "upstream_snapshot": FINEWIKI_SNAPSHOT,
        "upstream_snapshot_source": "Wikimedia Enterprise English Wikipedia main-namespace snapshot",
        "human_authored_source": True,
        "synthetic_training_records_included": 0,
        "sample_method": "uniform random 1% sample of the English FineWiki page corpus; pinned source file",
        "attribution": "Wikipedia contributors; FineWiki dataset (Guilherme Penedo); processed sample by agentlans",
        **prepared,
    }


def load_oasst_conversations(cache: Path) -> tuple[list[dict], dict]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    parquet_path = Path(
        hf_hub_download(
            repo_id=OASST_REPOSITORY,
            filename=OASST_FILE,
            repo_type="dataset",
            revision=OASST_REVISION,
            cache_dir=str(cache / "huggingface"),
        )
    )
    actual_sha = sha256_file(parquet_path)
    actual_size = parquet_path.stat().st_size
    if actual_sha != OASST_SHA256 or actual_size != OASST_SIZE:
        raise ValueError(
            "OASST1 parquet does not match its pinned source hash/size; "
            "refusing to prepare an unverified corpus."
        )

    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if not rows:
        raise ValueError("OASST1 source parquet is empty.")
    filtered = []
    dropped: dict[str, int] = defaultdict(int)
    for row in rows:
        accepted, reason = accept_oasst1_message(row)
        if accepted:
            filtered.append(row)
        else:
            dropped[reason] += 1

    by_tree: dict[str, list[dict]] = defaultdict(list)
    for row in filtered:
        tree_id = row.get("message_tree_id")
        if tree_id:
            by_tree[str(tree_id)].append(row)

    conversations = []
    incomplete_trees = 0
    for tree_id, messages in by_tree.items():
        by_id = {message["message_id"]: message for message in messages}
        children: dict[str, list[dict]] = defaultdict(list)
        root = None
        for message in messages:
            parent = message.get("parent_id")
            if parent is None:
                root = message
            else:
                children[str(parent)].append(message)
        if root is None or root.get("role") != "prompter":
            incomplete_trees += 1
            continue

        chain = []
        seen = set()
        current = root
        while current is not None:
            message_id = str(current["message_id"])
            if message_id in seen:
                break
            seen.add(message_id)
            chain.append(current)
            candidates = children.get(message_id, [])
            current = candidates[0] if candidates else None

        turns = []
        invalid_text = False
        for message in chain:
            role = "user" if message["role"] == "prompter" else "assistant"
            text, clean_reason = normalize_training_text(message["text"])
            if clean_reason != "accepted":
                dropped[f"normalized_text_{clean_reason}"] += 1
                invalid_text = True
                break
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n" + text
            else:
                turns.append({"role": role, "content": text})
        if invalid_text:
            incomplete_trees += 1
            continue
        if len(turns) < 2 or turns[0]["role"] != "user" or turns[-1]["role"] != "assistant":
            incomplete_trees += 1
            continue
        if any(turn["role"] == turns[index - 1]["role"] for index, turn in enumerate(turns) if index):
            incomplete_trees += 1
            continue
        conversations.append({
            "group_id": f"oasst1:{tree_id}",
            "messages": turns,
            "source": OASST_REPOSITORY,
            "license": "Apache-2.0",
            "human_authored": True,
            "synthetic": False,
        })

    if not conversations:
        raise ValueError("No OASST1 conversations survived strict quality filtering.")
    info = {
        "repository": OASST_REPOSITORY,
        "revision": OASST_REVISION,
        "file": OASST_FILE,
        "sha256": actual_sha,
        "size_bytes": actual_size,
        "license": "Apache-2.0",
        "human_authored": True,
        "synthetic_records_included": 0,
        "input_rows": len(rows),
        "accepted_messages": len(filtered),
        "conversations": len(conversations),
        "dropped_messages": dict(sorted(dropped.items())),
        "dropped_incomplete_trees": incomplete_trees,
    }
    return conversations, info


def load_dolly_conversations(cache: Path) -> tuple[list[dict], dict]:
    from huggingface_hub import hf_hub_download

    jsonl_path = Path(
        hf_hub_download(
            repo_id=DOLLY_REPOSITORY,
            filename=DOLLY_FILE,
            repo_type="dataset",
            revision=DOLLY_REVISION,
            cache_dir=str(cache / "huggingface"),
        )
    )
    records = []
    dropped = defaultdict(int)
    seen = set()
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                dropped["not_object"] += 1
                continue
            instruction, instruction_reason = normalize_training_text(row.get("instruction"))
            response, response_reason = normalize_training_text(row.get("response"))
            raw_context = row.get("context") or ""
            if raw_context:
                context, context_reason = normalize_training_text(raw_context)
            else:
                context, context_reason = "", "accepted"
            rejected = False
            for field, reason in (
                ("instruction", instruction_reason),
                ("context", context_reason),
                ("response", response_reason),
            ):
                if reason != "accepted":
                    dropped[f"{field}_{reason}"] += 1
                    rejected = True
            if rejected:
                continue
            user_text = instruction
            if context:
                user_text += "\n\nContext:\n" + context
            group = hashlib.sha256(
                (user_text + "\n" + response).encode("utf-8")
            ).hexdigest()
            if group in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(group)
            records.append({
                "group_id": f"dolly:{group}",
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": response},
                ],
                "category": row.get("category"),
                "source": DOLLY_REPOSITORY,
                "license": "CC-BY-SA-3.0",
                "human_authored": True,
                "synthetic": False,
            })
    if not records:
        raise ValueError("No usable Dolly records found.")
    return records, {
        "repository": DOLLY_REPOSITORY,
        "revision": DOLLY_REVISION,
        "file": DOLLY_FILE,
        "sha256": sha256_file(jsonl_path),
        "size_bytes": jsonl_path.stat().st_size,
        "license": "CC-BY-SA-3.0",
        "human_authored": True,
        "synthetic_records_included": 0,
        "conversations": len(records),
        "dropped": dict(sorted(dropped.items())),
    }


def write_pretraining_outputs(output: Path, wikitext: dict) -> dict:
    return wikitext["splits"]


def write_sft_splits(output: Path, conversations: list[dict]) -> dict:
    splits = {"train": [], "validation": [], "test": []}
    seen_content = set()
    for record in conversations:
        group = record["group_id"]
        content_key = hashlib.sha256(
            json.dumps(record["messages"], sort_keys=True).encode("utf-8")
        ).hexdigest()
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        split = stable_split(group)
        splits[split].append(record)

    details = {}
    for split, records in splits.items():
        if not records:
            raise ValueError(
                f"The instruction {split} split is empty; select more conversation data "
                "or review the stable group split before training."
            )
        count, digest = write_jsonl(output / f"sft_{split}.jsonl", records)
        details[split] = {
            "path": f"sft_{split}.jsonl",
            "conversations": count,
            "sha256": digest,
        }
    return details


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare pinned, non-synthetic English data for Q-Mini Add/Sub v2."
    )
    parser.add_argument("--output", default="data/qmini_addsub_v1")
    parser.add_argument("--cache", default=".cache/qmini_addsub_v1")
    parser.add_argument(
        "--accept-source-licenses",
        action="store_true",
        help="Confirm FineWiki CC-BY-SA-4.0, OASST1 Apache-2.0 and Dolly CC-BY-SA-3.0 terms before download.",
    )
    args = parser.parse_args()
    if not args.accept_source_licenses:
        parser.error(
            "Pass --accept-source-licenses only after reviewing FineWiki CC-BY-SA-4.0, "
            "OASST1 Apache-2.0 and Dolly CC-BY-SA-3.0 terms."
        )

    output = Path(args.output).expanduser().resolve()
    cache = Path(args.cache).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_dir():
            parser.error(f"Output path is not a directory: {output}")
        existing = [str(path) for path in output.iterdir()]
        if existing:
            parser.error(
                "Prepared output directory is not empty; choose a fresh versioned "
                "folder to avoid overwriting a prior dataset: " + ", ".join(existing[:8])
            )

    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.preparing-",
        dir=output.parent,
    ))
    try:
        print("Preparing hash-pinned FineWiki English Wikipedia sample...")
        finewiki_info = prepare_finewiki_sample(cache, staging)
        print("Preparing strictly filtered human OASST1 conversations...")
        oasst_records, oasst_info = load_oasst_conversations(cache)
        print("Preparing human-authored Databricks Dolly instructions...")
        dolly_records, dolly_info = load_dolly_conversations(cache)

        all_conversations = oasst_records + dolly_records
        sft_splits = write_sft_splits(staging, all_conversations)
        manifest = {
            "format": "qmini-addsub-data-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "language": "English",
            "synthetic_training_records_included": 0,
            "cleaning_policy": {
                "version": "qmini-addsub-text-clean-v2",
                "normalization": "Unicode NFC; normalized line endings; horizontal whitespace collapse; control-character removal",
                "rejections": ["short/empty text", "overlong text", "email addresses", "credential-like secrets", "explicitly labelled phone-contact strings"],
                "language_authority": "OASST1 lang=en metadata and pinned English Dolly source; no heuristic language detection is claimed",
                "maximum_text_characters": 32768
            },
            "pretraining": finewiki_info,
            "instruction_sources": {
                "oasst1": oasst_info,
                "dolly": dolly_info,
            },
            "approved_instruction_sources": [
                OASST_REPOSITORY,
                DOLLY_REPOSITORY,
            ],
            "instruction_splits": sft_splits,
            "excluded_sources": {
                "DailyDialog": "Not included by default: CC-BY-NC-SA-4.0 has non-commercial/share-alike conditions.",
                "TinyStories": "Excluded because this project requires human-authored, non-synthetic training data; FineWiki English Wikipedia is the human-authored pretraining source.",
                "Alpaca/UltraChat/teacher-generated corpora": "Excluded as model-generated or insufficiently verified synthetic data.",
            },
            "split_rule": "stable SHA-256 group split: 90% train, 5% validation, 5% test",
        }
        manifest_path = staging / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Commit a complete prepared dataset with one same-filesystem directory rename.
        if output.exists():
            output.rmdir()  # It was checked to be empty above.
        staging.replace(output)
        print(json.dumps({
            "manifest": str(output / "dataset_manifest.json"),
            "finewiki_english": finewiki_info["splits"],
            "oasst1_conversations": len(oasst_records),
            "dolly_conversations": len(dolly_records),
            "sft_splits": sft_splits,
            "synthetic_training_records_included": 0,
        }, ensure_ascii=False, indent=2))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
