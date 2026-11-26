from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    # Keep a patchable module attribute while allowing pure helper imports
    # without the optional network/data dependencies installed.
    hf_hub_download = None
try:
    import pyarrow.parquet as pq
except ImportError:
    # Keep the optional module patchable for deterministic source-prep tests.
    pq = None


from qmini_addsub.data import (
    accept_oasst1_message,
    normalize_training_text,
    sha256_file,
    stable_split,
    write_jsonl,
)


WIKITEXT_URL = (
    "https://s3.amazonaws.com/research.metamind.io/wikitext/"
    "wikitext-103-raw-v1.zip"
)
WIKITEXT_SHA256 = "91c00ae287f0d699e18605c84afc9e45c192bc6b7797ff8837e5474655a33794"
WIKITEXT_SIZE = 191_984_949
OASST_REPOSITORY = "OpenAssistant/oasst1"
OASST_REVISION = "fdf72ae0827c1cda404ff25b6603abec9e3399b"
OASST_FILE = "data/train-00000-of-00001-b42a775f407cee45.parquet"
OASST_SHA256 = "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e"
OASST_SIZE = 39_516_251
DOLLY_REPOSITORY = "databricks/databricks-dolly-15k"
DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DOLLY_FILE = "databricks-dolly-15k.jsonl"

SIMPLE_WIKI_REPOSITORY = "juno-labs/simple_wikipedia"
SIMPLE_WIKI_REVISION = "e95aee4ada4de4ec556d93505e577b29f9ff9a08"
SIMPLE_WIKI_FILE = "data/train-00000-of-00001.parquet"
SIMPLE_WIKI_SHA256 = "112b847312d4315c1c4cc75c02d2d153f17bce7ee4a3a1d57dc09385da72a02a"
SIMPLE_WIKI_SIZE = 439_202_943


def download_verified(url: str, destination: Path, sha256: str, size: int) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size == size and sha256_file(destination) == sha256:
            return destination
        destination.unlink()
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "Q-Mini-AddSub-Data-Prep/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    actual_size = temporary.stat().st_size
    actual_sha = sha256_file(temporary)
    if actual_size != size or actual_sha != sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Dataset integrity mismatch: size {actual_size}/{size}, "
            f"SHA-256 {actual_sha}/{sha256}"
        )
    temporary.replace(destination)
    return destination


def _require_hf_hub_download():
    downloader = hf_hub_download
    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download as downloader
        except ImportError as exc:
            raise RuntimeError(
                "Add/Sub dataset preparation requires the optional data dependencies. "
                "Install with: python -m pip install 'qmini[addsub-data]'."
            ) from exc
    return downloader


def prepare_simple_wikipedia(cache: Path, output: Path) -> dict:
    """Prepare hash-pinned, human-authored Simple English Wikipedia articles."""
    hf_hub_download_fn = _require_hf_hub_download()
    parquet_reader = pq
    if parquet_reader is None:
        try:
            import pyarrow.parquet as parquet_reader
        except ImportError as exc:
            raise RuntimeError(
                "Add/Sub dataset preparation requires the optional data dependencies. "
                "Install with: python -m pip install 'qmini[addsub-data]'."
            ) from exc
    parquet_path = Path(hf_hub_download_fn(
        repo_id=SIMPLE_WIKI_REPOSITORY,
        filename=SIMPLE_WIKI_FILE,
        repo_type="dataset",
        revision=SIMPLE_WIKI_REVISION,
        cache_dir=str(cache / "huggingface"),
    ))
    actual_size = parquet_path.stat().st_size
    actual_sha = sha256_file(parquet_path)
    if actual_size != SIMPLE_WIKI_SIZE or actual_sha != SIMPLE_WIKI_SHA256:
        raise ValueError(
            "Simple English Wikipedia file failed pinned size/SHA-256 verification."
        )

    parquet_file = parquet_reader.ParquetFile(parquet_path)
    split_paths = {
        name: output / f"pretrain_{name}.txt"
        for name in ("train", "validation", "test")
    }
    counts = {name: {"articles": 0, "bytes": 0} for name in split_paths}
    dropped = {
        "source_error": 0,
        "empty_or_rejected": 0,
        "missing_page_id": 0,
        "duplicate_page_id": 0,
        "duplicate_article": 0,
    }
    handles = {
        name: path.open("w", encoding="utf-8", newline="\n")
        for name, path in split_paths.items()
    }
    seen_page_ids: set[str] = set()
    seen_articles: set[str] = set()
    try:
        # Use bounded Parquet batches rather than materializing the full corpus.
        for batch in parquet_file.iter_batches(
            batch_size=512,
            columns=["page_id", "title", "content_no_link", "error"],
        ):
            for row in batch.to_pylist():
                if row.get("error") is True:
                    dropped["source_error"] += 1
                    continue
                raw_page_id = row.get("page_id")
                if raw_page_id is None or not str(raw_page_id).strip():
                    dropped["missing_page_id"] += 1
                    continue
                page_id = str(raw_page_id)
                if page_id in seen_page_ids:
                    dropped["duplicate_page_id"] += 1
                    continue
                seen_page_ids.add(page_id)

                title, title_reason = normalize_training_text(
                    row.get("title"), min_chars=1
                )
                content, content_reason = normalize_training_text(
                    row.get("content_no_link"), min_chars=20
                )
                if title_reason != "accepted" or content_reason != "accepted":
                    dropped["empty_or_rejected"] += 1
                    continue

                article = f"{title}\n{content}\n"
                content_key = hashlib.sha256(
                    (title + "\n" + content).encode("utf-8")
                ).hexdigest()
                if content_key in seen_articles:
                    dropped["duplicate_article"] += 1
                    continue
                seen_articles.add(content_key)

                split = stable_split(
                    page_id, seed="qmini-simplewiki-20260301"
                )
                handles[split].write(article)
                counts[split]["articles"] += 1
    finally:
        for handle in handles.values():
            handle.close()

    for split, path in split_paths.items():
        counts[split]["bytes"] = path.stat().st_size
        counts[split]["sha256"] = sha256_file(path)
        if counts[split]["articles"] == 0:
            raise ValueError(f"SimpleWiki {split} split is empty.")

    return {
        "repository": SIMPLE_WIKI_REPOSITORY,
        "revision": SIMPLE_WIKI_REVISION,
        "file": SIMPLE_WIKI_FILE,
        "dump_date": "2026-03-01",
        "sha256": actual_sha,
        "size_bytes": actual_size,
        "license": "CC-BY-SA-3.0",
        "text_field": "content_no_link",
        "human_authored_source": True,
        "generated_importance_labels_used": False,
        "split_rule": "stable SHA-256 page_id split: 90% train, 5% validation, 5% test",
        "splits": counts,
        "dropped": dropped,
    }


def prepare_wikitext(cache: Path, output: Path) -> dict:
    archive_path = download_verified(
        WIKITEXT_URL,
        cache / "wikitext-103-raw-v1.zip",
        WIKITEXT_SHA256,
        WIKITEXT_SIZE,
    )
    split_specs = {
        "train": "wiki.train.raw",
        "validation": "wiki.valid.raw",
        "test": "wiki.test.raw",
    }
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        for split, wanted in split_specs.items():
            matches = [name for name in members if Path(name).name == wanted]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one {wanted} in WikiText archive; found {matches}"
                )
            text = archive.read(matches[0]).decode("utf-8")
            target = output / f"pretrain_{split}.txt"
            target.write_text(text, encoding="utf-8", newline="\n")
            counts[split] = {
                "lines": sum(1 for line in text.splitlines() if line.strip()),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
    return {
        "repository": "Salesforce/wikitext",
        "subset": "wikitext-103-raw-v1",
        "upstream_archive": WIKITEXT_URL,
        "archive_sha256": WIKITEXT_SHA256,
        "archive_size_bytes": WIKITEXT_SIZE,
        "license": "CC-BY-SA / GFDL (upstream WikiText metadata; confirm applicable version and source terms)",
        "citation": "Merity et al., WikiText: A long-term dependency language modeling dataset",
        "splits": counts,
    }


def _select_oasst_conversation_path(
    messages: list[dict],
) -> tuple[list[dict] | None, str | None, bool]:
    """Select one deterministic, complete root-to-leaf path from an OASST tree.

    OASST message trees can branch. Choosing children[0] makes the training
    corpus depend on Parquet row order and can silently select a short or
    incomplete branch. Instead, validate message identity/root structure and
    choose the longest complete path; equal-length paths use message-ID order
    as a stable tie-break. Cyclic paths and duplicate IDs are rejected.
    """
    by_id: dict[str, dict] = {}
    for message in messages:
        raw_id = message.get("message_id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            return None, "invalid_message_id", False
        message_id = raw_id.strip()
        if message_id in by_id:
            return None, "duplicate_message_id", False
        by_id[message_id] = message

    roots = [
        message for message in messages
        if message.get("parent_id") is None
    ]
    if len(roots) != 1:
        return None, "invalid_root_count", False
    root = roots[0]
    if root.get("role") != "prompter":
        return None, "invalid_root_role", False

    children: dict[str, list[dict]] = defaultdict(list)
    orphan_messages = 0
    for message in messages:
        parent_id = message.get("parent_id")
        if parent_id is None:
            continue
        parent_key = str(parent_id)
        if parent_key not in by_id:
            # A parent may have been filtered out by OASST review/rank policy.
            # Its descendants are not reachable from the accepted root.
            orphan_messages += 1
            continue
        children[parent_key].append(message)

    for child_list in children.values():
        child_list.sort(key=lambda item: str(item["message_id"]))

    branching = any(len(child_list) > 1 for child_list in children.values())
    root_id = str(root["message_id"])
    stack: list[tuple[dict, tuple[dict, ...], frozenset[str]]] = [
        (root, (root,), frozenset((root_id,)))
    ]
    best_path: tuple[dict, ...] | None = None
    best_key: tuple[int, tuple[str, ...]] | None = None

    while stack:
        current, path, path_ids = stack.pop()
        current_id = str(current["message_id"])
        available_children = children.get(current_id, [])
        next_children = [
            child for child in available_children
            if str(child["message_id"]) not in path_ids
        ]
        has_cycle_edge = len(next_children) != len(available_children)

        if next_children:
            for child in reversed(next_children):
                child_id = str(child["message_id"])
                stack.append((
                    child,
                    (*path, child),
                    path_ids | frozenset((child_id,)),
                ))
            continue

        # A cyclic edge is not a valid leaf. Do not turn a malformed loop into
        # a plausible-looking truncated training conversation.
        if has_cycle_edge:
            continue
        if path[0].get("role") != "prompter" or path[-1].get("role") != "assistant":
            continue

        ids = tuple(str(item["message_id"]) for item in path)
        key = (-len(path), ids)
        if best_key is None or key < best_key:
            best_key = key
            best_path = path

    if best_path is None:
        reason = "orphaned_or_cyclic_tree" if orphan_messages else "no_complete_assistant_path"
        return None, reason, branching
    return list(best_path), None, branching


def load_oasst_conversations(cache: Path) -> tuple[list[dict], dict]:
    hf_hub_download_fn = _require_hf_hub_download()
    parquet_reader = pq
    if parquet_reader is None:
        try:
            import pyarrow.parquet as parquet_reader
        except ImportError as exc:
            raise RuntimeError(
                "Add/Sub dataset preparation requires the optional data dependencies. "
                "Install with: python -m pip install 'qmini[addsub-data]'."
            ) from exc
    parquet_path = Path(
        hf_hub_download_fn(
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

    table = parquet_reader.read_table(parquet_path)
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
    dropped_tree_reasons: dict[str, int] = defaultdict(int)
    branching_trees = 0
    for tree_id, messages in by_tree.items():
        chain, reason, branching = _select_oasst_conversation_path(messages)
        if branching:
            branching_trees += 1
        if reason is not None or chain is None:
            incomplete_trees += 1
            dropped_tree_reasons[reason or "invalid_path"] += 1
            continue

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
        "dropped_tree_reasons": dict(sorted(dropped_tree_reasons.items())),
        "branching_trees": branching_trees,
        "branch_policy": (
            "longest complete root-to-leaf path; equal-length paths are "
            "tie-broken by lexicographically sorted message IDs"
        ),
    }
    return conversations, info


def load_dolly_conversations(cache: Path) -> tuple[list[dict], dict]:
    hf_hub_download_fn = _require_hf_hub_download()
    jsonl_path = Path(
        hf_hub_download_fn(
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


def prepare_dataset(output: Path, cache: Path) -> dict:
    """Build every source/split in a sibling staging directory and publish atomically.

    Existing non-empty output is never overwritten. The download cache must be
    outside the output tree so a failed rebuild cannot damage a verified cache.
    """
    output = Path(output).expanduser()
    cache = Path(cache).expanduser()
    output_parent = output.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output = output_parent / output.name
    cache_resolved = cache.resolve()

    if output.is_symlink():
        raise ValueError("Refusing to prepare data through an output symlink.")
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"Dataset output exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(
                f"Dataset output is not empty: {output}. Select a new folder; "
                "verified data will never be overwritten implicitly."
            )
    if cache_resolved == output.resolve() or output.resolve() in cache_resolved.parents:
        raise ValueError("Dataset download cache must be outside the output directory.")

    cache.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.staging-",
        dir=str(output_parent),
    ))
    try:
        print("Preparing hash-pinned Simple English Wikipedia training data...")
        simplewiki_info = prepare_simple_wikipedia(cache, staging)
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
            "pretraining": simplewiki_info,
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
                "TinyStories": "Excluded because this project requires human-authored, non-synthetic training data; Simple English Wikipedia is the human-authored pretraining source.",
                "Alpaca/UltraChat/teacher-generated corpora": "Excluded as model-generated or insufficiently verified synthetic data.",
            },
            "split_rule": "stable SHA-256 group split: 90% train, 5% validation, 5% test",
        }
        manifest_path = staging / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # An empty folder may have been pre-created by Colab. Remove only that
        # empty folder; if anything was added meanwhile, rmdir safely fails.
        if output.exists():
            output.rmdir()
        staging.replace(output)
        return {
            "manifest": str(output / "dataset_manifest.json"),
            "simple_wikipedia": simplewiki_info["splits"],
            "oasst1_conversations": len(oasst_records),
            "dolly_conversations": len(dolly_records),
            "sft_splits": sft_splits,
            "synthetic_training_records_included": 0,
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare pinned, non-synthetic English data for Q-Mini Add/Sub v1."
    )
    parser.add_argument("--output", default="data/qmini_addsub_v1")
    parser.add_argument("--cache", default=".cache/qmini_addsub_v1")
    parser.add_argument(
        "--accept-source-licenses",
        action="store_true",
        help="Confirm attribution/share-alike and source-specific license terms before download.",
    )
    args = parser.parse_args()
    if not args.accept_source_licenses:
        parser.error(
            "Pass --accept-source-licenses only after reviewing SimpleWiki CC-BY-SA-3.0, "
            "OASST1 Apache-2.0 and Dolly CC-BY-SA-3.0 terms."
        )
    summary = prepare_dataset(Path(args.output), Path(args.cache))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
