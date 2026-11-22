from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

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


def load_oasst_conversations(cache: Path) -> tuple[list[dict], dict]:
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
            "Pass --accept-source-licenses only after reviewing WikiText CC-BY-SA/GFDL, "
            "OASST1 Apache-2.0 and Dolly CC-BY-SA-3.0 terms."
        )

    output, cache = Path(args.output), Path(args.cache)
    output.mkdir(parents=True, exist_ok=True)
    print("Preparing WikiText-103 language data (hash-pinned)...")
    wikitext_info = prepare_wikitext(cache, output)
    print("Preparing strictly filtered human OASST1 conversations...")
    oasst_records, oasst_info = load_oasst_conversations(cache)
    print("Preparing human-authored Databricks Dolly instructions...")
    dolly_records, dolly_info = load_dolly_conversations(cache)

    all_conversations = oasst_records + dolly_records
    sft_splits = write_sft_splits(output, all_conversations)
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
        "pretraining": wikitext_info,
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
            "TinyStories": "Excluded because this project requires human-authored, non-synthetic training data.",
            "Alpaca/UltraChat/teacher-generated corpora": "Excluded as model-generated or insufficiently verified synthetic data.",
        },
        "split_rule": "stable SHA-256 group split: 90% train, 5% validation, 5% test",
    }
    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "wikitext": wikitext_info["splits"],
        "oasst1_conversations": len(oasst_records),
        "dolly_conversations": len(dolly_records),
        "sft_splits": sft_splits,
        "synthetic_training_records_included": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
