# Dataset source provenance and release gates

Q-Mini keeps research preparation backward-compatible, but an auditable release corpus should use the explicit source-provenance contract below.

## Metadata file

Create a UTF-8 JSON file whose format is \`qmini-source-provenance-v1\`. Each data file is identified by its basename; basenames must be unique in one preparation run.

\`\`\`json
{
  "format": "qmini-source-provenance-v1",
  "sources": [
    {
      "path": "corpus.jsonl",
      "source_name": "Name of the source collection",
      "source_uri": "https://example.org/dataset",
      "license_spdx": "CC-BY-4.0",
      "collected_at": "2026-11-10",
      "language": "en",
      "transformation_version": "normalize-v1"
    }
  ]
}
\`\`\`

\`license_spdx\` must be an explicit SPDX-style identifier or expression; do not guess a license. Research preparation without metadata records \`UNKNOWN\` in the manifest and is not release-cleared.

## Commands

Pass the same source list and metadata to both tokenizer fitting and token-stream preparation:

\`\`\`bash
python -m qmini.cli tokenizer \
  --data-dir ./datasets \
  --source-metadata ./datasets/source_metadata.json \
  --release-mode \
  --output ./artifacts/tokenizer.json \
  --vocab-size 8192 \
  --base-config ./configs/qmini_research_152m.json \
  --config-output ./artifacts/config.json

python -m qmini.cli prepare \
  --data-dir ./datasets \
  --source-metadata ./datasets/source_metadata.json \
  --release-mode \
  --tokenizer ./artifacts/tokenizer.json \
  --output-dir ./artifacts/tokens \
  --valid-fraction 0.02
\`\`\`

Release mode requires source name, public HTTP(S) URI, non-unknown license expression, ISO collection date, BCP-47-style language tag, and transformation version. It screens text for common email addresses, common API-token formats, AWS access-key IDs, and private-key headers. A hit blocks the run; the message reports only a count and never prints matched content. Remove or quarantine flagged records in a controlled environment, then rerun.

The tokenizer sidecar and \`dataset.json\` record source file size, SHA-256, provenance fields, and a deterministic provenance hash. Preparation refuses a release-mode dataset when the tokenizer was not itself trained in release mode or when its source/provenance hash differs.

## Limits

This is a fail-closed first gate, not a complete compliance scanner or legal determination. The secret/PII patterns are heuristic and can miss personal data, credentials, unusual formats, and non-English identifiers; manual review is still required. The SPDX expression syntax is checked, but the identifier is not resolved against an authoritative SPDX license registry and the tool cannot verify that the stated license actually applies to the source. The current pipeline still creates train/validation streams; a separate independent test split, language/source concentration thresholds, richer text-quality filters, and full cross-split leakage audit remain open work.
