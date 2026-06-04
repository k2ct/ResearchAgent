"""
Chunk quality analysis & report.

Usage::

    python scripts/analyze_chunks.py              # print report to stdout
    python scripts/analyze_chunks.py --csv        # also write reports/chunk_analysis.csv
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path
import sys
from typing import Optional

# Fix encoding on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.rag.loaders import load_all_documents
from research_agent.rag.chunker import (
    chunk_documents_markdown_aware,
    get_chunking_config,
    get_doc_type_chunking_profile,
)
from research_agent.rag.schemas import SourceType


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _bar(label: str, width: int = 68) -> None:
    print(f"\n{'=' * width}")
    print(f"  {label}")
    print(f"{'=' * width}")


def _kv(key: str, value: str, indent: int = 2) -> None:
    print(f"{' ' * indent}{key}: {value}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    write_csv = "--csv" in (argv or sys.argv)

    _bar("Chunk Analysis Report")

    # 0. Current config
    config = get_chunking_config()
    print("\n  [Config] Environment / defaults:")
    _kv("RAG_CHUNK_MAX_CHARS", str(config["max_chars"]))
    _kv("RAG_CHUNK_MIN_CHARS", str(config["min_chars"]))
    _kv("RAG_CHUNK_OVERLAP_CHARS", str(config["overlap_chars"]))

    # 1. Load
    docs = load_all_documents()
    doc_count = len(docs)
    _kv("doc_count", str(doc_count))

    if doc_count == 0:
        print("\n  No documents found — nothing to analyse.")
        return 0

    # 2. Chunk (using auto-detection via per-doc profiles)
    chunks = chunk_documents_markdown_aware(docs)
    chunk_count = len(chunks)
    _kv("chunk_count", str(chunk_count))

    if chunk_count == 0:
        print("\n  No chunks produced.")
        return 0

    # 3. Numeric stats
    chunk_chars_list = [c.metadata.get("chunk_chars", 0) for c in chunks]
    avg_chars = sum(chunk_chars_list) / len(chunk_chars_list)
    min_chars = min(chunk_chars_list)
    max_chars = max(chunk_chars_list)

    print()
    _kv("avg_chunk_chars", f"{avg_chars:.0f}")
    _kv("min_chunk_chars", str(min_chars))
    _kv("max_chunk_chars", str(max_chars))
    _kv("chunks_per_doc", f"{chunk_count / doc_count:.1f}")

    # 4. Strategy distribution
    strategy_counter = Counter(
        c.metadata.get("chunk_strategy", "unknown") for c in chunks
    )
    print("\n  [chunk_strategy distribution]")
    for strategy, cnt in strategy_counter.most_common():
        pct = cnt / chunk_count * 100
        print(f"    {strategy:20s} {cnt:5d}  ({pct:5.1f}%)")

    # 5. source_type distribution
    st_counter = Counter(
        c.metadata.get("source_type", "unknown") for c in chunks
    )
    print("\n  [source_type distribution (chunks)]")
    for st, cnt in st_counter.most_common():
        pct = cnt / chunk_count * 100
        print(f"    {st:20s} {cnt:5d}  ({pct:5.1f}%)")

    # 6. Per source_type chunk count vs doc count
    print("\n  [Per source_type: docs → chunks]")
    doc_st_counter = Counter(d.metadata.get("source_type", "unknown") for d in docs)
    for st in sorted(set(list(doc_st_counter) + list(st_counter))):
        d_cnt = doc_st_counter.get(st, 0)
        c_cnt = st_counter.get(st, 0)
        ratio = c_cnt / d_cnt if d_cnt else 0
        print(f"    {st:20s} {d_cnt:3d} docs → {c_cnt:4d} chunks  (×{ratio:.1f})")

    # 7. Top 10 longest chunks
    sorted_by_len = sorted(chunks, key=lambda c: len(c.page_content), reverse=True)
    print("\n  [Top 10 longest chunks]")
    print(f"    {'#':<4} {'chars':<8} {'strategy':<18} {'source_type':<18} {'section_title'}")
    print(f"    {'-'*4} {'-'*8} {'-'*18} {'-'*18} {'-'*30}")
    for i, c in enumerate(sorted_by_len[:10], 1):
        meta = c.metadata
        st = meta.get("source_type", "?")
        strategy = meta.get("chunk_strategy", "?")
        sec = meta.get("section_title", "?")[:40]
        chars = len(c.page_content)
        print(f"    {i:<4} {chars:<8} {strategy:<18} {st:<18} {sec}")

    # 8. Top 10 shortest (non-empty)
    print("\n  [Top 10 shortest chunks]")
    print(f"    {'#':<4} {'chars':<8} {'strategy':<18} {'source_type':<18} {'section_title'}")
    print(f"    {'-'*4} {'-'*8} {'-'*18} {'-'*18} {'-'*30}")
    for i, c in enumerate(sorted_by_len[-10:], 1):
        meta = c.metadata
        st = meta.get("source_type", "?")
        strategy = meta.get("chunk_strategy", "?")
        sec = meta.get("section_title", "?")[:40]
        chars = len(c.page_content)
        print(f"    {i:<4} {chars:<8} {strategy:<18} {st:<18} {sec}")

    # 9. CSV output
    if write_csv:
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(exist_ok=True)
        csv_path = reports_dir / "chunk_analysis.csv"

        fieldnames = [
            "chunk_index", "chunk_count", "chunk_strategy",
            "section_title", "section_path", "chunk_chars",
            "source_type", "path", "title",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for c in chunks:
                row = {k: c.metadata.get(k, "") for k in fieldnames}
                writer.writerow(row)

        print(f"\n  CSV written to: {csv_path}")

    _bar("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
