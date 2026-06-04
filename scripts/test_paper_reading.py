"""
Test script for Paper Reading Pipeline v1.

Usage:
    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_paper_reading.py

Tests:
1. Single paper read (legacy paper_note format from data/papers/)
2. Single paper read (ingested PDF-parsed markdown from data/ingested/)
3. Batch read papers -> output to data/paper_notes/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.paper.paper_reader import (
    read_paper,
    batch_read_papers,
    load_paper_markdown,
    detect_paper_sections,
    extract_paper_metadata,
)

PASS = 0
FAIL = 0


def check(condition: bool, label: str):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {label}")
        PASS += 1
    else:
        print(f"  FAIL  {label}")
        FAIL += 1


def section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_note_preview(reading_note: str, max_chars: int = 2000):
    """Print a preview of the reading note."""
    if len(reading_note) <= max_chars:
        print(reading_note)
    else:
        print(reading_note[:max_chars])
        print(f"\n... (truncated, total {len(reading_note)} chars)")


# -- Test 1: Load paper markdown (legacy format) ---------------------


def test_load_legacy_paper():
    section("Test 1: Load legacy paper markdown (guardrail_agnostic.md)")

    path = PROJECT_ROOT / "data" / "papers" / "guardrail_agnostic.md"
    if not path.exists():
        check(False, f"File not found: {path}")
        return

    data = load_paper_markdown(path)
    check(data["path"].endswith("guardrail_agnostic.md"), "path is correct")
    check(len(data["content"]) > 0, f"content loaded ({len(data['content'])} chars)")
    check("source_type" in data["metadata"], "metadata has source_type")
    check(
        data["metadata"].get("source_type") == "paper_note",
        "source_type = paper_note"
    )
    check("title" in data["metadata"], "metadata has title")
    print(f"  INFO  Title: {data['metadata'].get('title')}")
    print(f"  INFO  Topic: {data['metadata'].get('topic')}")
    print(f"  INFO  Year: {data['metadata'].get('year')}")


# -- Test 2: Section detection (legacy format) -----------------------


def test_section_detection_legacy():
    section("Test 2: Section detection — legacy Chinese headings")

    path = PROJECT_ROOT / "data" / "papers" / "guardrail_agnostic.md"
    if not path.exists():
        check(False, f"File not found: {path}")
        return

    data = load_paper_markdown(path)
    sections = detect_paper_sections(data["content"])

    non_empty = {k: len(v) for k, v in sections.items() if v}
    print(f"  Detected sections (non-empty): {list(non_empty.keys())}")
    print(f"  Section sizes: {non_empty}")

    check(len(non_empty) > 0, "at least one section detected")
    check("method" in non_empty or "experiments" in non_empty or "results" in non_empty,
          "at least one of method/experiments/results detected")


# -- Test 3: Metadata extraction -------------------------------------


def test_metadata_extraction():
    section("Test 3: Paper metadata extraction")

    path = PROJECT_ROOT / "data" / "papers" / "guardrail_agnostic.md"
    if not path.exists():
        check(False, f"File not found: {path}")
        return

    data = load_paper_markdown(path)
    meta = extract_paper_metadata(path, data["metadata"], data["content"])

    print(f"  Title:   {meta['title']}")
    print(f"  Year:    {meta['year']}")
    print(f"  Venue:   {meta['venue']}")
    print(f"  Authors: {meta['authors']}")
    print(f"  Doc Type:{meta['doc_type']}")
    print(f"  Src Type:{meta['source_type']}")

    check(len(meta["title"]) > 0, "title is non-empty")
    check(meta["source_type"] == "paper_note", "source_type = paper_note")


# -- Test 4: Full read_paper (legacy) ---------------------------------


def test_read_paper_legacy():
    section("Test 4: Full read_paper — guardrail_agnostic.md")

    path = PROJECT_ROOT / "data" / "papers" / "guardrail_agnostic.md"
    if not path.exists():
        check(False, f"File not found: {path}")
        return

    result = read_paper(path)

    check(result["status"] == "success", f"status = {result['status']}")
    check(len(result["reading_note"]) > 0, "reading_note is non-empty")
    check(len(result["sources"]) > 0, "sources is non-empty")
    check(result["metadata"]["title"] != "", "title in metadata")

    print(f"\n  --- Reading Note Preview (first 2000 chars) ---")
    print_note_preview(result["reading_note"], 2000)

    print(f"\n  INFO  Sections detected ({len(result['sections'])}):")
    for k, v in result["sections"].items():
        print(f"    {k}: {v[:100]}")


# -- Test 5: Full read_paper (ingested format) ------------------------


def test_read_paper_ingested():
    section("Test 5: Full read_paper — ingested PDF paper")

    # Find ingested papers
    ingested_dir = PROJECT_ROOT / "data" / "ingested"
    ingested_papers = sorted(ingested_dir.glob("*.md")) if ingested_dir.exists() else []

    # Find a paper_note type in ingested
    test_path = None
    for p in ingested_papers:
        if p.stat().st_size == 0:
            continue
        data = load_paper_markdown(p)
        if data["metadata"].get("source_type") == "paper_note":
            test_path = p
            break

    if test_path is None:
        # Fall back to any non-empty ingested md
        for p in ingested_papers:
            if p.stat().st_size > 0:
                test_path = p
                break

    if test_path is None:
        print("  SKIP  No ingested papers available for testing")
        check(True, "skip (no data)")
        return

    print(f"  Using: {test_path.name}")

    result = read_paper(test_path)

    check(result["status"] == "success", f"status = {result['status']}")
    check(len(result["reading_note"]) > 0,
          f"reading_note ({len(result['reading_note'])} chars) is non-empty")
    check(len(result["sources"]) > 0, "sources is non-empty")

    print(f"\n  --- Reading Note Preview (first 2000 chars) ---")
    print_note_preview(result["reading_note"], 2000)

    print(f"\n  INFO  Sections detected ({len(result['sections'])}):")
    for k, v in result["sections"].items():
        print(f"    {k}: {v[:100]}")


# -- Test 6: Batch read papers ----------------------------------------


def test_batch_read():
    section("Test 6: Batch read papers -> data/paper_notes/")

    papers_dir = PROJECT_ROOT / "data" / "papers"
    output_dir = PROJECT_ROOT / "data" / "paper_notes"

    if not papers_dir.exists():
        check(False, f"Directory not found: {papers_dir}")
        return

    results = batch_read_papers(papers_dir, output_dir)

    succeeded = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    print(f"  Processed: {len(results)} papers")
    print(f"  Succeeded: {len(succeeded)}")
    print(f"  Failed:    {len(failed)}")

    for r in succeeded:
        out = r.get("output_path", "?")
        title = r["metadata"].get("title", "?")
        print(f"    -> {Path(out).name} ({title[:60]})")

    for r in failed:
        print(f"    FAIL: {r['paper_path']} — {r.get('error', 'unknown')}")

    check(len(succeeded) > 0, f"at least one paper processed successfully")
    check(
        output_dir.exists() and any(output_dir.glob("*_paper_note.md")),
        "output dir contains _paper_note.md files"
    )

    # Show output files
    outputs = sorted(output_dir.glob("*_paper_note.md"))
    print(f"\n  Output files ({len(outputs)}):")
    for out in outputs:
        size = out.stat().st_size
        print(f"    {out.name} ({size} chars)")


# -- Main -------------------------------------------------------------


def main():
    global PASS, FAIL

    print("=" * 70)
    print("Paper Reading Pipeline v1 — Test Suite")
    print("=" * 70)

    test_load_legacy_paper()
    test_section_detection_legacy()
    test_metadata_extraction()
    test_read_paper_legacy()
    test_read_paper_ingested()
    test_batch_read()

    print(f"\n{'=' * 70}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 70}")

    if FAIL > 0:
        print("\nSome tests FAILED. Check output for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
