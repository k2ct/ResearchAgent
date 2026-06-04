"""
Test suite for the Document Ingestion v1 pipeline.

Tests:
1. YAML front matter parsing (ingested docs)
2. Legacy ## 基本??息 parsing fallback
3. source_type inference from directory names
4. End-to-end: ingest -> load_all_documents -> verify
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.rag.loaders import (
    parse_front_matter,
    parse_basic_info,
    load_all_documents,
    summarize_documents,
    format_document_preview,
    load_markdown_document,
)
from research_agent.rag.schemas import DIR_TO_SOURCE_TYPE
from research_agent.ingestion.document_ingestor import (
    infer_source_type,
    build_metadata,
    write_ingested_markdown,
    ingest_file,
    ingest_directory,
    extract_text_from_md,
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
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── Test 1: YAML Front Matter Parsing ─────────────────────────


def test_parse_front_matter():
    section("Test 1: parse_front_matter (YAML front matter)")

    text = """---
source_type: paper_note
title: My Research Paper
year: 2026
topic: multimodal_bias
path: data/ingested/my_paper.md
---

# My Research Paper

Some content here.
"""
    meta = parse_front_matter(text)
    check(meta.get("source_type") == "paper_note", "source_type = paper_note")
    check(meta.get("title") == "My Research Paper", "title parsed")
    check(meta.get("year") == 2026, "year normalized to int")
    check(meta.get("topic") == "multimodal_bias", "topic parsed")
    check(meta.get("path") == "data/ingested/my_paper.md", "path parsed")


def test_parse_front_matter_fallback():
    section("Test 2: parse_front_matter returns {} for legacy docs")

    text = """# Some Paper

## 基本信息

- source_type: paper_note
- title: A Paper

Content here.
"""
    meta = parse_front_matter(text)
    check(meta == {}, "parse_front_matter returns empty dict for legacy format")

    basic = parse_basic_info(text)
    check(basic.get("source_type") == "paper_note", "parse_basic_info still works")
    check(basic.get("title") == "A Paper", "parse_basic_info title")


def test_parse_front_matter_malformed():
    section("Test 3: parse_front_matter handles malformed YAML")

    text = """---
this is not valid yaml: [
---
# Content
"""
    meta = parse_front_matter(text)
    check(meta == {}, "returns empty dict for malformed YAML")


# ── Test 2: source_type Inference ──────────────────────────


def test_infer_source_type():
    section("Test 4: infer_source_type from directory names")

    tests = [
        ("raw_docs/papers_pdf/paper.pdf", "paper_note"),
        ("raw_docs/notes_docx/note.docx", "note_doc"),
        ("raw_docs/slides_pptx/slide.pptx", "slide_doc"),
        ("raw_docs/misc_md/note.md", "misc_doc"),
        ("raw_docs/unknown_dir/file.pdf", "misc_doc"),  # fallback
    ]
    for path_str, expected in tests:
        result = infer_source_type(Path(path_str))
        check(result == expected, f"{path_str} -> {expected} (got {result})")


# ── Test 3: End-to-End Roundtrip ───────────────────────────


def test_ingest_roundtrip():
    section("Test 5: End-to-end ingest -> load_all_documents")

    raw_dir = PROJECT_ROOT / "raw_docs" / "misc_md"
    ingested_dir = PROJECT_ROOT / "data" / "ingested"

    raw_dir.mkdir(parents=True, exist_ok=True)
    ingested_dir.mkdir(parents=True, exist_ok=True)

    # Create a sample markdown note for testing
    sample_path = raw_dir / "sample_note.md"
    sample_path.write_text("""# Sample Research Note

This note is about LVLM hallucination and RAG.

## Key Points

- Hallucination detection in large vision-language models
- RAG can reduce hallucination by grounding generation in retrieved evidence
- CHAIR metric is commonly used for evaluation

## References

1. CHAIR: Evaluating Hallucination in Image Captioning (2023)
2. RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
""", encoding="utf-8")
    print(f"  Created sample: {sample_path}")

    # Ingest it
    result = ingest_file(sample_path, ingested_dir)
    check(result["status"] == "success", f"ingest status = success")
    check(result["source_type"] == "misc_doc", f"source_type = misc_doc")
    check(result["char_count"] > 0, f"char_count > 0")
    print(f"  Output: {result['output_path']}")

    # Verify output file exists
    output_path = ingested_dir / "sample_note.md"
    check(output_path.exists(), f"output file exists: {output_path}")

    # Verify front matter in output
    output_text = output_path.read_text(encoding="utf-8")
    check(output_text.startswith("---"), "output starts with --- (YAML front matter)")

    # Load via load_all_documents
    all_docs = load_all_documents()
    summary = summarize_documents(all_docs)
    print(f"  Document summary: {summary}")

    # Check that misc_doc is present
    check("misc_doc" in summary, "misc_doc appears in document summary")

    # Find the ingested doc
    ingested_docs = [d for d in all_docs if d.metadata.get("source_type") == "misc_doc"]
    check(len(ingested_docs) > 0, f"at least one misc_doc loaded ({len(ingested_docs)})")

    if ingested_docs:
        doc = ingested_docs[0]
        title = doc.metadata.get("title", "")
        check(title == "Sample Research Note", f"title = '{title}'")

        print(f"\n  --- Ingested Document Preview ---")
        preview = format_document_preview(doc, max_chars=300)
        for line in preview.splitlines():
            print(f"  {line}")

    # Clean up sample
    sample_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)
    print(f"\n  Cleaned up test files.")


# ── Main ───────────────────────────────────────────────────


def main():
    global PASS, FAIL

    print("=" * 60)
    print("Document Ingestion v1 Test Suite")
    print("=" * 60)

    test_parse_front_matter()
    test_parse_front_matter_fallback()
    test_parse_front_matter_malformed()
    test_infer_source_type()
    test_ingest_roundtrip()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    if FAIL > 0:
        print("\nSome tests FAILED. Check the output above for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
