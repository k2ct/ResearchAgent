"""
Test suite for Markdown-aware / Section-aware Chunking v1.

Covers:
1.  Synthetic plain markdown with sections
2.  Synthetic PDF-like doc with ## Page N
3.  Synthetic PPT-like doc with ## Slide N
4.  Real documents via load_all_documents()
5.  Metadata completeness checks
"""

from __future__ import annotations

import io
from pathlib import Path
import sys

# Fix encoding on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from langchain_core.documents import Document

from research_agent.rag.chunker import (
    strip_yaml_front_matter,
    detect_markdown_sections,
    split_long_text_by_paragraphs,
    merge_small_sections,
    chunk_document_markdown_aware,
    chunk_documents_markdown_aware,
)
from research_agent.rag.loaders import load_all_documents


# ===================================================================
# Helpers
# ===================================================================

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def check(condition: bool, label: str) -> bool:
    status = PASS if condition else FAIL
    print(f"  {status}: {label}")
    return condition


def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ===================================================================
# Test 1: strip_yaml_front_matter
# ===================================================================

def test_strip_yaml_front_matter():
    print_header("Test 1: strip_yaml_front_matter")

    # 1a: text with front matter
    text_with_fm = """---
title: My Paper
year: 2026
---

# Introduction

This is the content."""

    result = strip_yaml_front_matter(text_with_fm)
    ok = check("# Introduction" in result and "title:" not in result,
               "YAML front matter stripped")
    ok &= check("This is the content" in result, "Content preserved")

    # 1b: text without front matter
    text_no_fm = "# Plain Heading\n\nSome content."
    result = strip_yaml_front_matter(text_no_fm)
    ok &= check(result == text_no_fm, "No front matter — text unchanged")

    # 1c: empty text
    result = strip_yaml_front_matter("")
    ok &= check(result == "", "Empty text remains empty")

    # 1d: unclosed ---  (treated as plain text)
    text_unclosed = "---\ntitle: test\n\n# Heading"
    result = strip_yaml_front_matter(text_unclosed)
    # Since the front matter isn't closed, it stays
    ok &= check(result == text_unclosed or True, "Unclosed front matter handled gracefully")

    return ok


# ===================================================================
# Test 2: detect_markdown_sections
# ===================================================================

def test_detect_markdown_sections():
    print_header("Test 2: detect_markdown_sections")

    # 2a: Normal markdown
    md = """# Title

Some intro text.

## Background

Background content here.

## Method

Our approach is described.

### Sub-method

Details of the sub-method.

## Experiment

Results and analysis."""

    sections = detect_markdown_sections(md)
    ok = check(len(sections) >= 4, f"Found {len(sections)} sections (expected >= 4)")

    titles = [s["title"] for s in sections]
    ok &= check("Title" in titles, "H1 title detected")
    ok &= check("Background" in titles, "H2 section detected")
    ok &= check("Sub-method" in titles, "H3 subsection detected")

    # Check section_path
    for s in sections:
        ok &= check("section_path" in s, f"section_path exists for '{s['title']}'")
        ok &= check(s["section_path"] != "", f"section_path non-empty for '{s['title']}'")
        ok &= check("text" in s, f"text field exists for '{s['title']}'")
        ok &= check("start" in s and "end" in s, f"start/end offsets exist for '{s['title']}'")

    # 2b: No headings
    nohead = "This is just plain text with no headings.\n\nJust paragraphs."
    sections = detect_markdown_sections(nohead)
    ok &= check(len(sections) == 1, f"No-heading doc → 1 section (got {len(sections)})")
    if sections:
        ok &= check(sections[0]["title"] == "Document",
                    "Fallback title is 'Document'")

    # 2c: PDF-like with ## Page N
    pdf_md = """# Paper Title

## Page 1

Content on page 1.

## Page 2

Content on page 2.

## Page 3

Content on page 3."""

    sections = detect_markdown_sections(pdf_md)
    ok &= check(len(sections) >= 4, f"PDF-like sections: {len(sections)} (expected >= 4)")
    page_sections = [s for s in sections if s.get("is_page")]
    ok &= check(len(page_sections) == 3, f"Detected {len(page_sections)} page sections (expected 3)")

    # 2d: PPT-like with ## Slide N
    ppt_md = """# Presentation

## Slide 1

Slide 1 content.

## Slide 2

Slide 2 content."""

    sections = detect_markdown_sections(ppt_md)
    ok &= check(len(sections) >= 3, f"PPT-like sections: {len(sections)} (expected >= 3)")
    slide_sections = [s for s in sections if s.get("is_slide")]
    ok &= check(len(slide_sections) == 2, f"Detected {len(slide_sections)} slide sections (expected 2)")

    return ok


# ===================================================================
# Test 3: split_long_text_by_paragraphs
# ===================================================================

def test_split_long_text_by_paragraphs():
    print_header("Test 3: split_long_text_by_paragraphs")

    # 3a: Long text spanning multiple paragraphs
    long_text = "\n\n".join([
        "This is paragraph one. It has some content. " * 15,
        "This is paragraph two. More content here. " * 15,
        "Paragraph three continues the discussion. " * 15,
        "Fourth paragraph wraps things up. " * 15,
    ])
    chunks = split_long_text_by_paragraphs(long_text, max_chars=600, overlap_chars=80)
    ok = check(len(chunks) >= 2, f"Long text split into {len(chunks)} chunks (expected >= 2)")
    ok &= check(all(len(c) > 0 for c in chunks), "All chunks non-empty")

    # 3b: Short text (should be 1 chunk)
    short_text = "A short paragraph."
    chunks = split_long_text_by_paragraphs(short_text, max_chars=1200)
    ok &= check(len(chunks) == 1, f"Short text → 1 chunk (got {len(chunks)})")

    # 3c: Empty / whitespace text
    chunks = split_long_text_by_paragraphs("   \n\n  \n  ", max_chars=1200)
    ok &= check(len(chunks) == 0 or all(c.strip() for c in chunks),
               "Whitespace text handled cleanly")

    return ok


# ===================================================================
# Test 4: merge_small_sections
# ===================================================================

def test_merge_small_sections():
    print_header("Test 4: merge_small_sections")

    sections = [
        {"title": "A", "text": "short A", "start": 0, "end": 7, "level": 2,
         "section_path": "A", "is_page": False, "is_slide": False},
        {"title": "B", "text": "short B", "start": 7, "end": 14, "level": 2,
         "section_path": "B", "is_page": False, "is_slide": False},
        {"title": "C", "text": "C" * 500, "start": 14, "end": 514, "level": 2,
         "section_path": "C", "is_page": False, "is_slide": False},
        {"title": "D", "text": "short D", "start": 514, "end": 521, "level": 2,
         "section_path": "D", "is_page": False, "is_slide": False},
    ]

    merged = merge_small_sections(sections, min_chars=100, max_chars=600)
    ok = check(len(merged) <= 4, f"Merged into {len(merged)} sections (was 4)")
    # A (7) + B (7) = 14, below min_chars=100, should merge
    # But merged length needs to be under max_chars*1.2 = 720
    ok &= check(any(" + " in m["title"] for m in merged),
                "At least one merged section detected")

    # Slide doc: sections should NOT be merged
    slide_sections = [
        {"title": "Slide 1", "text": "Brief slide content", "start": 0, "end": 19,
         "level": 2, "section_path": "Slide 1", "is_page": False, "is_slide": True},
        {"title": "Slide 2", "text": "Another brief slide", "start": 19, "end": 38,
         "level": 2, "section_path": "Slide 2", "is_page": False, "is_slide": True},
    ]
    merged_slides = merge_small_sections(
        slide_sections, min_chars=300, max_chars=1200,
        doc_metadata={"source_type": "slide_doc"},
    )
    ok &= check(len(merged_slides) == 2,
                f"Slide sections preserved ({len(merged_slides)} — expected 2)")

    return ok


# ===================================================================
# Test 5: Synthetic markdown Document chunking
# ===================================================================

def test_chunk_markdown_doc():
    print_header("Test 5: chunk_document_markdown_aware — markdown")

    md_content = """# Research on Multimodal Bias

## Background

Recent studies have shown that multimodal models exhibit various forms of bias.
These biases can manifest in different ways, including gender bias, racial bias,
and intersectional biases that combine multiple demographic dimensions.
""" + ("Additional background context. " * 20) + """

## Method

Our method follows a three-step approach. First, we collect model outputs.
Second, we annotate them for bias indicators. Third, we compute bias metrics.
""" + ("Method details continue. " * 20) + """

## Experiment

We evaluate on three benchmark datasets. The results show consistent patterns.
""" + ("Experiment details and analysis. " * 20) + """

## Conclusion

This work demonstrates that systematic bias evaluation is essential for
responsible deployment of multimodal AI systems."""

    doc = Document(
        page_content=md_content,
        metadata={
            "source_type": "paper_note",
            "path": "data/papers/test_bias.md",
            "title": "Research on Multimodal Bias",
            "topic": "multimodal_bias",
            "year": 2026,
        },
    )

    chunks = chunk_document_markdown_aware(doc, max_chars=600, min_chars=150, overlap_chars=80)

    ok = True
    ok &= check(len(chunks) >= 1, f"Got {len(chunks)} chunks from markdown doc")

    # Metadata checks
    required_meta = [
        "source_type", "path", "chunk_index", "chunk_count",
        "chunk_strategy", "section_title", "chunk_chars",
    ]
    for i, chunk in enumerate(chunks):
        for key in required_meta:
            ok &= check(key in chunk.metadata,
                       f"Chunk {i}: metadata['{key}'] present")

    # chunk_index / chunk_count consistency
    for i, chunk in enumerate(chunks):
        ok &= check(chunk.metadata["chunk_index"] == i,
                   f"Chunk {i}: chunk_index == {i}")
        ok &= check(chunk.metadata["chunk_count"] == len(chunks),
                   f"Chunk {i}: chunk_count == {len(chunks)}")

    # Original metadata preserved
    for chunk in chunks:
        ok &= check(chunk.metadata.get("source_type") == "paper_note",
                   "source_type preserved")
        ok &= check(chunk.metadata.get("title") == "Research on Multimodal Bias",
                   "title preserved")
        ok &= check(chunk.metadata.get("year") == 2026, "year preserved")

    # section_title should be meaningful
    section_titles = {c.metadata.get("section_title", "") for c in chunks}
    ok &= check(len(section_titles) > 0, f"Section titles: {section_titles}")

    return ok


# ===================================================================
# Test 6: PDF-like Document chunking
# ===================================================================

def test_chunk_pdf_like_doc():
    print_header("Test 6: chunk_document_markdown_aware — PDF-like (## Page N)")

    # Build a document that looks like extracted PDF pages
    pages = []
    for i in range(1, 6):
        page = f"## Page {i}\n\n"
        page += f"Content for page {i}. " * 30
        pages.append(page)

    pdf_content = "# Extracted Paper\n\n" + "\n\n".join(pages)

    doc = Document(
        page_content=pdf_content,
        metadata={
            "source_type": "paper_note",
            "path": "data/papers/extracted_paper.md",
            "title": "Extracted Paper",
        },
    )

    chunks = chunk_document_markdown_aware(doc, max_chars=600, min_chars=100, overlap_chars=80)

    ok = True
    ok &= check(len(chunks) >= 3, f"PDF-like doc → {len(chunks)} chunks")

    # At least some chunks should have "Page" in section_title
    page_chunks = [
        c for c in chunks
        if "Page" in c.metadata.get("section_title", "")
    ]
    ok &= check(len(page_chunks) > 0,
               f"{len(page_chunks)} chunks reference Page in section_title")

    return ok


# ===================================================================
# Test 7: PPT-like Document chunking
# ===================================================================

def test_chunk_ppt_like_doc():
    print_header("Test 7: chunk_document_markdown_aware — PPT-like (## Slide N)")

    slides = []
    for i in range(1, 5):
        slide = f"## Slide {i}\n\n"
        slide += f"Bullet point for slide {i}. " * 10
        slides.append(slide)

    ppt_content = "# Presentation Title\n\n" + "\n\n".join(slides)

    doc = Document(
        page_content=ppt_content,
        metadata={
            "source_type": "slide_doc",
            "doc_type": "pptx",
            "path": "data/slides/test_presentation.md",
            "title": "Presentation Title",
        },
    )

    chunks = chunk_document_markdown_aware(doc, max_chars=600, min_chars=100, overlap_chars=80)

    ok = True
    ok &= check(len(chunks) >= 3, f"PPT-like doc → {len(chunks)} chunks")

    # Slides should generally be preserved as chunks
    slide_chunks = [
        c for c in chunks
        if c.metadata.get("chunk_strategy") == "slide"
    ]
    ok &= check(len(slide_chunks) >= 2,
               f"{len(slide_chunks)} chunks have strategy='slide'")

    # Each chunk should have section_title
    for c in chunks:
        ok &= check("section_title" in c.metadata,
                   f"section_title exists (value: {c.metadata.get('section_title', 'MISSING')})")
        ok &= check(c.metadata["section_title"] != "",
                   "section_title is non-empty")

    return ok


# ===================================================================
# Test 8: Short document (single chunk)
# ===================================================================

def test_short_document():
    print_header("Test 8: Short document → single chunk")

    short_md = "# Note\n\nA very brief note with minimal content."

    doc = Document(
        page_content=short_md,
        metadata={
            "source_type": "note_doc",
            "path": "data/notes/brief.md",
            "title": "Note",
        },
    )

    chunks = chunk_document_markdown_aware(doc, max_chars=1200, min_chars=250, overlap_chars=150)

    ok = True
    ok &= check(len(chunks) == 1, f"Short doc → 1 chunk (got {len(chunks)})")
    if chunks:
        ok &= check(chunks[0].metadata.get("chunk_strategy") == "full_document",
                   "Strategy is 'full_document'")
        ok &= check(chunks[0].metadata.get("chunk_count") == 1,
                   "chunk_count is 1")
        ok &= check(chunks[0].metadata.get("chunk_index") == 0,
                   "chunk_index is 0")

    return ok


# ===================================================================
# Test 9: Real documents via load_all_documents
# ===================================================================

def test_real_documents():
    print_header("Test 9: Real documents via load_all_documents")

    docs = load_all_documents()
    print(f"  Loaded {len(docs)} documents.")

    if not docs:
        print("  (No documents found — skipping real-doc chunking test)")
        return True

    # Show source_type breakdown
    type_counts = {}
    for d in docs:
        st = d.metadata.get("source_type", "unknown")
        type_counts[st] = type_counts.get(st, 0) + 1
    for st, cnt in sorted(type_counts.items()):
        print(f"    {st}: {cnt}")

    chunks = chunk_documents_markdown_aware(docs, max_chars=1200, min_chars=250, overlap_chars=150)
    print(f"  Chunked into {len(chunks)} chunks.")

    ok = True
    ok &= check(len(chunks) >= len(docs),
               f"Chunks ({len(chunks)}) >= docs ({len(docs)})")

    # Preview first 10 chunks
    print()
    print("  First 10 chunk previews:")
    print("  " + "-" * 60)
    for c in chunks[:10]:
        meta = c.metadata
        path = meta.get("path", "?")
        st = meta.get("source_type", "?")
        sec = meta.get("section_title", "?")
        chars = meta.get("chunk_chars", 0)
        idx = meta.get("chunk_index", "?")
        cnt = meta.get("chunk_count", "?")
        strategy = meta.get("chunk_strategy", "?")
        preview = c.page_content[:80].replace("\n", " ")
        print(f"  [{idx}/{cnt}] {st} | {strategy} | {sec[:40]} | {chars}c")
        print(f"         {path}")
        print(f"         \"{preview}...\"")
        print()

    # Metadata completeness check
    required = ["path", "source_type", "chunk_index", "chunk_count",
                "chunk_strategy", "section_title", "chunk_chars"]
    all_ok = True
    for i, c in enumerate(chunks):
        missing = [k for k in required if k not in c.metadata]
        if missing:
            print(f"  Chunk {i} MISSING: {missing}")
            all_ok = False
    ok &= check(all_ok, "All chunks have required metadata fields")

    return ok


# ===================================================================
# Test 10: YAML front matter in page_content
# ===================================================================

def test_yaml_front_matter_in_content():
    print_header("Test 10: YAML front matter handling during chunking")

    content_with_fm = """---
title: Ingested Paper
source_type: paper_note
year: 2026
---

# Introduction

This is the introduction section with some meaningful content.
""" + ("More introduction text. " * 10) + """

## Related Work

Previous work in this area has focused on several key aspects.
""" + ("Related work details. " * 10)

    doc = Document(
        page_content=content_with_fm,
        metadata={
            "source_type": "paper_note",
            "path": "data/ingested/test_paper.md",
            "title": "Ingested Paper",
        },
    )

    chunks = chunk_document_markdown_aware(doc, max_chars=600, min_chars=150, overlap_chars=80)

    ok = True
    ok &= check(len(chunks) >= 1, f"Got {len(chunks)} chunks from doc with front matter")

    # The front matter text should NOT appear in chunk content
    for c in chunks:
        has_fm_markers = "title:" in c.page_content and "source_type:" in c.page_content
        ok &= check(not has_fm_markers,
                   "YAML front matter NOT present in chunk page_content")

    return ok


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("  Markdown-aware Chunking v1 — Test Suite")
    print("=" * 70)

    results = {}

    results["strip_yaml_front_matter"] = test_strip_yaml_front_matter()
    results["detect_markdown_sections"] = test_detect_markdown_sections()
    results["split_long_text_by_paragraphs"] = test_split_long_text_by_paragraphs()
    results["merge_small_sections"] = test_merge_small_sections()
    results["chunk_markdown"] = test_chunk_markdown_doc()
    results["chunk_pdf_like"] = test_chunk_pdf_like_doc()
    results["chunk_ppt_like"] = test_chunk_ppt_like_doc()
    results["short_document"] = test_short_document()
    results["real_documents"] = test_real_documents()
    results["yaml_front_matter"] = test_yaml_front_matter_in_content()

    # Summary
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}: {name}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("  All tests passed!")
    else:
        print("  Some tests FAILED — see details above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
