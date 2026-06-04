"""
Test suite for chunking configuration & per-doc-type profiles.

Covers:
1.  Default config loading (env vars → hardcoded fallback)
2.  Env var override
3.  Per-doc-type profile lookup
4.  slide_doc profile: overlap=0, preserve_slide=True
5.  paper_note profile: larger max_chars
6.  Chunk metadata completeness with auto-detection
7.  chunk_document_markdown_aware with None params uses profiles
"""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys

# Fix encoding on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from langchain_core.documents import Document

from research_agent.rag.chunker import (
    get_chunking_config,
    get_doc_type_chunking_profile,
    chunk_document_markdown_aware,
    chunk_documents_markdown_aware,
)


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
# Test 1: Default config
# ===================================================================

def test_default_config():
    print_header("Test 1: get_chunking_config() — default (no env vars)")

    # Unset env vars to test defaults
    for key in ("RAG_CHUNK_MAX_CHARS", "RAG_CHUNK_MIN_CHARS", "RAG_CHUNK_OVERLAP_CHARS"):
        os.environ.pop(key, None)

    cfg = get_chunking_config()
    ok = True
    ok &= check(cfg["max_chars"] == 1200, f"default max_chars = {cfg['max_chars']} (expected 1200)")
    ok &= check(cfg["min_chars"] == 250, f"default min_chars = {cfg['min_chars']} (expected 250)")
    ok &= check(cfg["overlap_chars"] == 150, f"default overlap_chars = {cfg['overlap_chars']} (expected 150)")
    return ok


# ===================================================================
# Test 2: Env var override
# ===================================================================

def test_env_override():
    print_header("Test 2: get_chunking_config() — env var override")

    os.environ["RAG_CHUNK_MAX_CHARS"] = "800"
    os.environ["RAG_CHUNK_MIN_CHARS"] = "100"
    os.environ["RAG_CHUNK_OVERLAP_CHARS"] = "60"

    cfg = get_chunking_config()
    ok = True
    ok &= check(cfg["max_chars"] == 800, f"env max_chars = {cfg['max_chars']} (expected 800)")
    ok &= check(cfg["min_chars"] == 100, f"env min_chars = {cfg['min_chars']} (expected 100)")
    ok &= check(cfg["overlap_chars"] == 60, f"env overlap_chars = {cfg['overlap_chars']} (expected 60)")

    # Clean up
    for key in ("RAG_CHUNK_MAX_CHARS", "RAG_CHUNK_MIN_CHARS", "RAG_CHUNK_OVERLAP_CHARS"):
        os.environ.pop(key, None)

    return ok


# ===================================================================
# Test 3: doc-type profiles
# ===================================================================

def test_doc_type_profiles():
    print_header("Test 3: get_doc_type_chunking_profile()")

    ok = True

    # slide_doc
    p = get_doc_type_chunking_profile({"source_type": "slide_doc"})
    ok &= check(p.get("max_chars") == 1000, f"slide_doc max_chars = {p.get('max_chars')}")
    ok &= check(p.get("overlap_chars") == 0, f"slide_doc overlap = {p.get('overlap_chars')} (expected 0)")
    ok &= check(p.get("preserve_slide") is True, f"slide_doc preserve_slide = {p.get('preserve_slide')}")

    # pptx (via doc_type)
    p = get_doc_type_chunking_profile({"doc_type": "pptx"})
    ok &= check(p.get("overlap_chars") == 0, f"pptx overlap = {p.get('overlap_chars')} (expected 0)")

    # paper_note
    p = get_doc_type_chunking_profile({"source_type": "paper_note"})
    ok &= check(p.get("max_chars") == 1500, f"paper_note max_chars = {p.get('max_chars')} (expected 1500)")
    ok &= check(p.get("overlap_chars") == 180, f"paper_note overlap = {p.get('overlap_chars')} (expected 180)")

    # pdf (via doc_type)
    p = get_doc_type_chunking_profile({"doc_type": "pdf"})
    ok &= check(p.get("max_chars") == 1500, f"pdf max_chars = {p.get('max_chars')} (expected 1500)")

    # experiment_doc
    p = get_doc_type_chunking_profile({"source_type": "experiment_doc"})
    ok &= check(p.get("max_chars") == 1000, f"experiment_doc max_chars = {p.get('max_chars')} (expected 1000)")
    ok &= check(p.get("overlap_chars") == 120, f"experiment_doc overlap = {p.get('overlap_chars')} (expected 120)")

    # note_doc
    p = get_doc_type_chunking_profile({"source_type": "note_doc"})
    ok &= check(p.get("max_chars") == 1200, f"note_doc max_chars = {p.get('max_chars')} (expected 1200)")

    # dataset_doc — should return empty dict (fall back to env)
    p = get_doc_type_chunking_profile({"source_type": "dataset_doc"})
    ok &= check(p == {}, f"dataset_doc profile = empty dict (got {p})")

    # unknown type — empty dict
    p = get_doc_type_chunking_profile({"source_type": "unknown_type"})
    ok &= check(p == {}, f"unknown type profile = empty dict (got {p})")

    return ok


# ===================================================================
# Test 4: Auto-detection via None params
# ===================================================================

def test_auto_detection():
    print_header("Test 4: chunk_document_markdown_aware — auto-detection (params=None)")

    # Build a paper_note doc that should get paper profile (max_chars=1500)
    paper_content = """# A Paper on Bias

## Introduction

This paper explores multimodal bias in large vision-language models.
""" + ("More paper content. " * 80)

    paper_doc = Document(
        page_content=paper_content,
        metadata={
            "source_type": "paper_note",
            "path": "data/papers/auto_test.md",
            "title": "Test Paper",
        },
    )

    # Call with None — should use paper_note profile (max_chars=1500)
    chunks = chunk_document_markdown_aware(paper_doc)
    ok = True
    ok &= check(len(chunks) >= 1, f"paper_note auto → {len(chunks)} chunks")
    for c in chunks:
        ok &= check("chunk_strategy" in c.metadata, "chunk_strategy present")
        ok &= check("section_title" in c.metadata, "section_title present")

    # Build a slide_doc that should get slide profile (overlap=0)
    slide_content = """# My Presentation

## Slide 1

Slide one content. """ + ("Bullet point. " * 15) + """

## Slide 2

Slide two content. """ + ("Another bullet. " * 15)

    slide_doc = Document(
        page_content=slide_content,
        metadata={
            "source_type": "slide_doc",
            "doc_type": "pptx",
            "path": "data/slides/auto_test.md",
            "title": "Test Slides",
        },
    )

    chunks = chunk_document_markdown_aware(slide_doc)
    ok &= check(len(chunks) >= 2, f"slide auto → {len(chunks)} chunks (expected >= 2)")
    slide_chunks = [c for c in chunks if c.metadata.get("chunk_strategy") == "slide"]
    ok &= check(len(slide_chunks) >= 2, f"slide strategy chunks: {len(slide_chunks)}")

    return ok


# ===================================================================
# Test 5: Explicit params override profile
# ===================================================================

def test_explicit_override():
    print_header("Test 5: Explicit params override profile")

    paper_doc = Document(
        page_content="# Title\n\nShort content. " * 10,
        metadata={
            "source_type": "paper_note",
            "path": "data/papers/override_test.md",
            "title": "Override Test",
        },
    )

    # With explicit max_chars=400 (much smaller than paper profile 1500)
    chunks_small = chunk_document_markdown_aware(paper_doc, max_chars=400)
    # With profile auto-detection (max_chars=1500)
    chunks_auto = chunk_document_markdown_aware(paper_doc)

    ok = True
    # With smaller max_chars we should get MORE chunks
    ok &= check(len(chunks_small) >= len(chunks_auto),
                f"Explicit max_chars=400 → {len(chunks_small)} chunks, "
                f"auto → {len(chunks_auto)} (smaller max → more chunks)")

    return ok


# ===================================================================
# Test 6: Batch auto-detection
# ===================================================================

def test_batch_auto_detection():
    print_header("Test 6: chunk_documents_markdown_aware — mixed doc types")

    docs = [
        Document(
            page_content="# Paper\n\nContent. " * 30,
            metadata={"source_type": "paper_note", "path": "p1.md", "title": "P1"},
        ),
        Document(
            page_content="# Slides\n\n## Slide 1\n\nContent. " * 10 + "\n\n## Slide 2\n\nMore. " * 10,
            metadata={"source_type": "slide_doc", "doc_type": "pptx", "path": "s1.md", "title": "S1"},
        ),
    ]

    chunks = chunk_documents_markdown_aware(docs)
    ok = True
    ok &= check(len(chunks) >= 2, f"Batch → {len(chunks)} chunks")

    # Each doc should have its own profile applied
    paper_chunks = [c for c in chunks if c.metadata.get("source_type") == "paper_note"]
    slide_chunks = [c for c in chunks if c.metadata.get("source_type") == "slide_doc"]

    ok &= check(len(paper_chunks) >= 1, f"paper chunks: {len(paper_chunks)}")
    ok &= check(len(slide_chunks) >= 1, f"slide chunks: {len(slide_chunks)}")

    # Slide chunks should have strategy "slide" or "section"
    for c in slide_chunks:
        ok &= check(c.metadata.get("chunk_strategy") in ("slide", "section"),
                   f"slide chunk strategy: {c.metadata.get('chunk_strategy')}")

    return ok


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("  Chunking Config & Profiles — Test Suite")
    print("=" * 70)

    results = {}

    results["default_config"] = test_default_config()
    results["env_override"] = test_env_override()
    results["doc_type_profiles"] = test_doc_type_profiles()
    results["auto_detection"] = test_auto_detection()
    results["explicit_override"] = test_explicit_override()
    results["batch_auto"] = test_batch_auto_detection()

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
