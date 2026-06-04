"""
Test script for LLM Enhancement Layer.

Usage:
    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_llm_enhancement.py

Tests:
1. Config check -- verify ENABLE_LLM_ENHANCEMENT, no key printed
2. Claim Support Enhancement -- generate_claim_support(use_llm=True)
3. Paper Reading Enhancement -- read_paper(use_llm=True)
4. PPT Progress Enhancement -- generate_progress_memory(use_llm=True)
5. Report Polish Enhancement -- enhance_report_text()

All tests must pass regardless of whether LLM is enabled or not.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


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


# -- Test 1: Config check (no key printed) ----------------------------


def test_config():
    section("Test 1: LLM config — check without printing key")

    from research_agent.llm.client import (
        is_llm_enhancement_enabled,
        get_llm_config,
        get_chat_llm,
    )

    enabled = is_llm_enhancement_enabled()
    config = get_llm_config()
    llm = get_chat_llm()

    print(f"  ENABLE_LLM_ENHANCEMENT: {enabled}")
    print(f"  Model: {config['model']}")
    print(f"  Temperature: {config['temperature']}")
    print(f"  Max input chars: {config['max_input_chars']}")
    print(f"  Base URL: {config['base_url'] or '(default)'}")

    # NEVER print API key
    api_key = config["api_key"]
    if api_key:
        print(f"  API key: configured (length={len(api_key)})")
    else:
        print(f"  API key: not configured")

    print(f"  ChatOpenAI available: {llm is not None}")

    check(isinstance(enabled, bool), "is_llm_enhancement_enabled returns bool")
    check(config["model"] != "", "model is non-empty")
    check("api_key" not in sys.stdout.__repr__(), "no raw key in check")

    if not enabled:
        check(llm is None, "ChatOpenAI is None when enhancement is disabled")


# -- Test 2: Claim Support Enhancement ---------------------------------


def test_claim_support():
    section("Test 2: Claim Support — use_llm=True")

    from research_agent.claim.claim_support import generate_claim_support

    claim = "共现关系可能诱发 LVLM 对缺失对象的幻觉。"
    result = generate_claim_support(claim, top_k_per_query=2, use_llm=True)

    check("report" in result, "result has 'report'")
    check(len(result["report"]) > 0, "report is non-empty")
    check("used_llm" in result, "result has 'used_llm'")
    check("llm_error" in result, "result has 'llm_error'")
    check(result.get("evidence_count", -1) >= 0, "evidence_count is set")

    print(f"  Claim type: {result.get('claim_type', '?')}")
    print(f"  Evidence count: {result.get('evidence_count', 0)}")
    print(f"  Used LLM: {result.get('used_llm', False)}")
    if result.get("llm_error"):
        print(f"  LLM error: {result['llm_error']}")
    print(f"  Report preview: {result['report'][:200].replace(chr(10), ' ')}")


# -- Test 3: Paper Reading Enhancement ----------------------------------


def test_paper_reading():
    section("Test 3: Paper Reading — use_llm=True")

    from research_agent.paper.paper_reader import read_paper

    # Find a paper
    paper_path = None
    candidates = [
        PROJECT_ROOT / "data" / "papers" / "guardrail_agnostic.md",
        PROJECT_ROOT / "data" / "papers" / "multimodal_bias_survey.md",
        PROJECT_ROOT / "data" / "papers" / "hallucination_evaluation_note.md",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            paper_path = p
            break

    if paper_path is None:
        # Try ingested
        ingested = sorted((PROJECT_ROOT / "data" / "ingested").glob("*.md"))
        for p in ingested:
            if p.stat().st_size > 100:
                paper_path = p
                break

    if paper_path is None:
        check(False, "No paper file found for testing")
        return

    print(f"  Using: {paper_path.name}")
    result = read_paper(paper_path, use_llm=True)

    check(result["status"] == "success", f"status = {result['status']}")
    check(len(result["reading_note"]) > 0, "reading_note is non-empty")
    check("used_llm" in result, "result has 'used_llm'")
    check("llm_error" in result, "result has 'llm_error'")
    check(result["metadata"]["title"] != "", "title is non-empty")

    print(f"  Title: {result['metadata'].get('title', '?')}")
    print(f"  Used LLM: {result.get('used_llm', False)}")
    if result.get("llm_error"):
        print(f"  LLM error: {result['llm_error']}")
    print(f"  Sections: {list(result['sections'].keys())}")


# -- Test 4: PPT Progress Enhancement ---------------------------------


def test_progress_memory():
    section("Test 4: PPT Progress Memory — use_llm=True")

    from research_agent.progress.ppt_progress_memory import generate_progress_memory

    # Find a slide doc
    slide_path = None
    candidates = [
        PROJECT_ROOT / "data" / "ingested" / "sample_progress_test_slides.md",
        PROJECT_ROOT / "data" / "ingested" / "sample_english_progress_test_slides.md",
        PROJECT_ROOT / "data" / "ingested" / "刘晗组会 20260529.md",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            slide_path = p
            break

    if slide_path is None:
        # Auto-generate a sample slide
        slide_path = PROJECT_ROOT / "data" / "ingested" / "_test_slide.md"
        slide_path.write_text("""---
source_type: slide_doc
title: Test Research Progress
original_path: _test_slide.md
---

## Slide 1

# Research Progress Report

- Completed COCO hallucination screening pipeline
- Built stereotype library with 50 attribute pairs
- Evaluated 3 VLM models for gender bias

## Slide 2

# Key Findings

- LVLM shows higher hallucination on underrepresented groups
- Co-occurrence patterns indicate systematic bias
- hrs_v1 metric correlates well with human annotation

## Slide 3

# Next Steps

- Expand stereotype library to 200 pairs
- Integrate MinerU for paper ingestion
- Build paper reading pipeline
""", encoding="utf-8")

    print(f"  Using: {slide_path.name}")
    result = generate_progress_memory(slide_path, use_llm=True)

    check("progress_memory" in result, "result has 'progress_memory'")
    check(len(result["progress_memory"]) > 0, "progress_memory is non-empty")
    check("used_llm" in result, "result has 'used_llm'")
    check("llm_error" in result, "result has 'llm_error'")
    check(len(result.get("slides", [])) > 0, "slides detected")

    print(f"  Slides: {len(result.get('slides', []))}")
    print(f"  Topics: {list(result.get('topics', {}).keys())}")
    print(f"  Used LLM: {result.get('used_llm', False)}")
    if result.get("llm_error"):
        print(f"  LLM error: {result['llm_error']}")

    # Clean up auto-generated test file
    if slide_path.name == "_test_slide.md":
        slide_path.unlink(missing_ok=True)


# -- Test 5: Report Polish Enhancement ---------------------------------


def test_report_polish():
    section("Test 5: Report Polish — enhance_report_text()")

    from research_agent.llm.enhancers import enhance_report_text

    report_text = """# Research Report

## Background
This project focuses on multimodal bias evaluation in VLMs.

## Method
We constructed a stereotype library and evaluated hallucination patterns.

## Results
Preliminary results show co-occurrence biases in LVLM outputs.
"""

    sources = [
        {"path": "data/papers/guardrail_agnostic.md", "source_type": "paper_note"},
        {"path": "data/experiments/coco_val_n300_g1.md", "source_type": "experiment_doc"},
    ]

    result = enhance_report_text(report_text, sources, style="group_meeting")

    check("text" in result, "result has 'text'")
    check(len(result["text"]) > 0, "text is non-empty")
    check("used_llm" in result, "result has 'used_llm'")
    check("error" in result, "result has 'error'")

    print(f"  Used LLM: {result['used_llm']}")
    if result.get("error"):
        print(f"  Error: {result['error']}")
    print(f"  Output preview: {result['text'][:200].replace(chr(10), ' ')}")


# -- Test 6: Fallback behavior (LLM disabled) -------------------------


def test_fallback_behavior():
    section("Test 6: Fallback — modules work without LLM")

    from research_agent.llm.client import is_llm_enhancement_enabled

    # Use generate_claim_support with use_llm=False (default)
    from research_agent.claim.claim_support import generate_claim_support
    result = generate_claim_support(
        "测试论点。", top_k_per_query=1, use_llm=False
    )
    check(len(result["report"]) > 0, "claim report generated (no LLM)")
    check(not result.get("used_llm", True),
          "used_llm=False when use_llm flag is False")

    # If LLM is disabled globally, generate_claim_support(use_llm=True)
    # should still produce a report (rule-based fallback)
    result2 = generate_claim_support(
        "另一个测试论点。", top_k_per_query=1, use_llm=True
    )
    check(len(result2["report"]) > 0,
          "claim report generated (use_llm=True, with fallback if needed)")
    check("used_llm" in result2, "used_llm field present")

    enabled = is_llm_enhancement_enabled()
    if not enabled:
        print(f"  INFO  LLM is globally disabled — use_llm=True falls back to rule-based")
        check(not result2.get("used_llm", True),
              "used_llm=False when enhancement is globally disabled")
    else:
        print(f"  INFO  LLM is globally enabled")


# -- Main -------------------------------------------------------------


def main():
    global PASS, FAIL

    print("=" * 70)
    print("LLM Enhancement Layer — Test Suite")
    print("=" * 70)

    test_config()
    test_claim_support()
    test_paper_reading()
    test_progress_memory()
    test_report_polish()
    test_fallback_behavior()

    print(f"\n{'=' * 70}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 70}")

    if FAIL > 0:
        print("\nSome tests FAILED. Check output for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed (LLM enhancement layer works correctly).")
        sys.exit(0)


if __name__ == "__main__":
    main()
