"""
Test suite for Memory Writer.

Usage::

    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_memory_writer.py

Tests all 10 decision functions and the full write pipeline.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.memory.writer import (
    _SCHEMA_AVAILABLE,
    _STORE_AVAILABLE,
    detect_explicit_memory_instruction,
    infer_memory_level,
    infer_owner_agent,
    infer_memory_type,
    infer_memory_scope,
    infer_tags,
    should_promote_mid_to_long,
    build_memory_record_from_source,
    write_memory_from_source,
)

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def check(condition: bool, label: str) -> bool:
    status = PASS if condition else FAIL
    print(f"  {status}: {label}")
    return condition


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _f(record, key: str, default=None):
    """Access a field from either a dict or a dataclass MemoryRecord."""
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


# ═══════════════════════════════════════════════════════════════════════════
# Test 1 — explicit long-term instruction
# ═══════════════════════════════════════════════════════════════════════════

def test_explicit_long_term():
    section("Test 1: explicit long-term instruction")
    text = "请记住：我的长期研究方向是共现关系对LVLM幻觉的影响。"
    result = detect_explicit_memory_instruction(text)
    ok = True
    ok &= check(result["has_instruction"], "has_instruction=True")
    ok &= check(result["suggested_level"] == "long_term",
                f"suggested_level={result['suggested_level']} (expected long_term)")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 2 — explicit mid-term instruction
# ═══════════════════════════════════════════════════════════════════════════

def test_explicit_mid_term():
    section("Test 2: explicit mid-term instruction")
    text = "这个阶段我还要继续完善 MinerU PDF 解析和 RAG 检索。"
    level = infer_memory_level(text, source_module="")
    ok = True
    ok &= check(level["memory_level"] in ("mid_term", "long_term"),
                f"memory_level={level['memory_level']}")
    # Tags should pick up MinerU/RAG/PDF
    tags = infer_tags(text)
    ok &= check(any("MinerU" in t or "RAG" in t or "PDF" in t for t in tags),
                f"tags contain tech terms: {tags}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 3 — explicit short-term
# ═══════════════════════════════════════════════════════════════════════════

def test_explicit_short_term():
    section("Test 3: explicit short-term instruction")
    text = "临时记一下，这次终端报错是端口占用。"
    result = detect_explicit_memory_instruction(text)
    ok = True
    ok &= check(result["has_instruction"], "has_instruction=True")
    ok &= check(result["suggested_level"] == "short_term",
                f"suggested_level={result['suggested_level']}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 4 — paper_reading source
# ═══════════════════════════════════════════════════════════════════════════

def test_paper_source():
    section("Test 4: paper_reading source → paper_agent + paper_note")
    record = build_memory_record_from_source(
        content="## Abstract\n\nThis paper investigates hallucination in LVLMs.\n\n## Conclusion\n\nWe find that bias correlates with hallucination.",
        source_module="paper_reading",
        source_path="data/papers/hallucination_paper.md",
        source_title="Hallucination in LVLMs",
    )
    ok = True
    ok &= check(_f(record, "owner_agent") == "paper_agent",
                f"owner_agent={_f(record, 'owner_agent')}")
    ok &= check(_f(record, "memory_type") == "paper_note",
                f"memory_type={_f(record, 'memory_type')}")
    ok &= check(_f(record, "memory_level") in ("long_term", "mid_term"),
                f"memory_level={_f(record, 'memory_level')} (paper→elevated)")
    tags = _f(record, "tags", [])
    ok &= check(any("hallucination" in t.lower() or "LVLM" in t for t in tags),
                f"tags include hallucination/LVLM: {tags}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 5 — claim_support source
# ═══════════════════════════════════════════════════════════════════════════

def test_claim_source():
    section("Test 5: claim_support source → claim_agent + shared")
    record = build_memory_record_from_source(
        content="Claim: bias leads to hallucination. Evidence: supported by 3 papers.",
        source_module="claim_support",
        source_path="claims/bias_hallucination.md",
        source_title="Bias-Hallucination Link",
    )
    ok = True
    ok &= check(_f(record, "owner_agent") == "claim_agent",
                f"owner_agent={_f(record, 'owner_agent')}")
    ok &= check(_f(record, "memory_type") == "claim_support",
                f"memory_type={_f(record, 'memory_type')}")
    sw = _f(record, "shared_with", [])
    ok &= check(len(sw) > 0, f"shared_with non-empty: {sw}")
    ok &= check("paper_agent" in sw, f"shared_with includes paper_agent: {sw}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 6 — ppt_progress source
# ═══════════════════════════════════════════════════════════════════════════

def test_ppt_progress_source():
    section("Test 6: ppt_progress source → progress_agent + shared")
    record = build_memory_record_from_source(
        content="## Slide 5\n\n下一步计划\n\n- 构建 stereotype library\n- 下一步引入 OpenImages-MIAP\n- future work: cross-model comparison",
        source_module="ppt_progress",
        source_path="data/progress_memory/report_progress.md",
        source_title="Group Meeting 2026-06-05",
    )
    ok = True
    ok &= check(_f(record, "owner_agent") == "progress_agent",
                f"owner_agent={_f(record, 'owner_agent')}")
    ok &= check(_f(record, "memory_type") == "progress_update",
                f"memory_type={_f(record, 'memory_type')}")
    sw = _f(record, "shared_with", [])
    ok &= check(len(sw) > 0, f"shared_with non-empty: {sw}")
    ok &= check("experiment_agent" in sw, f"shared_with includes experiment_agent: {sw}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 7 — mid→long promotion
# ═══════════════════════════════════════════════════════════════════════════

def test_mid_to_long_promotion():
    section("Test 7: mid_term promoted to long_term after 3+ mentions")
    content = "stereotype library construction for LVLM bias evaluation"
    existing = [
        {"memory_level": "mid_term", "tags": ["stereotype", "bias", "LVLM"],
         "content": "Building stereotype library for bias evaluation",
         "summary": "stereotype library construction"},
        {"memory_level": "mid_term", "tags": ["stereotype", "bias"],
         "content": "Stereotype attribute collection for COCO dataset",
         "summary": "attribute collection"},
        {"memory_level": "mid_term", "tags": ["stereotype", "evaluation"],
         "content": "Evaluating stereotype impact on hallucination",
         "summary": "evaluation of stereotype impact"},
    ]
    result = should_promote_mid_to_long(content, existing)
    ok = True
    ok &= check(result["promote"], f"promote=True (matched={result['matched_count']})")
    ok &= check(result["matched_count"] >= 3,
                f"matched_count={result['matched_count']} (expected >=3)")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 8 — no promotion when completed
# ═══════════════════════════════════════════════════════════════════════════

def test_no_promotion_when_completed():
    section("Test 8: no promotion when completion marker present")
    content = "stereotype library construction for LVLM bias evaluation"
    existing = [
        {"memory_level": "mid_term", "tags": ["stereotype", "bias"],
         "content": "Building stereotype library — completed",
         "summary": "completed stereotype library"},
        {"memory_level": "mid_term", "tags": ["stereotype", "bias"],
         "content": "Stereotype attribute collection finished",
         "summary": "collection finished"},
        {"memory_level": "mid_term", "tags": ["stereotype", "bias"],
         "content": "Evaluating stereotype — done",
         "summary": "evaluation done"},
    ]
    result = should_promote_mid_to_long(content, existing)
    ok = True
    ok &= check(not result["promote"],
                f"promote=False (has completion markers, matched={result['matched_count']})")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 9 — write_memory_from_source
# ═══════════════════════════════════════════════════════════════════════════

def test_write_memory():
    section("Test 9: write_memory_from_source")
    result = write_memory_from_source(
        content="请记住：共现关系导致的LVLM幻觉是本项目的长期核心研究方向。",
        source_module="paper_reading",
        source_path="data/papers/cooccurrence_hallucination.md",
        source_title="Co-occurrence and Hallucination",
    )
    ok = True
    ok &= check("ok" in result, "result has 'ok'")
    ok &= check("record" in result, "result has 'record'")
    ok &= check("decision" in result, "result has 'decision'")
    record = result["record"]
    ok &= check(_f(record, "memory_level") == "long_term",
                f"memory_level={_f(record, 'memory_level')} (expected long_term from explicit instruction)")
    decision = result["decision"]
    ok &= check("level_reason" in decision, "decision has level_reason")
    ok &= check("owner_agent" in decision, "decision has owner_agent")

    summary = _f(record, "summary", "")
    print(f"  record summary: {str(summary)[:100]}")
    print(f"  decision: {decision}")
    if not _STORE_AVAILABLE:
        print(f"  NOTE: store.py not available — record built but not persisted.")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 10 — infer_owner_agent from content fallback
# ═══════════════════════════════════════════════════════════════════════════

def test_owner_from_content():
    section("Test 10: infer_owner_agent from content (no source_module)")
    ok = True
    ok &= check(infer_owner_agent("", "", "this paper explores bias") == "paper_agent",
                "paper content → paper_agent")
    ok &= check(infer_owner_agent("", "", "the experiment shows results") == "experiment_agent",
                "experiment content → experiment_agent")
    ok &= check(infer_owner_agent("", "", "the code has a bug") == "code_agent",
                "code content → code_agent")
    ok &= check(infer_owner_agent("", "", "random text") == "memory_agent",
                "unknown content → memory_agent")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 11 — memory_type from content
# ═══════════════════════════════════════════════════════════════════════════

def test_memory_type_inference():
    section("Test 11: infer_memory_type from content keywords")
    ok = True
    ok &= check(infer_memory_type("", "fix the bug in the code") == "code_note",
                "bug → code_note")
    ok &= check(infer_memory_type("", "TODO: build stereotype library") == "todo",
                "TODO → todo")
    ok &= check(infer_memory_type("", "my preference is to use pymupdf") == "user_preference",
                "preference → user_preference")
    ok &= check(infer_memory_type("unknown_source", "general content") == "general_note",
                "unknown → general_note")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 12 — tags coverage
# ═══════════════════════════════════════════════════════════════════════════

def test_tags_coverage():
    section("Test 12: infer_tags bilingual coverage")
    cn = infer_tags("大语言模型的幻觉问题在视觉语言模型中表现为偏见和刻板印象共现")
    en = infer_tags("hallucination in VLM exhibits bias and stereotype co-occurrence with fairness concerns")

    ok = True
    ok &= check(len(cn) > 0, f"Chinese tags: {cn}")
    ok &= check(len(en) > 0, f"English tags: {en}")
    ok &= check(any("hallucination" in t.lower() or "幻觉" in t for t in cn + en),
                "hallucination/幻觉 detected")
    ok &= check(any("bias" in t.lower() or "偏见" in t for t in cn + en),
                "bias/偏见 detected")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 60)
    print("  Memory Writer — Test Suite")
    print("=" * 60)
    print(f"  schema.py available: {_SCHEMA_AVAILABLE}")
    print(f"  store.py available:   {_STORE_AVAILABLE}")
    if not _SCHEMA_AVAILABLE:
        print("  NOTE: schema.py not merged yet — using fallback record builder.")
    if not _STORE_AVAILABLE:
        print("  NOTE: store.py not merged yet — records built but not persisted.")

    results = {}
    results["explicit_long"] = test_explicit_long_term()
    results["explicit_mid"] = test_explicit_mid_term()
    results["explicit_short"] = test_explicit_short_term()
    results["paper_source"] = test_paper_source()
    results["claim_source"] = test_claim_source()
    results["ppt_progress"] = test_ppt_progress_source()
    results["mid_to_long"] = test_mid_to_long_promotion()
    results["no_promotion"] = test_no_promotion_when_completed()
    results["write_memory"] = test_write_memory()
    results["owner_content"] = test_owner_from_content()
    results["type_inference"] = test_memory_type_inference()
    results["tags"] = test_tags_coverage()

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
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
