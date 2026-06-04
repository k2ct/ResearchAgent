"""
Test script for Memory Integration Adapters.

Usage:
    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_memory_adapters.py

Tests:
1. Claim Support adapter
2. Paper Reading adapter
3. Progress Memory adapter
4. Report adapter
5. Experiment Analysis adapter
6. save_module_result dispatcher
7. auto_write=False (record built but not written)
8. Integration with Memory Store / Retriever
9. Integration with Privacy Scope
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.memory.store import (
    ensure_memory_store,
    load_memories,
    MEMORY_STORE_PATH,
)
from research_agent.memory.adapters import (
    save_claim_support_result,
    save_paper_reading_result,
    save_progress_memory_result,
    save_report_result,
    save_experiment_analysis_result,
    save_generic_result,
    save_module_result,
    _safe_get,
    _join_nonempty,
    _extract_sources_text,
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


# ── Helpers for test data ────────────────────────────────────────

def _make_fake_claim_result() -> dict:
    return {
        "claim": "共现关系可能诱发 LVLM 对缺失对象的幻觉。",
        "claim_type": "theoretical_claim",
        "evidence_count": 2,
        "report": "这是一个基于本地知识库的论点支持报告。\n\n## Evidence\n...",
        "sources": [
            {"path": "data/papers/guardrail_agnostic.md", "source_type": "paper_note"},
            {"path": "data/experiments/coco_val_n300_g1.md", "source_type": "experiment_doc"},
        ],
        "grouped_evidence": {"theory": [], "experiment": []},
        "used_llm": True,
        "llm_error": "",
    }


def _make_fake_paper_result() -> dict:
    return {
        "paper_path": "data/papers/test_paper.md",
        "metadata": {"title": "Test Paper on VLM Hallucination", "year": 2026},
        "sections": {"method": "method text", "experiments": "exp text"},
        "reading_note": "这是一篇关于 VLM 幻觉检测的结构化论文阅读笔记。\n\n## Method\n...",
        "sources": ["data/papers/test_paper.md"],
        "used_llm": True,
        "llm_error": "",
        "status": "success",
    }


def _make_fake_progress_result() -> dict:
    return {
        "source_path": "data/ingested/test_slides.md",
        "metadata": {"title": "Test Group Meeting 2026"},
        "slides": [{"slide_number": 1, "title": "Slide 1"}, {"slide_number": 2, "title": "Slide 2"}],
        "topics": {
            "next_steps": ["继续实验", "扩展 stereotype library"],
            "experiments": ["COCO hallucination screening"],
            "research_questions": ["如何评估 VLM 偏见"],
            "keywords": ["VLM", "hallucination"],
        },
        "progress_memory": "## Progress Summary\n本周完成了组会汇报，讨论了 VLM 偏见评估实验。",
        "memory_records": ["[next_step] 继续实验", "[experiment_update] COCO n300"],
        "used_llm": True,
        "llm_error": "",
    }


def _make_fake_report_result() -> dict:
    return {
        "report_text": "组会汇报：本周完成 COCO 幻觉筛选实验。",
        "task_type": "report_generation",
        "report_style": "group_meeting",
        "sources": [
            {"path": "data/experiments/coco_val_n300_g1.md", "source_type": "experiment_doc"},
        ],
        "used_llm": True,
        "evidence_status": "passed",
    }


def _make_fake_experiment_result() -> dict:
    return {
        "analysis": "CSV analysis shows mean_extra_object_rate 0.23.",
        "file_path": "data/experiments/sample_metrics.csv",
        "tool_used": "csv_analyzer",
        "metrics": {"mean_extra_object_rate": 0.23, "mean_precision": 0.82},
        "evidence_status": "passed",
        "sources": [
            {"path": "data/experiments/coco_val_n300_g1.md", "source_type": "experiment_doc"},
        ],
    }


# ── Helper Tests ──────────────────────────────────────────────────


def test_helpers():
    section("Test 0: Helper functions")

    check(_safe_get({"a": 1}, "a") == 1, "_safe_get: key exists")
    check(_safe_get({"a": 1}, "b", "default") == "default", "_safe_get: key missing")
    check(_safe_get(None, "a", "x") == "x", "_safe_get: None input")

    text = _join_nonempty(["hello", "", None, "world"])
    check("hello" in text and "world" in text, "_join_nonempty")

    sources = [
        {"path": "a/b.md", "source_type": "paper_note"},
        {"path": "c/d.md", "source_type": "experiment_doc"},
    ]
    sources_text = _extract_sources_text(sources)
    check("a/b.md" in sources_text, "_extract_sources_text includes path")
    check("paper_note" in sources_text, "_extract_sources_text includes source_type")


# ── Test 1: Claim Support adapter ──────────────────────────────────


def test_claim_support_adapter():
    section("Test 1: save_claim_support_result")

    result = save_claim_support_result(_make_fake_claim_result(), auto_write=True)

    check(result.get("ok"), f"ok: {result.get('ok')}")
    check(result.get("adapter") == "claim_support", f"adapter: {result.get('adapter')}")

    record = result.get("record")
    check(record is not None, "record exists")

    if isinstance(record, dict):
        mt = record.get("memory_type", "")
        owner = record.get("owner_agent", "")
        check(mt == "claim_support" or "claim" in mt,
              f"memory_type is claim_support: {mt}")
        check(owner == "claim_agent" or "claim" in owner,
              f"owner_agent is claim_agent: {owner}")
        content = record.get("content", "")
        check("共现关系" in content, "content contains claim text")
        print(f"  memory_type: {mt}")
        print(f"  owner_agent: {owner}")
        print(f"  memory_id: {record.get('memory_id', '?')[:20]}...")


# ── Test 2: Paper Reading adapter ─────────────────────────────────


def test_paper_reading_adapter():
    section("Test 2: save_paper_reading_result")

    result = save_paper_reading_result(_make_fake_paper_result(), auto_write=True)

    check(result.get("ok"), f"ok: {result.get('ok')}")
    check(result.get("adapter") == "paper_reading", f"adapter: {result.get('adapter')}")

    record = result.get("record")
    check(record is not None, "record exists")

    if isinstance(record, dict):
        mt = record.get("memory_type", "")
        owner = record.get("owner_agent", "")
        check(mt == "paper_note", f"memory_type is paper_note: {mt}")
        check(owner == "paper_agent" or "paper" in owner,
              f"owner_agent is paper_agent: {owner}")
        sp = record.get("source_path", "")
        check("test_paper.md" in sp, f"source_path contains paper path: {sp}")
        print(f"  memory_type: {mt}")
        print(f"  owner_agent: {owner}")
        print(f"  source_path: {sp}")


# ── Test 3: Progress Memory adapter ────────────────────────────────


def test_progress_memory_adapter():
    section("Test 3: save_progress_memory_result")

    result = save_progress_memory_result(_make_fake_progress_result(), auto_write=True)

    check(result.get("ok"), f"ok: {result.get('ok')}")
    check(result.get("adapter") == "progress_memory", f"adapter: {result.get('adapter')}")

    record = result.get("record")
    check(record is not None, "record exists")

    if isinstance(record, dict):
        mt = record.get("memory_type", "")
        owner = record.get("owner_agent", "")
        check(mt == "progress_update", f"memory_type is progress_update: {mt}")
        check(owner == "progress_agent" or "progress" in owner,
              f"owner_agent is progress_agent: {owner}")
        meta = record.get("metadata", {})
        check(meta.get("slide_count") == 2,
              f"slide_count in metadata: {meta.get('slide_count')}")
        print(f"  memory_type: {mt}")
        print(f"  owner_agent: {owner}")


# ── Test 4: Report adapter ─────────────────────────────────────────


def test_report_adapter():
    section("Test 4: save_report_result")

    result = save_report_result(_make_fake_report_result(), auto_write=True)

    check(result.get("ok"), f"ok: {result.get('ok')}")
    check(result.get("adapter") == "report_writer", f"adapter: {result.get('adapter')}")

    record = result.get("record")
    check(record is not None, "record exists")

    if isinstance(record, dict):
        mt = record.get("memory_type", "")
        owner = record.get("owner_agent", "")
        check(mt == "report_summary", f"memory_type is report_summary: {mt}")
        check(owner == "report_agent" or "report" in owner,
              f"owner_agent is report_agent: {owner}")
        print(f"  memory_type: {mt}")
        print(f"  owner_agent: {owner}")


# ── Test 5: Experiment Analysis adapter ────────────────────────────


def test_experiment_adapter():
    section("Test 5: save_experiment_analysis_result")

    result = save_experiment_analysis_result(_make_fake_experiment_result(), auto_write=True)

    check(result.get("ok"), f"ok: {result.get('ok')}")
    check(result.get("adapter") == "experiment_analysis",
          f"adapter: {result.get('adapter')}")

    record = result.get("record")
    check(record is not None, "record exists")

    if isinstance(record, dict):
        mt = record.get("memory_type", "")
        owner = record.get("owner_agent", "")
        check(mt == "experiment_result", f"memory_type is experiment_result: {mt}")
        check(owner == "experiment_agent" or "experiment" in owner,
              f"owner_agent is experiment_agent: {owner}")
        print(f"  memory_type: {mt}")
        print(f"  owner_agent: {owner}")


# ── Test 6: save_module_result dispatcher ──────────────────────────


def test_dispatcher():
    section("Test 6: save_module_result dispatcher")

    # Test each known module name
    tests = [
        ("claim_support", "claim_support"),
        ("paper_reading", "paper_reading"),
        ("ppt_progress", "progress_memory"),
        ("progress_memory", "progress_memory"),
        ("report_writer", "report_writer"),
        ("experiment_tool", "experiment_analysis"),
        ("experiment_analysis", "experiment_analysis"),
    ]

    for module_name, expected_adapter in tests:
        result = save_module_result({"report": "test"}, module_name, auto_write=True)
        actual = result.get("adapter", "")
        check(actual == expected_adapter,
              f"save_module_result('{module_name}') -> adapter='{expected_adapter}' (got '{actual}')")

    # Unknown module falls back to generic
    result = save_module_result({"content": "test"}, "unknown_module", auto_write=True)
    check(result.get("adapter") == "generic",
          f"unknown module falls back to generic: {result.get('adapter')}")


# ── Test 7: auto_write=False ───────────────────────────────────────


def test_auto_write_false():
    section("Test 7: auto_write=False")

    records_before = load_memories()
    count_before = len(records_before)

    result = save_claim_support_result(_make_fake_claim_result(), auto_write=False)

    check(result.get("ok"), f"ok: {result.get('ok')}")
    check(result.get("record") is not None, "record was built")

    # Verify store was NOT modified
    records_after = load_memories()
    count_after = len(records_after)
    check(count_before == count_after,
          f"store count unchanged ({count_before} → {count_after})")


# ── Test 8: Integration with Retriever ─────────────────────────────


def test_retriever_integration():
    section("Test 8: Integration with Memory Retriever")

    # Write a claim with distinctive content
    save_claim_support_result(_make_fake_claim_result(), auto_write=True)

    # Load and search
    from research_agent.memory.store import load_memories as store_load
    from research_agent.memory.retriever import retrieve_memories

    records = store_load()
    check(len(records) > 0, f"store has records: {len(records)}")

    # Search by keyword
    found = retrieve_memories(records, keyword="共现关系")
    check(len(found) >= 1,
          f"keyword '共现关系' finds >=1 records ({len(found)})")

    # Search by memory_type
    found2 = retrieve_memories(records, memory_type="claim_support")
    check(len(found2) >= 1,
          f"memory_type='claim_support' finds >=1 records ({len(found2)})")

    # Search by source_module
    found3 = retrieve_memories(records, source_module="claim_support")
    check(len(found3) >= 1,
          f"source_module='claim_support' finds >=1 records ({len(found3)})")

    if found:
        r = found[0]
        print(f"  Found: [{r.get('memory_id', '?')[:12]}...] {r.get('summary', '')[:80]}")


# ── Test 9: Integration with Privacy Scope ─────────────────────────


def test_privacy_integration():
    section("Test 9: Integration with Privacy Scope")

    from research_agent.memory.privacy_scope import (
        check_access,
        filter_accessible,
    )

    # Write a claim (should be claim_agent owner)
    save_claim_support_result(_make_fake_claim_result(), auto_write=True)

    records = load_memories()
    # Find the most recent claim_support record
    claim_records = [r for r in records if r.get("memory_type") == "claim_support"]
    if not claim_records:
        check(False, "No claim_support records found for privacy test")
        return

    record = claim_records[-1]  # most recent

    owner = record.get("owner_agent", "")
    print(f"  Record owner: {owner}")
    print(f"  Record scope: {record.get('memory_scope', '?')}")
    print(f"  Record shared_with: {record.get('shared_with', [])}")

    # Owner should have access
    can_owner = check_access(record, owner)
    check(can_owner, f"owner '{owner}' can access own record")

    # Coordinator (universal agent) should have access
    can_coord = check_access(record, "coordinator")
    check(can_coord, "coordinator (universal) can access")

    # Unrelated agent should NOT have access (private scope)
    scope = record.get("memory_scope", "private")
    if scope == "private":
        can_other = check_access(record, "code_agent")
        check(not can_other, "unrelated agent (code_agent) cannot access private record")

    # filter_accessible
    accessible = filter_accessible(records, owner)
    check(len(accessible) >= 1,
          f"filter_accessible returns >=1 records for owner ({len(accessible)})")

    print(f"  filter_accessible for '{owner}': {len(accessible)}/{len(records)} records")


# ── Main ───────────────────────────────────────────────────────────


def main():
    global PASS, FAIL

    print("=" * 60)
    print("Memory Integration Adapters — Test Suite")
    print("=" * 60)

    ensure_memory_store()

    test_helpers()
    test_claim_support_adapter()
    test_paper_reading_adapter()
    test_progress_memory_adapter()
    test_report_adapter()
    test_experiment_adapter()
    test_dispatcher()
    test_auto_write_false()
    test_retriever_integration()
    test_privacy_integration()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    print(f"\n  Note: Test data written to {MEMORY_STORE_PATH}.")
    print(f"  data/memory/*.jsonl is git-ignored.")

    if FAIL > 0:
        print("\nSome tests FAILED. Check output for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
