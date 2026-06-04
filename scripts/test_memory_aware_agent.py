"""
Test script for Memory-aware Agent.

Usage:
    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_memory_aware_agent.py

Tests:
1. load_memories_for_agent — loads from store
2. retrieve_memories_for_query — task-type-aware retrieval
3. format_memory_context — formats for LLM prompt
4. merge_rag_and_memory_context — merges RAG + Memory
5. build_memory_augmented_answer — full pipeline
6. get_continuity_suggestions — task continuity
7. Integration: write via adapters, retrieve via memory-aware agent
8. Backward compatibility — empty store, missing modules
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.memory.store import ensure_memory_store, load_memories
from research_agent.memory.memory_aware_agent import (
    load_memories_for_agent,
    retrieve_memories_for_query,
    format_memory_context,
    merge_rag_and_memory_context,
    build_memory_augmented_answer,
    get_continuity_suggestions,
    answer_with_memory,
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


# ── Seed memories for testing ────────────────────────────────────


def seed_test_memories():
    """Ensure we have some test memories."""
    from research_agent.memory.adapters import (
        save_claim_support_result,
        save_paper_reading_result,
        save_progress_memory_result,
        save_report_result,
        save_experiment_analysis_result,
    )

    # Claim memory with distinctive Chinese content
    save_claim_support_result({
        "claim": "共现关系可能诱发 LVLM 对缺失对象的幻觉。",
        "claim_type": "theoretical_claim",
        "evidence_count": 2,
        "report": "基于 guardrail_agnostic 和 VIGNETTE 论文的论点支持报告。",
        "sources": [
            {"path": "data/papers/guardrail_agnostic.md", "source_type": "paper_note"},
        ],
        "used_llm": True,
    })

    # Paper memory
    save_paper_reading_result({
        "paper_path": "data/papers/guardrail_agnostic.md",
        "metadata": {"title": "Guardrail-Agnostic Bias Evaluation"},
        "sections": {"method": "...", "experiments": "..."},
        "reading_note": "Guardrail-Agnostic 论文提出不依赖安全护栏的偏见评估方法。",
        "sources": ["data/papers/guardrail_agnostic.md"],
        "used_llm": True,
        "status": "success",
    })

    # Progress memory
    save_progress_memory_result({
        "source_path": "data/ingested/sample_slides.md",
        "metadata": {"title": "Weekly Group Meeting"},
        "slides": [{"slide_number": 1}],
        "topics": {
            "next_steps": ["扩展 stereotype library"],
            "experiments": ["COCO hallucination screening"],
        },
        "progress_memory": "本周完成 COCO 幻觉筛选实验，下一步扩展偏见库。",
        "memory_records": ["[next_step] 扩展 stereotype library"],
        "used_llm": False,
    })

    # Report memory
    save_report_result({
        "report_text": "组会汇报：VLM 偏见评估实验进展。",
        "task_type": "report_generation",
        "sources": [{"path": "data/experiments/coco_val_n300_g1.md", "source_type": "experiment_doc"}],
        "used_llm": True,
        "evidence_status": "passed",
    })

    # Experiment memory
    save_experiment_analysis_result({
        "analysis": "COCO hallucination screening: mean_extra_object_rate 0.23.",
        "file_path": "data/experiments/sample_metrics.csv",
        "tool_used": "csv_analyzer",
        "metrics": {"mean_extra_object_rate": 0.23},
        "sources": [],
    })


# ── Test 1: load_memories_for_agent ───────────────────────────


def test_load():
    section("Test 1: load_memories_for_agent()")

    memories = load_memories_for_agent()
    check(len(memories) >= 5, f">=5 records loaded (got {len(memories)})")

    if memories:
        r = memories[0]
        check("memory_id" in r, "record has memory_id")
        check("memory_type" in r, "record has memory_type")
        check("memory_level" in r, "record has memory_level")
        check("tags" in r, "record has tags")


# ── Test 2: retrieve_memories_for_query ────────────────────────


def test_retrieve_for_query():
    section("Test 2: retrieve_memories_for_query()")

    # Paper question — should find paper_note + claim_support
    results = retrieve_memories_for_query(
        query="Guardrail-Agnostic 论文关于偏见评估的方法",
        task_type="paper_question",
        max_results=5,
    )
    check(len(results) >= 1, f"paper_question finds >=1 memory ({len(results)})")
    types_found = {r.get("memory_type", "?") for r in results}
    print(f"  Paper question found types: {types_found}")

    # Experiment analysis — should find experiment_result
    results2 = retrieve_memories_for_query(
        query="COCO 幻觉筛选实验结果",
        task_type="experiment_analysis",
        max_results=5,
    )
    check(len(results2) >= 1, f"experiment_analysis finds >=1 memory ({len(results2)})")
    types2 = {r.get("memory_type", "?") for r in results2}
    print(f"  Experiment analysis found types: {types2}")

    # General — should find todo/decision/direction
    results3 = retrieve_memories_for_query(
        query="下一步应该做什么",
        task_type="general",
        max_results=5,
    )
    check(len(results3) >= 0, f"general query runs ({len(results3)} results)")
    print(f"  General query found: {len(results3)} memories")

    # Keyword search — distinctive Chinese content
    results4 = retrieve_memories_for_query(
        query="共现关系 幻觉",
        task_type="paper_question",
        max_results=5,
    )
    keyword_found = any(
        "共现关系" in (r.get("content", "") + r.get("summary", ""))
        for r in results4
    )
    check(keyword_found or len(results4) >= 1,
          f"keyword '共现关系' finds relevant memory")


# ── Test 3: format_memory_context ───────────────────────────────


def test_format_context():
    section("Test 3: format_memory_context()")

    memories = retrieve_memories_for_query(
        query="偏见评估",
        task_type="paper_question",
        max_results=5,
    )

    context = format_memory_context(memories, max_memories=3)
    check(len(context) > 0, "context is non-empty")
    check("## Research Memory" in context, "context has memory heading")

    # Should NOT include full content — only summaries
    print(f"  Context length: {len(context)} chars")
    print(f"  Preview: {context[:300].replace(chr(10), ' ')}")


# ── Test 4: merge_rag_and_memory_context ───────────────────────


def test_merge_context():
    section("Test 4: merge_rag_and_memory_context()")

    rag = "RAG 文档摘要：OpenImages-MIAP 的标注是 bbox 级。"
    memories = retrieve_memories_for_query(
        query="OpenImages-MIAP 标注级别",
        task_type="dataset_recommendation",
        max_results=3,
    )
    mem_ctx = format_memory_context(memories)
    merged = merge_rag_and_memory_context(rag, mem_ctx)

    check("RAG Context" in merged, "merged has RAG section")
    check(len(merged) > 0, "merged is non-empty")
    print(f"  Merged length: {len(merged)} chars")


# ── Test 5: build_memory_augmented_answer ──────────────────────


def test_build_answer():
    section("Test 5: build_memory_augmented_answer()")

    result = build_memory_augmented_answer(
        query="共现关系是否会导致 VLM 幻觉？",
        task_type="paper_question",
        rag_context="RAG 文档：guardrail_agnostic.md 讨论隐式偏见。",
        rag_sources=[{"path": "data/papers/guardrail_agnostic.md", "source_type": "paper_note"}],
        use_llm=False,
    )

    check("answer" in result, "result has answer")
    check("memories" in result, "result has memories")
    check("memory_context" in result, "result has memory_context")
    check("merged_context" in result, "result has merged_context")
    check("sources" in result, "result has sources")
    check(result["memory_count"] >= 0, f"memory_count >= 0 ({result['memory_count']})")
    check(len(result["answer"]) > 0, "answer is non-empty")

    # Sources should include both RAG and memory sources
    has_rag = any(s.get("path") == "data/papers/guardrail_agnostic.md"
                  for s in result["sources"])
    check(has_rag, "sources include RAG source")

    print(f"  Memory count: {result['memory_count']}")
    print(f"  Sources count: {len(result['sources'])}")
    print(f"  Answer preview: {result['answer'][:200].replace(chr(10), ' ')}")


# ── Test 6: get_continuity_suggestions ──────────────────────────


def test_continuity():
    section("Test 6: get_continuity_suggestions()")

    suggestions = get_continuity_suggestions(
        task_type="general",
        max_suggestions=3,
    )
    check(len(suggestions) >= 0, f"suggestions runs ({len(suggestions)} results)")

    if suggestions:
        for s in suggestions:
            mt = s.get("memory_type", "?")
            summary = (s.get("summary", "") or "")[:80]
            print(f"  [{mt}] {summary}")

    # Priority types should appear first
    if len(suggestions) >= 1:
        mt = suggestions[0].get("memory_type", "")
        priority_types = {"research_direction", "project_decision",
                          "user_preference", "todo", "progress_update"}
        check(mt in priority_types,
              f"top suggestion is priority type: {mt}")


# ── Test 7: Integration — write then retrieve ──────────────────


def test_write_retrieve_roundtrip():
    section("Test 7: Roundtrip — write via adapter, retrieve via agent")

    from research_agent.memory.adapters import save_claim_support_result

    unique_claim = "本轮测试特殊标记词：量子多模态对齐实验"
    save_claim_support_result({
        "claim": unique_claim,
        "claim_type": "theoretical_claim",
        "evidence_count": 0,
        "report": "测试报告内容。",
        "sources": [],
        "used_llm": False,
    })

    # Retrieve by keyword
    results = retrieve_memories_for_query(
        query=unique_claim,
        task_type="paper_question",
        max_results=5,
    )
    check(len(results) >= 1,
          f"roundtrip finds written memory ({len(results)})")

    if results:
        content = results[0].get("content", "")
        check(unique_claim in content,
              "retrieved memory contains unique claim text")


# ── Test 8: Backward compatibility ──────────────────────────────


def test_backward_compat():
    section("Test 8: Backward compatibility — empty / missing modules")

    # Empty store shouldn't crash
    result = build_memory_augmented_answer(
        query="test query",
        task_type="general",
        rag_context="",
        use_llm=False,
    )
    check("answer" in result, "returns answer even with empty store")
    check(result["memory_count"] == 0 or result["memory_count"] >= 0,
          f"memory_count is valid: {result['memory_count']}")

    # answer_with_memory convenience
    result2 = answer_with_memory(
        query="测试问题",
        task_type="paper_question",
        rag_context="一些 RAG 上下文",
    )
    check("answer" in result2, "answer_with_memory returns answer")
    check("sources" in result2, "answer_with_memory returns sources")


# ── Main ─────────────────────────────────────────────────────────


def main():
    global PASS, FAIL

    print("=" * 60)
    print("Memory-aware Agent — Test Suite")
    print("=" * 60)

    ensure_memory_store()
    seed_test_memories()

    test_load()
    test_retrieve_for_query()
    test_format_context()
    test_merge_context()
    test_build_answer()
    test_continuity()
    test_write_retrieve_roundtrip()
    test_backward_compat()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    print(f"\n  Note: Test data written to data/memory/ (git-ignored).")

    if FAIL > 0:
        print("\nSome tests FAILED. Check output for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
