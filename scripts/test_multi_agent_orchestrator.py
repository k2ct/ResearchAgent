"""
Integration test: Multi-Agent Orchestrator → Profiles → Handoff → Memory.

Usage::

    cd F:/ResearchAgent
    HF_HUB_OFFLINE=1 ENABLE_MULTI_AGENT=true ./.conda/python.exe scripts/test_multi_agent_orchestrator.py
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Force HuggingFace offline (models must be pre-cached via build_index.py)
os.environ["HF_HUB_OFFLINE"] = "1"
# Enable multi-agent for this test
os.environ["ENABLE_MULTI_AGENT"] = "true"

# [FIX] Wrap imports that may fail due to missing HF cache or path issues.
# build_graph triggers sentence_transformers loading → requires HF_HUB_OFFLINE.
# run_cli imports build_graph at module level → fallback if unavailable.
_BUILD_GRAPH = None
_create_initial_state = None

try:
    from research_agent.graph.workflow import build_graph as _BUILD_GRAPH
except Exception as _e:
    _BUILD_GRAPH_ERROR = str(_e)

try:
    from run_cli import create_initial_state as _create_initial_state
except ImportError:
    pass


def build_graph():
    """Safe wrapper: returns None if the graph module failed to import."""
    if _BUILD_GRAPH is None:
        return None
    return _BUILD_GRAPH()


def create_initial_state(query: str) -> dict:
    """Safe wrapper: fallback initial state if run_cli is unavailable."""
    if _create_initial_state is not None:
        return _create_initial_state(query)
    # Fallback: minimal state for tests that don't need the full state
    return {
        "query": query,
        "task_type": "",
        "result": "",
        "final_answer": "",
        "classifier_source": "",
        "route_reason": "",
        "retrieved_docs": [],
        "sources": [],
        "tool_used": "none",
        "tool_result": {},
        "tool_result_text": "",
        "evidence_status": "",
        "evidence_reason": "",
        "evidence_warnings": [],
        "memory_context": "",
        "retrieved_memories": [],
        "memory_count": 0,
        "memory_used": False,
        "memory_error": "",
        "multi_agent_enabled": False,
        "primary_agent": "",
        "handoff_plan": {},
        "handoff_results": [],
        "handoff_summary": "",
        "handoff_sources": [],
        "handoff_memory_ids": [],
        "handoff_count": 0,
        "memory_written": False,
        "memory_write_error": "",
    }


from research_agent.agents.orchestrator import (
    run_multi_agent_pipeline,
    is_multi_agent_enabled,
)
from research_agent.agents.profiles import select_agent_for_task, get_agent_profile
from research_agent.agents.handoff import (
    build_handoff_plan, aggregate_handoff_results,
    HandoffPlan, HandoffResult,
)
from research_agent.memory.store import load_memories
from research_agent.memory.privacy_scope import filter_accessible

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


# ═══════════════════════════════════════════════════════════════════════════
# Test 1 — multi-agent toggle
# ═══════════════════════════════════════════════════════════════════════════

def test_toggle():
    section("Test 1: ENABLE_MULTI_AGENT toggle")
    ok = True
    ok &= check(is_multi_agent_enabled(), "ENABLE_MULTI_AGENT=true → enabled")
    # Verify the graph builder reads it
    graph = build_graph()
    ok &= check(graph is not None, "graph builds with multi-agent enabled")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 2 — agent selection for 5 task types
# ═══════════════════════════════════════════════════════════════════════════

def test_agent_selection():
    section("Test 2: select_agent_for_task — 5 task types")
    cases = [
        ("paper_question", "论文 bias evaluation 的最新方法", "paper_agent"),
        ("experiment_analysis", "分析 coco_val_n300_g1 实验结果", "experiment_agent"),
        ("claim_support", "LVLM hallucination is caused by co-occurrence bias", "claim_agent"),
        ("report_generation", "帮我生成 coco_val_n300_g1 实验的组会汇报文本", "report_agent"),
        ("general", "我今天应该怎么安排科研任务", "general_agent"),
    ]
    ok = True
    for task_type, query, expected in cases:
        sel = select_agent_for_task(task_type=task_type, query=query)
        ok &= check(sel["agent_id"] == expected,
                    f"{task_type} → {sel['agent_id']} (expected {expected})")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 3 — handoff plan generation for 5 task types
# ═══════════════════════════════════════════════════════════════════════════

def test_handoff_plans():
    section("Test 3: build_handoff_plan — 5 task types")
    cases = [
        ("paper_question", "论文 LVLM bias evaluation"),
        ("experiment_analysis", "分析 coco_val_n300_g1"),
        ("claim_support", "co-occurrence causes hallucination"),
        ("report_generation", "生成组会汇报"),
        ("general", "今天怎么安排科研任务"),
    ]
    ok = True
    for task_type, query in cases:
        plan = build_handoff_plan(root_query=query, task_type=task_type)
        ok &= check(len(plan.handoffs) >= 1,
                    f"{task_type}: {len(plan.handoffs)} handoffs (≥1)")
        # All handoffs should be valid
        from research_agent.agents.handoff import validate_handoff_request
        for h in plan.handoffs:
            errors = validate_handoff_request(h)
            real_errors = [e for e in errors if not e.startswith("WARNING")]
            ok &= check(len(real_errors) == 0,
                        f"  {h.to_agent}: valid (errors={real_errors})")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 4 — full orchestrator pipeline (RAG + Memory)
# ═══════════════════════════════════════════════════════════════════════════

def test_orchestrator_pipeline():
    section("Test 4: run_multi_agent_pipeline with RAG + Memory")
    # Simulate RAG docs
    rag_docs = [
        {"metadata": {"path": "data/papers/guardrail_agnostic.md",
                       "source_type": "paper_note",
                       "title": "Guardrail-Agnostic Bias Evaluation"}},
        {"metadata": {"path": "data/experiments/coco_val_n300_g1.md",
                       "source_type": "experiment_doc",
                       "title": "COCO n300 g1 Hallucination Screening"}},
    ]
    # Load real memories
    memories = load_memories()
    mem_dicts = []
    for m in memories[:10]:
        if isinstance(m, dict):
            mem_dicts.append(m)
        else:
            mem_dicts.append({
                "memory_id": getattr(m, "memory_id", ""),
                "memory_type": getattr(m, "memory_type", ""),
                "summary": getattr(m, "summary", ""),
                "tags": getattr(m, "tags", []),
                "owner_agent": getattr(m, "owner_agent", ""),
                "memory_scope": getattr(m, "memory_scope", "private"),
            })

    result = run_multi_agent_pipeline(
        query="请总结最近关于 LVLM bias 和 hallucination 的研究进展",
        task_type="report_generation",
        rag_docs=rag_docs,
        memory_context="Previous research on multimodal bias evaluation",
        retrieved_memories=mem_dicts,
        auto_write_memory=True,
    )

    ok = True
    ok &= check(bool(result["primary_agent"]),
                f"primary_agent={result['primary_agent']}")
    ok &= check(result["handoff_count"] >= 2,
                f"handoff_count={result['handoff_count']} (≥2 for report_generation)")
    ok &= check(bool(result["combined_answer"]),
                f"combined_answer length={len(result['combined_answer'])}")
    ok &= check("Multi-Agent Result" in result["combined_answer"],
                "combined_answer contains 'Multi-Agent Result'")
    ok &= check(len(result["handoff_results"]) > 0,
                f"handoff_results={len(result['handoff_results'])} items")
    ok &= check(bool(result["handoff_summary"]),
                f"handoff_summary={result['handoff_summary'][:100]}")

    print(f"  primary_agent: {result['primary_agent']}")
    print(f"  handoff_count: {result['handoff_count']}")
    print(f"  handoff_summary: {result['handoff_summary']}")
    print(f"  memory_written: {result['memory_written']}")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 5 — LangGraph integration (multi-agent workflow)
# ═══════════════════════════════════════════════════════════════════════════

def test_langgraph_integration():
    section("Test 5: LangGraph workflow with multi-agent node")
    graph = build_graph()
    state = create_initial_state("请总结最近关于 LVLM bias 的研究进展")

    ok = True
    try:
        result = graph.invoke(state)
        ok &= check("final_answer" in result, "graph produces final_answer")
        ok &= check("task_type" in result, "graph produces task_type")
        # Multi-agent fields should be present when enabled
        ok &= check("handoff_count" in result, "handoff_count in state")
        if result.get("multi_agent_enabled"):
            ok &= check(result.get("handoff_count", 0) >= 0, "handoff_count >= 0")
            print(f"  task_type: {result.get('task_type')}")
            print(f"  primary_agent: {result.get('primary_agent')}")
            print(f"  handoff_count: {result.get('handoff_count')}")
            print(f"  handoff_summary: {result.get('handoff_summary', '')[:120]}")
            print(f"  memory_used: {result.get('memory_used')}")
            print(f"  memory_count: {result.get('memory_count')}")
            print(f"  memory_written: {result.get('memory_written')}")
        else:
            print(f"  multi_agent_enabled=False (toggle off)")
    except Exception as e:
        ok &= check(False, f"graph.invoke failed: {e}")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 6 — Memory write-back verification
# ═══════════════════════════════════════════════════════════════════════════

def test_memory_writeback():
    section("Test 6: Memory write-back from orchestrator")
    from research_agent.memory.store import query_memories

    # Query for records written by the orchestrator
    found = query_memories(tags=["experiment", "benchmark", "COCO"])
    ok = True
    ok &= check(len(found) >= 1,
                f"query finds {len(found)} records (adapter-written)")

    # Verify privacy scope compatibility
    all_records = load_memories()
    if all_records:
        # Check that coordinator can see everything
        visible_coord = filter_accessible(all_records, "coordinator")
        ok &= check(len(visible_coord) == len(all_records),
                    f"coordinator sees all {len(visible_coord)}/{len(all_records)}")

        # paper_agent should see subset
        visible_paper = filter_accessible(all_records, "paper_agent")
        ok &= check(len(visible_paper) > 0, "paper_agent sees some records")
        print(f"  Memory store: {len(all_records)} total, "
              f"paper={len(visible_paper)}, coordinator={len(visible_coord)}")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 7 — compatibility with existing tests
# ═══════════════════════════════════════════════════════════════════════════

def test_existing_compatibility():
    section("Test 7: compatibility with existing modules")
    ok = True

    # Profiles import
    ok &= check(get_agent_profile("paper_agent") is not None,
                "get_agent_profile('paper_agent') works")
    ok &= check(get_agent_profile("experiment_agent") is not None,
                "get_agent_profile('experiment_agent') works")

    # Handoff create → validate → aggregate round-trip
    from research_agent.agents.handoff import (
        create_handoff_request, validate_handoff_request,
        HandoffPlan, HandoffResult,
    )
    req = create_handoff_request(
        from_agent="coordinator_agent", to_agent="paper_agent",
        task="Find papers on bias",
        input_text="LVLM bias and hallucination",
        task_type="paper_question",
    )
    errors = validate_handoff_request(req)
    real_errors = [e for e in errors if not e.startswith("WARNING")]
    ok &= check(len(real_errors) == 0, f"handoff request valid ({len(errors)} validation notes)")

    # Aggregate round-trip
    plan = HandoffPlan(plan_id="test_plan", root_task="paper_question",
                        root_query="test", handoffs=[req])
    results = [HandoffResult(
        handoff_id=req.handoff_id, from_agent=req.from_agent,
        to_agent=req.to_agent, status="completed",
        result_text="Paper notes on bias evaluation.",
        confidence=0.8, sources=[{"path": "data/papers/bias.md"}],
    )]
    agg = aggregate_handoff_results(plan, results)
    ok &= check(agg["completed"] == 1, "aggregate: 1 completed")
    ok &= check(len(agg["sources"]) == 1, "aggregate: 1 source")
    ok &= check("Paper Agent" in agg["combined_answer"],
                "aggregate: contains agent label")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 8 — graph builds without multi-agent (backward compat)
# ═══════════════════════════════════════════════════════════════════════════

def test_backward_compat():
    section("Test 8: backward compatibility — multi-agent disabled")
    # Temporarily disable
    saved = os.environ.pop("ENABLE_MULTI_AGENT", None)
    os.environ["ENABLE_MULTI_AGENT"] = "false"

    ok = True
    try:
        graph = build_graph()
        state = create_initial_state("OpenImages-MIAP 的性别标注是图像级还是 bbox 级？")
        result = graph.invoke(state)
        ok &= check("final_answer" in result, "graph produces final_answer (single-agent mode)")
        ok &= check(result.get("multi_agent_enabled", False) == False,
                    "multi_agent_enabled=False (not activated)")
    except Exception as e:
        ok &= check(False, f"single-agent graph failed: {e}")
    finally:
        if saved is not None:
            os.environ["ENABLE_MULTI_AGENT"] = saved
        else:
            os.environ["ENABLE_MULTI_AGENT"] = "true"

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 60)
    print("  Multi-Agent Orchestrator — Integration Test Suite")
    print("=" * 60)
    print(f"  ENABLE_MULTI_AGENT: {os.getenv('ENABLE_MULTI_AGENT', 'false')}")

    results = {}
    results["toggle"] = test_toggle()
    results["agent_selection"] = test_agent_selection()
    results["handoff_plans"] = test_handoff_plans()
    results["orchestrator"] = test_orchestrator_pipeline()
    results["langgraph"] = test_langgraph_integration()
    results["memory_writeback"] = test_memory_writeback()
    results["compatibility"] = test_existing_compatibility()
    results["backward_compat"] = test_backward_compat()

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    all_passed = True
    for name, ok_val in results.items():
        status = PASS if ok_val else FAIL
        print(f"  {status}: {name}")
        if not ok_val:
            all_passed = False

    print()
    if all_passed:
        print("  All tests passed!")
    else:
        print("  Some tests FAILED — see details above.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
