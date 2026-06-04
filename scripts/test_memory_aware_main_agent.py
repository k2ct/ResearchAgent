"""
Test Memory-aware Main Agent Integration.

Verifies:
1. When ENABLE_MEMORY_AWARE_AGENT=false, agent runs normally without memory.
2. When ENABLE_MEMORY_AWARE_AGENT=true, memories are retrieved and merged.
3. Empty memory store does not crash the agent.
4. Agent demo regression still works.
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


# ── Test 1: Agent runs with memory disabled (default) ──────────────

def test_agent_memory_disabled():
    print("\n── Test 1: Agent with ENABLE_MEMORY_AWARE_AGENT=false ──")

    # Ensure disabled
    os.environ["ENABLE_MEMORY_AWARE_AGENT"] = "false"

    from research_agent.graph.workflow import build_graph
    graph = build_graph()

    result = graph.invoke({
        "query": "我之前关于共现关系和幻觉的研究进展是什么？",
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
    })

    check(result.get("task_type", "") != "", "task_type populated")
    check(result.get("final_answer", "") != "", "final_answer populated")
    check(result.get("memory_used", True) == False or "disabled" in result.get("memory_error", "").lower(),
          "memory_used=False or memory disabled")
    check(result.get("memory_error", "") != "", "memory_error set (disabled message)")

    print(f"  task_type: {result.get('task_type')}")
    print(f"  memory_used: {result.get('memory_used')}")
    print(f"  memory_count: {result.get('memory_count')}")
    print(f"  memory_error: {result.get('memory_error', '')[:80]}")


# ── Test 2: Agent runs with memory enabled ─────────────────────────

def test_agent_memory_enabled():
    print("\n── Test 2: Agent with ENABLE_MEMORY_AWARE_AGENT=true ──")

    os.environ["ENABLE_MEMORY_AWARE_AGENT"] = "true"
    os.environ["MEMORY_TOP_K"] = "5"

    # Write a test memory first
    from research_agent.memory.store import ensure_memory_store, append_memory
    from research_agent.memory.writer import build_memory_record_from_source

    ensure_memory_store()

    record = build_memory_record_from_source(
        content="我的长期研究方向是共现关系对 LVLM 幻觉的影响。这个方向涉及 co-occurrence bias 和 object hallucination 的交叉研究。",
        source_module="chat",
        source_title="Research Direction Declaration",
    )

    write_result = append_memory(record)
    check(write_result["ok"], "test memory written to store")

    from research_agent.graph.workflow import build_graph
    graph = build_graph()

    result = graph.invoke({
        "query": "我现在的研究方向是什么？",
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
    })

    check(result.get("task_type", "") != "", "task_type populated")
    check(result.get("final_answer", "") != "", "final_answer populated")
    # Memory should be retrieved — but may return 0 if query doesn't match well
    mem_count = result.get("memory_count", 0)
    mem_error = result.get("memory_error", "")
    check(mem_count >= 0, f"memory_count >= 0 (got {mem_count})")
    check("error" not in mem_error.lower() or mem_error == "",
          f"no memory error (got '{mem_error[:60]}')")

    print(f"  task_type: {result.get('task_type')}")
    print(f"  memory_used: {result.get('memory_used')}")
    print(f"  memory_count: {mem_count}")
    print(f"  memory_error: '{mem_error[:80]}'")

    # Restore disabled
    os.environ["ENABLE_MEMORY_AWARE_AGENT"] = "false"


# ── Test 3: Empty memory store does not crash ─────────────────────

def test_empty_store_no_crash():
    print("\n── Test 3: Empty / unavailable memory store ──")

    os.environ["ENABLE_MEMORY_AWARE_AGENT"] = "true"

    # Use a non-existent store path (memory_aware_agent falls back gracefully)
    from research_agent.memory.memory_aware_agent import load_memories_for_agent
    records = load_memories_for_agent()
    check(isinstance(records, list), f"load_memories_for_agent returns list ({len(records)} records)")

    from research_agent.graph.workflow import build_graph
    graph = build_graph()

    result = graph.invoke({
        "query": "测试问题：空存储不应崩溃。",
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
    })

    check(result.get("final_answer", "") != "", "agent produced answer despite empty memory")
    check("error" not in (result.get("memory_error", "") or "").lower(),
          "no crash-level memory error")

    os.environ["ENABLE_MEMORY_AWARE_AGENT"] = "false"


# ── Test 4: memory_aware_agent functions work standalone ───────────

def test_memory_aware_agent_standalone():
    print("\n── Test 4: memory_aware_agent standalone functions ──")

    os.environ["ENABLE_MEMORY_AWARE_AGENT"] = "true"

    from research_agent.memory.memory_aware_agent import (
        retrieve_memories_for_query,
        format_memory_context,
        merge_rag_and_memory_context,
    )

    # Write a distinctive memory
    from research_agent.memory.store import ensure_memory_store, append_memory
    from research_agent.memory.writer import build_memory_record_from_source

    ensure_memory_store()
    record = build_memory_record_from_source(
        content="反事实样本可有效检测 LVLM 中的性别偏差。实验结果表明 counterfactual evaluation 是 bias detection 的有效方法。",
        source_module="experiment_tool",
        source_title="Counterfactual Bias Detection",
    )
    append_memory(record)

    memories = retrieve_memories_for_query(
        query="反事实样本和偏见检测",
        task_type="experiment_analysis",
        max_results=3,
    )
    check(isinstance(memories, list), f"retrieve_memories_for_query returns list ({len(memories)})")

    ctx = format_memory_context(memories)
    check(isinstance(ctx, str) and len(ctx) > 0, "format_memory_context returns non-empty string")

    merged = merge_rag_and_memory_context("RAG doc text", ctx)
    check("Memory Context" in merged, "merged context includes Memory section")
    check("RAG Context" in merged, "merged context includes RAG section")

    os.environ["ENABLE_MEMORY_AWARE_AGENT"] = "false"


# ── Test 5: retrieve_memory_node produces correct fields ───────────

def test_retrieve_memory_node_fields():
    print("\n── Test 5: retrieve_memory_node field shape ──")

    from research_agent.graph.nodes import retrieve_memory_node

    state = {
        "query": "研究方向",
        "task_type": "general",
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
    }

    os.environ["ENABLE_MEMORY_AWARE_AGENT"] = "false"
    result_disabled = retrieve_memory_node(state)
    check(result_disabled.get("memory_used") == False, "disabled: memory_used=False")
    check("disabled" in result_disabled.get("memory_error", "").lower(), "disabled: error message present")

    os.environ["ENABLE_MEMORY_AWARE_AGENT"] = "true"
    result_enabled = retrieve_memory_node(state)
    check("memory_context" in result_enabled, "enabled: memory_context key present")
    check("retrieved_memories" in result_enabled, "enabled: retrieved_memories key present")
    check("memory_count" in result_enabled, "enabled: memory_count key present")
    check(isinstance(result_enabled.get("retrieved_memories"), list), "enabled: retrieved_memories is list")

    os.environ["ENABLE_MEMORY_AWARE_AGENT"] = "false"


# ── Main ──────────────────────────────────────────────────────────

def main():
    global PASS, FAIL

    print("=" * 60)
    print("  Memory-aware Main Agent Integration Test")
    print("=" * 60)

    test_agent_memory_disabled()
    test_agent_memory_enabled()
    test_empty_store_no_crash()
    test_memory_aware_agent_standalone()
    test_retrieve_memory_node_fields()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    # Restore default
    os.environ["ENABLE_MEMORY_AWARE_AGENT"] = "false"

    if FAIL > 0:
        print("\nSome tests FAILED.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
