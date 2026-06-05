"""
Test script for Multi-Agent Trace & Evaluation.

Usage:
    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_multi_agent_tracing.py

Tests:
1. create_trace_from_orchestrator_result — trace fields complete
2. append_trace / load_traces — write and read back
3. summarize_trace — markdown output
4. evaluate_handoff_quality — strong
5. evaluate_handoff_quality — medium
6. evaluate_handoff_quality — weak
7. Trace file is git-ignored
8. Orchestrator integration: trace + trace_quality in result
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# Enable multi-agent for orchestrator test
os.environ["ENABLE_MULTI_AGENT"] = "true"

from research_agent.agents.tracing import (
    MultiAgentTrace,
    create_trace_from_orchestrator_result,
    append_trace,
    load_traces,
    summarize_trace,
    evaluate_handoff_quality,
    trace_and_evaluate,
    _TRACES_DIR,
    _TRACES_PATH,
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


# ── Helper: fake orchestrator result ──────────────────────────────

def _fake_orchestrator_result(**overrides) -> dict:
    """Build a fake orchestrator result for testing."""
    result = {
        "primary_agent": "coordinator_agent",
        "handoff_plan": {
            "plan_id": "plan_test_001",
            "root_task": "paper_question",
            "handoff_count": 3,
            "targets": ["progress_agent", "paper_agent", "report_agent"],
        },
        "handoff_results": [
            {
                "handoff_id": "ho_001",
                "to_agent": "progress_agent",
                "status": "completed",
                "result_text": "Recent progress: completed COCO screening.",
                "confidence": 0.85,
                "sources_count": 2,
            },
            {
                "handoff_id": "ho_002",
                "to_agent": "paper_agent",
                "status": "completed",
                "result_text": "Found guardrail_agnostic paper notes.",
                "confidence": 0.90,
                "sources_count": 1,
            },
            {
                "handoff_id": "ho_003",
                "to_agent": "report_agent",
                "status": "failed",
                "error": "Report generation skipped.",
                "confidence": 0.0,
                "sources_count": 0,
            },
        ],
        "handoff_sources": [
            {"path": "data/papers/guardrail_agnostic.md"},
            {"path": "data/experiments/coco_val_n300_g1.md"},
        ],
        "handoff_memory_ids": ["mem_001", "mem_002", "mem_003"],
        "combined_answer": "# Multi-Agent Result\n\n## Progress Agent\n...",
        "memory_written": True,
        "memory_write_error": "",
        "handoff_count": 3,
        "handoff_summary": "Multi-Agent: coordinator_agent coordinated 3 handoffs",
    }
    result.update(overrides)
    return result


# ── Test 1: create_trace ────────────────────────────────────────


def test_create_trace():
    section("Test 1: create_trace_from_orchestrator_result")

    result = _fake_orchestrator_result()
    trace = create_trace_from_orchestrator_result(
        result,
        query="根据组会进展生成汇报大纲",
        task_type="report_generation",
    )

    check("trace_id" in trace, "trace_id present")
    check(trace["query"] == "根据组会进展生成汇报大纲", "query stored")
    check(trace["task_type"] == "report_generation", "task_type stored")
    check(trace["primary_agent"] == "coordinator_agent", "primary_agent stored")
    check(len(trace["handoff_results"]) == 3, "3 handoff results stored")
    check(len(trace["sources"]) == 2, "2 sources stored")
    check(len(trace["memory_ids"]) == 3, "3 memory_ids stored")
    check(trace["memory_written"] == True, "memory_written stored")
    check(len(trace["final_answer_preview"]) > 0, "final_answer_preview non-empty")
    check(len(trace["created_at"]) > 0, "created_at auto-filled")
    # Should have 1 error (failed handoff)
    check(len(trace["errors"]) == 1, f"1 error recorded (got {len(trace['errors'])})")

    print(f"  trace_id: {trace['trace_id'][:20]}...")


# ── Test 2: append_trace / load_traces ──────────────────────────


def test_append_load():
    section("Test 2: append_trace / load_traces")

    # Clear existing traces for clean test
    if _TRACES_PATH.exists():
        _TRACES_PATH.write_text("", encoding="utf-8")

    result = _fake_orchestrator_result()
    trace = create_trace_from_orchestrator_result(result, query="test query")

    append_result = append_trace(trace)
    check(append_result["ok"], f"append ok: {append_result.get('ok')}")

    append_trace(create_trace_from_orchestrator_result(
        _fake_orchestrator_result(), query="query 2"
    ))
    append_trace(create_trace_from_orchestrator_result(
        _fake_orchestrator_result(), query="query 3"
    ))

    loaded = load_traces(limit=10)
    check(len(loaded) == 3, f"loaded 3 traces (got {len(loaded)})")
    check(loaded[0]["query"] == "test query", "first trace query matches")
    check(_TRACES_PATH.exists(), "traces file exists")


# ── Test 3: summarize_trace ──────────────────────────────────────


def test_summarize():
    section("Test 3: summarize_trace — markdown output")

    result = _fake_orchestrator_result()
    trace = create_trace_from_orchestrator_result(result, query="test")

    summary = summarize_trace(trace)
    check(len(summary) > 0, "summary non-empty")
    check("# Multi-Agent Trace" in summary, "title present")
    check("## Query" in summary, "Query section")
    check("## Task" in summary, "Task section")
    check("## Handoff Plan" in summary, "Handoff Plan section")
    check("## Agent Results" in summary, "Agent Results section")
    check("## Memory Used" in summary, "Memory section")
    check("## Sources" in summary, "Sources section")
    check("## Errors" in summary, "Errors section")

    print(f"  Summary length: {len(summary)} chars")


# ── Test 4: evaluate_handoff_quality — strong ───────────────────


def test_eval_strong():
    section("Test 4: evaluate_handoff_quality — strong")

    result = _fake_orchestrator_result()
    # Make all handoffs successful
    for hr in result["handoff_results"]:
        hr["status"] = "completed"
        hr["confidence"] = 0.85
    result["handoff_memory_ids"] = ["mem_001"]

    quality = evaluate_handoff_quality(result)
    check(quality["quality_label"] == "strong",
          f"label=strong (got {quality['quality_label']})")
    check(quality["completed_count"] == 3, f"completed=3")
    check(quality["failed_count"] == 0, "failed=0")
    check(len(quality["warnings"]) == 0,
          f"no warnings (got {quality['warnings']})")

    print(f"  Label: {quality['quality_label']}")
    print(f"  Avg confidence: {quality['avg_confidence']}")


# ── Test 5: evaluate_handoff_quality — medium ───────────────────


def test_eval_medium():
    section("Test 5: evaluate_handoff_quality — medium")

    result = _fake_orchestrator_result()
    # 1 completed (low confidence) + 2 failed → medium
    for hr in result["handoff_results"]:
        hr["status"] = "failed"
        hr["confidence"] = 0.0
    result["handoff_results"][0]["status"] = "completed"
    result["handoff_results"][0]["confidence"] = 0.55
    result["handoff_memory_ids"] = []

    quality = evaluate_handoff_quality(result)
    check(quality["quality_label"] == "medium",
          f"label=medium (got {quality['quality_label']})")
    check(quality["completed_count"] == 1, "completed=1")
    check(quality["failed_count"] == 2, "failed=2")
    check(len(quality["warnings"]) >= 1,
          f"warnings present ({quality['warnings']})")

    print(f"  Label: {quality['quality_label']}")
    print(f"  Warnings: {quality['warnings']}")


# ── Test 6: evaluate_handoff_quality — weak ─────────────────────


def test_eval_weak():
    section("Test 6: evaluate_handoff_quality — weak")

    result = _fake_orchestrator_result()
    # All failed
    for hr in result["handoff_results"]:
        hr["status"] = "failed"
    result["handoff_sources"] = []

    quality = evaluate_handoff_quality(result)
    check(quality["quality_label"] == "weak",
          f"label=weak (got {quality['quality_label']})")
    check(quality["completed_count"] == 0, "completed=0")
    check(quality["failed_count"] == 3, "failed=3")
    check(len(quality["warnings"]) >= 2, "multiple warnings")

    print(f"  Label: {quality['quality_label']}")
    print(f"  Warnings: {quality['warnings']}")


# ── Test 7: trace_and_evaluate convenience ──────────────────────


def test_trace_and_evaluate():
    section("Test 7: trace_and_evaluate() convenience")

    result = _fake_orchestrator_result()
    output = trace_and_evaluate(
        result,
        query="test convenience",
        task_type="paper_question",
        save_trace=True,
    )

    check("trace" in output, "has trace")
    check("quality" in output, "has quality")
    check("saved" in output, "has saved flag")
    check(output["saved"] == True, f"saved=True (got {output['saved']})")
    check(output["quality"]["quality_label"] in ("strong", "medium", "weak"),
          f"quality label valid: {output['quality']['quality_label']}")


# ── Test 8: Orchestrator integration ────────────────────────────


def test_orchestrator_integration():
    section("Test 8: Orchestrator includes trace + trace_quality")

    from research_agent.agents.orchestrator import run_multi_agent_pipeline

    result = run_multi_agent_pipeline(
        query="根据组会进展，找论文支持，并生成汇报大纲",
        task_type="report_generation",
        auto_write_memory=True,
    )

    check("trace" in result, "orchestrator result has trace")
    check("trace_quality" in result, "orchestrator result has trace_quality")

    if result.get("trace"):
        check(len(result["trace"].get("trace_id", "")) > 0, "trace has trace_id")
        print(f"  trace_id: {result['trace'].get('trace_id', '?')[:20]}...")

    if result.get("trace_quality"):
        q = result["trace_quality"]
        print(f"  quality_label: {q.get('quality_label', '?')}")
        print(f"  completed: {q.get('completed_count', 0)}, failed: {q.get('failed_count', 0)}")
        check(q.get("quality_label") is not None, "quality_label present")


# ── Test 9: Trace file is git-ignored ───────────────────────────


def test_gitignore():
    section("Test 9: Trace file is git-ignored")

    # Check that .gitignore contains the traces rule
    gitignore_path = PROJECT_ROOT / ".gitignore"
    gitignore_text = gitignore_path.read_text(encoding="utf-8")

    has_traces_rule = "data/traces/" in gitignore_text
    check(has_traces_rule, "data/traces/ rule in .gitignore")

    # Verify the traces directory and file exist (created by previous tests)
    check(_TRACES_DIR.exists(), "data/traces/ directory exists")
    check(_TRACES_PATH.exists(), "traces file exists")

    print(f"  .gitignore contains 'data/traces/': {has_traces_rule}")
    print(f"  Traces dir: {_TRACES_DIR}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    global PASS, FAIL

    print("=" * 60)
    print("Multi-Agent Trace & Evaluation — Test Suite")
    print("=" * 60)

    test_create_trace()
    test_append_load()
    test_summarize()
    test_eval_strong()
    test_eval_medium()
    test_eval_weak()
    test_trace_and_evaluate()
    test_orchestrator_integration()
    test_gitignore()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    print(f"\n  Traces stored in: {_TRACES_DIR} (git-ignored)")

    if FAIL > 0:
        print("\nSome tests FAILED. Check output for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
