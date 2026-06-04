"""
Test script for Agent Handoff / Communication.

Usage:
    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_agent_handoff.py

Tests 14 areas:
1. create_handoff_request — basic field completeness
2. validate_handoff_request — valid request has no errors
3. from_agent == to_agent → error
4. priority out of range → error
5. handoff_request dict round-trip
6. handoff_result dict round-trip
7. build_handoff_plan — multi-agent query
8. build_handoff_plan — task_type=paper_question
9. build_handoff_plan — task_type=claim_support
10. build_handoff_plan — fallback to general_agent
11. aggregate_handoff_results — merges outputs
12. sources / memory_ids dedup
13. append_handoff_log / load_handoff_log
14. profiles.py compatibility (graceful fallback)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.agents.handoff import (
    HandoffRequest,
    HandoffResult,
    HandoffPlan,
    create_handoff_request,
    validate_handoff_request,
    handoff_request_to_dict,
    handoff_request_from_dict,
    handoff_result_to_dict,
    handoff_result_from_dict,
    build_handoff_plan,
    aggregate_handoff_results,
    append_handoff_log,
    load_handoff_log,
    _PROFILES_AVAILABLE,
    _known_agent_ids,
    _known_handoff_targets,
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


# ── Test 1: create_handoff_request ──────────────────────────────


def test_create_request():
    section("Test 1: create_handoff_request — field completeness")

    req = create_handoff_request(
        from_agent="coordinator_agent",
        to_agent="paper_agent",
        task="Read and summarise a paper",
        input_text="Guardrail-Agnostic Societal Bias Evaluation",
        task_type="paper_question",
        memory_scope=["shared"],
        required_memory_types=["paper_note", "claim_support"],
        expected_output="Structured reading note",
        priority=4,
        reason="User asked about this paper",
        metadata={"paper_path": "data/papers/guardrail_agnostic.md"},
    )

    check(len(req.handoff_id) > 0, "handoff_id is non-empty")
    check(req.from_agent == "coordinator_agent", "from_agent set")
    check(req.to_agent == "paper_agent", "to_agent set")
    check(req.task_type == "paper_question", "task_type set")
    check(req.priority == 4, "priority set")
    check(len(req.created_at) > 0, "created_at auto-filled")
    check("paper_note" in req.required_memory_types, "required_memory_types includes paper_note")
    check("shared" in req.memory_scope, "memory_scope includes shared")


# ── Test 2: validate — valid request ────────────────────────────


def test_validate_valid():
    section("Test 2: validate_handoff_request — valid request, no errors")

    req = create_handoff_request(
        from_agent="coordinator_agent",
        to_agent="paper_agent",
        task="Summarise paper",
        input_text="Paper content here",
    )
    errors = validate_handoff_request(req)
    # Only errors (not warnings) should block
    real_errors = [e for e in errors if not e.startswith("WARNING:")]
    check(len(real_errors) == 0,
          f"no real errors for valid request (got {real_errors})")

    if errors:
        print(f"  INFO Warnings: {errors}")


# ── Test 3: from_agent == to_agent → error ──────────────────────


def test_validate_self_handoff():
    section("Test 3: validate — from_agent == to_agent → error")

    req = create_handoff_request(
        from_agent="paper_agent",
        to_agent="paper_agent",  # same!
        task="Do something",
        input_text="Test",
    )
    errors = validate_handoff_request(req)
    has_error = any(
        "must differ" in e.lower() or "differ" in e.lower()
        for e in errors
    )
    check(has_error, f"self-handoff produces error: {errors}")


# ── Test 4: priority out of range → error ───────────────────────


def test_validate_priority():
    section("Test 4: validate — priority out of range → error")

    req = create_handoff_request(
        from_agent="a", to_agent="b", task="t", input_text="i",
        priority=10,  # invalid
    )
    errors = validate_handoff_request(req)
    has_error = any("priority" in e.lower() for e in errors)
    check(has_error, f"priority=10 produces error: {errors}")


# ── Test 5: handoff_request dict round-trip ─────────────────────


def test_request_roundtrip():
    section("Test 5: handoff_request dict round-trip")

    req = create_handoff_request(
        from_agent="coordinator_agent",
        to_agent="report_agent",
        task="Generate report",
        input_text="Query text",
        metadata={"key": "value"},
    )
    d = handoff_request_to_dict(req)
    restored = handoff_request_from_dict(d)

    check(restored.handoff_id == req.handoff_id, "handoff_id preserved")
    check(restored.from_agent == req.from_agent, "from_agent preserved")
    check(restored.to_agent == req.to_agent, "to_agent preserved")
    check(restored.task == req.task, "task preserved")
    check(restored.metadata.get("key") == "value", "metadata preserved")


# ── Test 6: handoff_result dict round-trip ──────────────────────


def test_result_roundtrip():
    section("Test 6: handoff_result dict round-trip")

    result = HandoffResult(
        handoff_id="ho_test",
        from_agent="a",
        to_agent="b",
        status="completed",
        result_text="Done.",
        confidence=0.85,
        sources=[{"path": "data/test.md"}],
        memory_ids=["mem_001"],
    )
    d = handoff_result_to_dict(result)
    restored = handoff_result_from_dict(d)

    check(restored.handoff_id == "ho_test", "handoff_id preserved")
    check(restored.status == "completed", "status preserved")
    check(restored.confidence == 0.85, "confidence preserved")
    check(len(restored.sources) == 1, "sources preserved")
    check(restored.memory_ids == ["mem_001"], "memory_ids preserved")


# ── Test 7: build_handoff_plan — multi-agent query ──────────────


def test_plan_multi_agent():
    section("Test 7: build_handoff_plan — multi-agent query")

    plan = build_handoff_plan(
        root_query="根据最近组会进展，找论文支持，并生成汇报大纲",
        task_type="general",
    )

    check(len(plan.handoffs) >= 3,
          f"multi-agent plan has >=3 handoffs (got {len(plan.handoffs)})")

    agents = {h.to_agent for h in plan.handoffs}
    check("progress_agent" in agents, "plan includes progress_agent")
    check("paper_agent" in agents, "plan includes paper_agent")
    check("report_agent" in agents, "plan includes report_agent")
    print(f"  Agents in plan: {agents}")


# ── Test 8: build_handoff_plan — paper_question ─────────────────


def test_plan_paper():
    section("Test 8: build_handoff_plan — task_type=paper_question")

    plan = build_handoff_plan(
        root_query="What is guardrail-agnostic evaluation?",
        task_type="paper_question",
    )
    check(len(plan.handoffs) >= 1,
          f"paper_question plan has >=1 handoff ({len(plan.handoffs)})")
    check(plan.handoffs[0].to_agent == "paper_agent",
          f"first handoff to paper_agent: {plan.handoffs[0].to_agent}")
    check("paper_note" in plan.handoffs[0].required_memory_types,
          "required_memory_types includes paper_note")


# ── Test 9: build_handoff_plan — claim_support ──────────────────


def test_plan_claim():
    section("Test 9: build_handoff_plan — task_type=claim_support")

    plan = build_handoff_plan(
        root_query="共现关系是否会导致幻觉？",
        task_type="claim_support",
    )
    check(len(plan.handoffs) >= 1,
          f"claim_support plan has >=1 handoff ({len(plan.handoffs)})")
    agents = {h.to_agent for h in plan.handoffs}
    check("claim_agent" in agents, "plan includes claim_agent")
    # May also include paper_agent, experiment_agent
    print(f"  Agents in plan: {agents}")


# ── Test 10: build_handoff_plan — fallback ──────────────────────


def test_plan_fallback():
    section("Test 10: build_handoff_plan — unknown task → general_agent")

    plan = build_handoff_plan(
        root_query="What is the weather?",
        task_type="unknown_type",
    )
    check(len(plan.handoffs) == 1, f"fallback has 1 handoff (got {len(plan.handoffs)})")
    check(plan.handoffs[0].to_agent == "general_agent",
          f"fallback routed to general_agent: {plan.handoffs[0].to_agent}")


# ── Test 11: aggregate_handoff_results ──────────────────────────


def test_aggregate():
    section("Test 11: aggregate_handoff_results()")

    plan = build_handoff_plan("Test query", "paper_question")

    results = [
        HandoffResult(
            handoff_id=plan.handoffs[0].handoff_id,
            from_agent="coordinator_agent",
            to_agent="paper_agent",
            status="completed",
            result_text="Paper analysis: This paper discusses bias evaluation.",
            confidence=0.9,
            sources=[{"path": "data/papers/test.md"}],
            memory_ids=["mem_001", "mem_002"],
        ),
        HandoffResult(
            handoff_id="ho_fake_2",
            from_agent="coordinator_agent",
            to_agent="experiment_agent",
            status="failed",
            error="No experiment data available",
        ),
    ]

    agg = aggregate_handoff_results(plan, results)

    check(agg["completed"] == 1, f"completed=1 (got {agg['completed']})")
    check(agg["failed"] == 1, f"failed=1 (got {agg['failed']})")
    check("Paper Agent" in agg["combined_answer"], "combined_answer has Paper Agent")
    check("Failed" in agg["combined_answer"], "combined_answer has Failed section")
    check(len(agg["sources"]) >= 1, f"sources non-empty ({len(agg['sources'])})")

    print(f"  Completed: {agg['completed']}, Failed: {agg['failed']}")
    print(f"  Combined answer length: {len(agg['combined_answer'])} chars")


# ── Test 12: sources + memory_ids dedup ─────────────────────────


def test_dedup():
    section("Test 12: sources and memory_ids dedup")

    plan = build_handoff_plan("Test", "paper_question")

    results = [
        HandoffResult(
            handoff_id="ho_1",
            from_agent="c", to_agent="paper_agent", status="completed",
            result_text="Result 1",
            sources=[{"path": "a.md"}, {"path": "b.md"}],
            memory_ids=["mem_001", "mem_002"],
        ),
        HandoffResult(
            handoff_id="ho_2",
            from_agent="c", to_agent="paper_agent", status="completed",
            result_text="Result 2",
            sources=[{"path": "b.md"}, {"path": "c.md"}],  # b.md duplicated
            memory_ids=["mem_002", "mem_003"],               # mem_002 duplicated
        ),
    ]

    agg = aggregate_handoff_results(plan, results)

    check(len(agg["sources"]) == 3, f"3 unique sources (got {len(agg['sources'])})")
    check(len(agg["memory_ids"]) == 3,
          f"3 unique memory_ids (got {len(agg['memory_ids'])})")


# ── Test 13: handoff log ────────────────────────────────────────


def test_handoff_log():
    section("Test 13: append_handoff_log / load_handoff_log")

    # Write a few events
    for i in range(3):
        result = append_handoff_log({
            "event": "handoff_created",
            "from_agent": "coordinator_agent",
            "to_agent": f"agent_{i}",
            "task": f"task_{i}",
        })
        check(result["ok"], f"log append {i} ok")

    # Load
    entries = load_handoff_log(limit=10)
    check(len(entries) >= 3, f"loaded >=3 entries (got {len(entries)})")

    print(f"  Log entries loaded: {len(entries)}")


# ── Test 14: profiles.py compatibility ──────────────────────────


def test_profiles_compat():
    section("Test 14: profiles.py compatibility")

    print(f"  _PROFILES_AVAILABLE: {_PROFILES_AVAILABLE}")

    # Known agent IDs should always be available
    ids = _known_agent_ids()
    check(len(ids) >= 5, f"known agent IDs has >=5 entries ({len(ids)})")
    check("coordinator_agent" in ids, "coordinator_agent in known IDs")
    check("paper_agent" in ids, "paper_agent in known IDs")

    # Handoff targets should always be available
    targets = _known_handoff_targets("coordinator_agent")
    check(len(targets) >= 3, f"coordinator has >=3 handoff targets ({len(targets)})")

    if not _PROFILES_AVAILABLE:
        print(f"  INFO profiles.py not available — using fallback agent list")
        check(True, "fallback agent list works")
    else:
        print(f"  INFO profiles.py is available")


# ── Main ─────────────────────────────────────────────────────────


def main():
    global PASS, FAIL

    print("=" * 60)
    print("Agent Handoff / Communication — Test Suite")
    print("=" * 60)
    print(f"  profiles.py available: {_PROFILES_AVAILABLE}")

    test_create_request()
    test_validate_valid()
    test_validate_self_handoff()
    test_validate_priority()
    test_request_roundtrip()
    test_result_roundtrip()
    test_plan_multi_agent()
    test_plan_paper()
    test_plan_claim()
    test_plan_fallback()
    test_aggregate()
    test_dedup()
    test_handoff_log()
    test_profiles_compat()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    print(f"\n  Note: handoff_log.jsonl written to data/memory/ (git-ignored).")

    if FAIL > 0:
        print("\nSome tests FAILED. Check output for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
