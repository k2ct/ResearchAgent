"""
Test suite for Agent Conflict Resolution & Arbitration.

Usage::

    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_agent_arbitration.py
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

from research_agent.agents.arbitration import (
    classify_result_stance,
    detect_agent_conflicts,
    arbitrate_results,
    build_coordinator_final_summary,
)
from research_agent.agents.handoff import HandoffResult

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

def _hr(agent: str, text: str, confidence: float = 0.7,
        sources: list = None, memory_ids: list = None) -> HandoffResult:
    return HandoffResult(
        handoff_id=f"ho_{agent}", from_agent="coordinator_agent",
        to_agent=agent, status="completed", result_text=text,
        confidence=confidence, sources=sources or [],
        memory_ids=memory_ids or [],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 1 — stance classification (CN/EN)
# ═══════════════════════════════════════════════════════════════════════════

def test_stance_classification():
    section("Test 1: classify_result_stance — CN + EN keywords")
    ok = True
    ok &= check(classify_result_stance("实验结果表明该方法支持我们的假设") == "support",
                "CN: support detected")
    ok &= check(classify_result_stance("The evidence supports the claim and confirms the hypothesis") == "support",
                "EN: support detected")
    ok &= check(classify_result_stance("证据不足，无法确定结论") == "uncertain",
                "CN: uncertain detected")
    ok &= check(classify_result_stance("Insufficient evidence to draw conclusions") == "uncertain",
                "EN: uncertain detected")
    ok &= check(classify_result_stance("反驳了原有假设") == "oppose",
                "CN: oppose detected")
    ok &= check(classify_result_stance("The results contradict the original claim") == "oppose",
                "EN: oppose detected")
    ok &= check(classify_result_stance("error: execution failed") == "failed",
                "failed detected")
    ok &= check(classify_result_stance("") == "neutral",
                "empty → neutral")
    ok &= check(classify_result_stance("This is a general observation about the topic.") == "neutral",
                "general text → neutral")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 2 — all support → no conflict, position=support
# ═══════════════════════════════════════════════════════════════════════════

def test_all_support():
    section("Test 2: all support → final_position=support, no conflict")
    results = [
        _hr("paper_agent", "The evidence supports the claim. Papers show consistent bias patterns.", 0.85),
        _hr("experiment_agent", "Experiments confirm the hypothesis. Metrics indicate bias.", 0.80),
        _hr("claim_agent", "Claim is supported by theoretical and empirical evidence.", 0.90),
    ]
    conflicts = detect_agent_conflicts(results)
    ok = True
    ok &= check(not conflicts["has_conflict"],
                f"no conflict (types={conflicts['conflict_types']})")

    arb = arbitrate_results(results, root_query="test")
    ok &= check(arb["final_position"] == "support",
                f"final_position={arb['final_position']}")
    ok &= check(arb["confidence"] > 0.6,
                f"confidence={arb['confidence']} (>0.6)")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 3 — support + uncertain → mixed
# ═══════════════════════════════════════════════════════════════════════════

def test_support_plus_uncertain():
    section("Test 3: support + uncertain → mixed")
    results = [
        _hr("paper_agent", "The evidence supports the claim about bias.", 0.85),
        _hr("experiment_agent", "Insufficient evidence — need more experiments.", 0.40),
    ]
    conflicts = detect_agent_conflicts(results)
    ok = True
    ok &= check(conflicts["has_conflict"],
                f"conflict detected: {conflicts['conflict_types']}")
    ok &= check("support_vs_uncertain" in conflicts["conflict_types"],
                "support_vs_uncertain type")

    arb = arbitrate_results(results, root_query="test")
    ok &= check(arb["final_position"] == "mixed",
                f"final_position={arb['final_position']}")
    ok &= check(arb["confidence"] < 0.85,
                f"confidence reduced: {arb['confidence']}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 4 — failed result → conflict
# ═══════════════════════════════════════════════════════════════════════════

def test_failed_result():
    section("Test 4: failed result → conflict")
    results = [
        _hr("paper_agent", "Supports the claim.", 0.8),
        _hr("experiment_agent", "error: execution failed, timeout.", 0.0),
    ]
    conflicts = detect_agent_conflicts(results)
    ok = True
    ok &= check(conflicts["has_conflict"], "conflict detected")
    ok &= check("agent_failed" in conflicts["conflict_types"],
                f"agent_failed type: {conflicts['conflict_types']}")

    arb = arbitrate_results(results)
    ok &= check("Warning" in arb["arbitration_text"],
                "warning in arbitration text")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 5 — confidence gap → conflict
# ═══════════════════════════════════════════════════════════════════════════

def test_confidence_gap():
    section("Test 5: confidence gap > 0.4 → conflict")
    results = [
        _hr("paper_agent", "Strong evidence supports the claim.", 0.95),
        _hr("experiment_agent", "Some evidence but results are noisy.", 0.35),
    ]
    conflicts = detect_agent_conflicts(results)
    ok = True
    ok &= check(conflicts["has_conflict"], "conflict detected")
    ok &= check("confidence_gap" in conflicts["conflict_types"],
                f"confidence_gap type: {conflicts['conflict_types']}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 6 — build_coordinator_final_summary
# ═══════════════════════════════════════════════════════════════════════════

def test_coordinator_summary():
    section("Test 6: build_coordinator_final_summary")
    results = [
        _hr("paper_agent", "Evidence supports the bias-hallucination link.", 0.85,
            sources=[{"path": "p1.md"}, {"path": "p2.md"}]),
        _hr("experiment_agent", "Insufficient evidence from experiments.", 0.40),
        _hr("claim_agent", "Claim is partially supported, needs more data.", 0.60),
    ]
    arb = arbitrate_results(results, root_query="Does bias cause hallucination?")
    summary = build_coordinator_final_summary(arb, results, root_query="Does bias cause hallucination?")

    ok = True
    ok &= check("Coordinator Final Arbitration" in summary, "title present")
    ok &= check("Final Position" in summary, "section 1 present")
    ok &= check("Evidence Agreement" in summary, "section 2 present")
    ok &= check("Conflicts" in summary, "section 3 present")
    ok &= check("Recommended Wording" in summary, "section 4 present")
    ok &= check("Next Actions" in summary, "section 5 present")
    ok &= check(len(summary) > 500, f"summary length={len(summary)} (>500)")

    print(f"\n  --- Summary (first 800 chars) ---")
    print(summary[:800])
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 7 — orchestrator integration
# ═══════════════════════════════════════════════════════════════════════════

def test_orchestrator_integration():
    section("Test 7: orchestrator produces arbitration")
    from research_agent.agents.orchestrator import run_multi_agent_pipeline

    result = run_multi_agent_pipeline(
        query="Does co-occurrence bias cause LVLM hallucination?",
        task_type="claim_support",
        rag_docs=[],
        memory_context="",
        retrieved_memories=[],
        auto_write_memory=False,
    )

    ok = True
    ok &= check("arbitration" in result, "result has 'arbitration' key")
    arb = result.get("arbitration")
    if arb:
        ok &= check("final_position" in arb, "arbitration has final_position")
        ok &= check("conflicts" in arb, "arbitration has conflicts")
        print(f"  final_position: {arb.get('final_position')}")
        print(f"  confidence: {arb.get('confidence')}")
        print(f"  has_conflict: {arb.get('conflicts', {}).get('has_conflict')}")
    else:
        print("  arbitration=None (expected if handoff produced single result)")

    # Verify combined_answer is not broken
    ok &= check(bool(result.get("combined_answer")),
                "combined_answer still present")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 60)
    print("  Agent Arbitration — Test Suite")
    print("=" * 60)

    results = {}
    results["stance"] = test_stance_classification()
    results["all_support"] = test_all_support()
    results["mixed"] = test_support_plus_uncertain()
    results["failed"] = test_failed_result()
    results["confidence_gap"] = test_confidence_gap()
    results["summary"] = test_coordinator_summary()
    results["orchestrator"] = test_orchestrator_integration()

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
