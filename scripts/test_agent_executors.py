"""
Test Specialist Agent Executors.

Verifies that all 8 agent executors return valid HandoffResults
without raising exceptions.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.agents.handoff import create_handoff_request, HandoffResult
from research_agent.agents.executors import execute_handoff_request

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


# ── Test each agent ──────────────────────────────────────────────

AGENTS = [
    ("paper_agent", "Summarise the Guardrail-Agnostic paper"),
    ("claim_agent", "Find evidence: co-occurrence induces LVLM hallucination"),
    ("progress_agent", "What is the current research progress?"),
    ("report_agent", "Generate a group meeting report"),
    ("experiment_agent", "Analyse data/experiments/sample_metrics.csv"),
    ("memory_agent", "What memories do I have about hallucination?"),
    ("code_agent", "ModuleNotFoundError: No module named langgraph"),
    ("general_agent", "How should I plan my research today?"),
]


def test_agent_executor(agent_id: str, task: str):
    print(f"\n── {agent_id} ──")

    req = create_handoff_request(
        from_agent="coordinator_agent",
        to_agent=agent_id,
        task=task,
        input_text=task,
    )

    result = execute_handoff_request(req)

    check(result is not None, "result not None")
    check(isinstance(result, HandoffResult) or hasattr(result, "status"),
          f"result is HandoffResult-like (type={type(result).__name__})")
    check(result.status in ("completed", "failed"),
          f"status is completed/failed (got '{result.status}')")
    check(len(result.result_text) > 0 or result.status == "failed",
          "result_text non-empty or status=failed")
    check(0.0 <= result.confidence <= 1.0,
          f"confidence in [0,1] (got {result.confidence})")

    print(f"  status={result.status} confidence={result.confidence:.2f} "
          f"sources={len(result.sources)} mem_ids={len(result.memory_ids)}")
    if result.status == "failed":
        print(f"  error={result.error[:100]}")


def test_unknown_agent_fallback():
    print("\n── unknown agent → general fallback ──")

    req = create_handoff_request(
        from_agent="coordinator_agent",
        to_agent="unknown_xyz_agent",
        task="some task",
        input_text="test",
    )

    result = execute_handoff_request(req)
    check(result.status in ("completed", "failed"),
          f"unknown agent: status={result.status}")
    # Should fall back to general_agent
    print(f"  to_agent={result.to_agent} status={result.status}")


def test_no_exceptions():
    print("\n── No exceptions from any executor ──")

    # Test with None docs/memories
    for agent_id, task in AGENTS:
        req = create_handoff_request(
            from_agent="coordinator_agent",
            to_agent=agent_id,
            task=task,
            input_text=task,
        )
        try:
            result = execute_handoff_request(req, rag_docs=None, memories=None)
            check(result is not None, f"{agent_id}: handles None inputs")
        except Exception as e:
            check(False, f"{agent_id}: raised {type(e).__name__}: {e}")


# ── Main ────────────────────────────────────────────────────────

def main():
    global PASS, FAIL

    print("=" * 60)
    print("  Agent Executors Test Suite")
    print("=" * 60)

    for agent_id, task in AGENTS:
        test_agent_executor(agent_id, task)

    test_unknown_agent_fallback()
    test_no_exceptions()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    if FAIL > 0:
        print("\nSome tests FAILED.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
