"""
Test suite for Memory Privacy & Scope.

Usage::

    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_memory_privacy.py
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

from research_agent.memory.privacy_scope import (
    classify_scope,
    check_access,
    filter_accessible,
    get_scope,
    set_scope,
    list_shared_with,
    validate_scope_on_write,
    auto_classify_record_scope,
    auto_classify_all,
    apply_scope_update_to_store,
    scope_summary,
)

from research_agent.memory.schema import create_memory_record
from research_agent.memory.store import load_memories, query_memories, append_memory

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
# Helpers — create test records with known scopes
# ═══════════════════════════════════════════════════════════════════════════

def _make(owner, mtype, level="mid_term", scope="private", shared=None, tags=None):
    r = create_memory_record(
        content=f"Test content from {owner}",
        memory_type=mtype,
        memory_level=level,
        owner_agent=owner,
        memory_scope=scope,
        shared_with=shared or [],
        tags=tags or [],
    )
    return r

# ═══════════════════════════════════════════════════════════════════════════
# Test 1 — classify_scope by type
# ═══════════════════════════════════════════════════════════════════════════

def test_classify_by_type():
    section("Test 1: classify_scope by memory_type")
    ok = True
    cases = [
        ("research_direction", "global"),
        ("project_decision", "global"),
        ("claim_support", "shared"),
        ("paper_note", "shared"),
        ("progress_update", "shared"),
        ("experiment_result", "shared"),
        ("report_summary", "shared"),
        ("todo", "private"),
        ("code_note", "private"),
        ("user_preference", "private"),
        ("general_note", "private"),
    ]
    for mtype, expected_scope in cases:
        result = classify_scope(mtype)
        ok &= check(result["memory_scope"] == expected_scope,
                    f"{mtype} → {result['memory_scope']} (expected {expected_scope})")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 2 — classify_scope with tag promotion
# ═══════════════════════════════════════════════════════════════════════════

def test_classify_by_tag():
    section("Test 2: classify_scope by tag promotion")
    ok = True
    # A general_note with "pipeline" tag → promoted to shared
    result = classify_scope("general_note", tags=["pipeline", "RAG"])
    ok &= check(result["memory_scope"] == "shared",
                f"general_note+pipeline tag → {result['memory_scope']} (expected shared)")
    # A general_note with "shared_knowledge" tag → promoted to global
    result = classify_scope("general_note", tags=["shared_knowledge"])
    ok &= check(result["memory_scope"] == "global",
                f"general_note+shared_knowledge tag → {result['memory_scope']} (expected global)")
    # A general_note with no special tags → private
    result = classify_scope("general_note", tags=["random"])
    ok &= check(result["memory_scope"] == "private",
                f"general_note+random tag → {result['memory_scope']} (expected private)")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 3 — private scope: only owner can access
# ═══════════════════════════════════════════════════════════════════════════

def test_private_scope_access():
    section("Test 3: private scope — only owner_agent can access")
    r = _make("paper_agent", "general_note", scope="private")
    ok = True
    ok &= check(check_access(r, "paper_agent"), "owner can access private")
    ok &= check(not check_access(r, "experiment_agent"), "other agent CANNOT access private")
    ok &= check(not check_access(r, "claim_agent"), "another agent CANNOT access private")
    # Coordinator always has access
    ok &= check(check_access(r, "coordinator"), "coordinator can access private")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 4 — shared scope: owner + shared_with can access
# ═══════════════════════════════════════════════════════════════════════════

def test_shared_scope_access():
    section("Test 4: shared scope — owner + shared_with can access")
    r = _make("paper_agent", "paper_note", scope="shared",
              shared=["claim_agent", "report_agent"])
    ok = True
    ok &= check(check_access(r, "paper_agent"), "owner can access shared")
    ok &= check(check_access(r, "claim_agent"), "shared_with agent can access")
    ok &= check(check_access(r, "report_agent"), "another shared_with agent can access")
    ok &= check(not check_access(r, "experiment_agent"), "non-shared agent CANNOT access")
    ok &= check(not check_access(r, "progress_agent"), "non-shared agent CANNOT access")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 5 — global scope: all agents can access
# ═══════════════════════════════════════════════════════════════════════════

def test_global_scope_access():
    section("Test 5: global scope — all agents can access")
    r = _make("coordinator", "research_direction", scope="global")
    ok = True
    for agent in ["paper_agent", "claim_agent", "experiment_agent", "progress_agent",
                  "report_agent", "code_agent", "memory_agent", "coordinator"]:
        ok &= check(check_access(r, agent), f"{agent} can access global")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 6 — filter_accessible
# ═══════════════════════════════════════════════════════════════════════════

def test_filter_accessible():
    section("Test 6: filter_accessible")
    records = [
        _make("paper_agent", "paper_note", scope="private"),
        _make("paper_agent", "paper_note", scope="shared", shared=["claim_agent"]),
        _make("claim_agent", "claim_support", scope="shared", shared=["paper_agent"]),
        _make("coordinator", "research_direction", scope="global"),
        _make("experiment_agent", "experiment_result", scope="private"),
    ]
    # paper_agent can see: own private, own shared, claim_agent's shared (paper_agent in shared_with), global
    filtered = filter_accessible(records, "paper_agent")
    ok = True
    ok &= check(len(filtered) == 4, f"paper_agent sees {len(filtered)} records (expected 4)")
    # experiment_agent can see: own private, global
    filtered2 = filter_accessible(records, "experiment_agent")
    ok &= check(len(filtered2) == 2, f"experiment_agent sees {len(filtered2)} records (expected 2)")
    # Coordinator sees all
    filtered3 = filter_accessible(records, "coordinator")
    ok &= check(len(filtered3) == 5, f"coordinator sees {len(filtered3)} records (expected 5)")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 7 — set_scope / get_scope / list_shared_with
# ═══════════════════════════════════════════════════════════════════════════

def test_scope_getter_setter():
    section("Test 7: set_scope / get_scope / list_shared_with")
    r = _make("paper_agent", "general_note", scope="private")

    # get_scope
    s = get_scope(r)
    ok = True
    ok &= check(s["memory_scope"] == "private", f"get_scope → private (got {s['memory_scope']})")
    ok &= check(s["shared_with"] == [], "shared_with empty for private")
    ok &= check(len(list_shared_with(r)) == 0, "list_shared_with → []")

    # set_scope to shared
    result = set_scope(r, "shared", ["claim_agent", "report_agent"])
    ok &= check(result["ok"], "set_scope shared → ok")
    ok &= check(result["previous_scope"] == "private", "previous_scope = private")
    ok &= check(len(list_shared_with(r)) == 2, f"list_shared_with → 2 agents: {list_shared_with(r)}")

    # set_scope to global
    set_scope(r, "global")
    ok &= check(list_shared_with(r) == ["*"], "list_shared_with for global → ['*']")

    # set_scope invalid
    result = set_scope(r, "ultra_secret")
    ok &= check(not result["ok"], "invalid scope → not ok")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 8 — validate_scope_on_write
# ═══════════════════════════════════════════════════════════════════════════

def test_validate_on_write():
    section("Test 8: validate_scope_on_write")
    ok = True

    # Valid private
    r1 = _make("paper_agent", "general_note", scope="private")
    v1 = validate_scope_on_write(r1)
    ok &= check(v1["valid"], "private scope valid")
    ok &= check(len(v1["changes"]) == 0, "no changes for valid private")

    # Private with stray shared_with → auto-clear
    r2 = _make("paper_agent", "general_note", scope="private", shared=["claim_agent"])
    v2 = validate_scope_on_write(r2)
    ok &= check(v2["valid"], "private+shared_with → still valid (auto-cleared)")
    ok &= check(len(v2["changes"]) > 0, f"changes applied: {v2['changes']}")

    # Global with shared_with → auto-clear
    r3 = _make("coordinator", "research_direction", scope="global", shared=["paper_agent"])
    v3 = validate_scope_on_write(r3)
    ok &= check(v3["valid"], "global+shared_with → still valid (auto-cleared)")

    # Invalid scope → auto-classify
    r4_dict = {"memory_type": "paper_note", "memory_level": "long_term",
               "memory_scope": "ultra_term", "shared_with": [], "owner_agent": "paper_agent",
               "tags": ["bias", "hallucination"]}
    v4 = validate_scope_on_write(r4_dict)
    ok &= check(v4["valid"], "invalid scope auto-classified → valid")
    ok &= check(len(v4["changes"]) > 0, f"auto-classified: {v4['changes']}")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 9 — auto_classify_record_scope
# ═══════════════════════════════════════════════════════════════════════════

def test_auto_classify():
    section("Test 9: auto_classify_record_scope")
    r = _make("claim_agent", "claim_support", scope="private")  # should be shared per rules
    result = auto_classify_record_scope(r)
    ok = True
    ok &= check(result["ok"], "auto-classify ok")
    new_scope = get_scope(r)["memory_scope"]
    ok &= check(new_scope == "shared", f"claim_support auto-classified to shared (got {new_scope})")
    ok &= check("paper_agent" in list_shared_with(r), "shared_with includes paper_agent")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 10 — scope_summary
# ═══════════════════════════════════════════════════════════════════════════

def test_scope_summary():
    section("Test 10: scope_summary")
    records = [
        _make("paper_agent", "general_note", scope="private"),
        _make("paper_agent", "paper_note", scope="shared", shared=["claim_agent"]),
        _make("claim_agent", "claim_support", scope="shared", shared=["paper_agent", "report_agent"]),
        _make("coordinator", "research_direction", scope="global"),
    ]
    summary = scope_summary(records)
    ok = True
    ok &= check(summary["total"] == 4, f"total=4 (got {summary['total']})")
    ok &= check(summary["by_scope"]["private"] == 1, "1 private")
    ok &= check(summary["by_scope"]["shared"] == 2, "2 shared")
    ok &= check(summary["by_scope"]["global"] == 1, "1 global")
    ok &= check(summary["by_owner"]["paper_agent"] == 2, "paper_agent owns 2")
    print(f"  by_scope: {summary['by_scope']}")
    print(f"  by_owner: {summary['by_owner']}")
    print(f"  shared_with_counts: {summary['shared_with_counts']}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 11 — store integration (real write + query)
# ═══════════════════════════════════════════════════════════════════════════

def test_store_integration():
    section("Test 11: apply_scope_update_to_store + query")
    # Create a record, write with private scope, then update to shared
    r = create_memory_record(
        content="Privacy test: store integration — shared experiment result.",
        memory_type="experiment_result",
        memory_level="mid_term",
        owner_agent="experiment_agent",
        memory_scope="private",
        shared_with=[],
        tags=["experiment", "benchmark", "COCO"],
    )
    ok = True

    # Update to shared with proper shared_with
    result = apply_scope_update_to_store(r, "shared", ["claim_agent", "report_agent", "progress_agent"])
    ok &= check(result["ok"], f"apply scope update ok={result['ok']}")
    if result["ok"]:
        updated = result["record"]
        ok &= check(get_scope(updated)["memory_scope"] == "shared",
                    f"scope updated to shared")
        ok &= check(len(list_shared_with(updated)) == 3,
                    f"3 agents in shared_with: {list_shared_with(updated)}")
        # Query should find it
        found = query_memories(tags=["experiment"])
        ok &= check(len(found) >= 1, f"query by tag finds record (found {len(found)})")
    else:
        print(f"  INFO: write error: {result.get('error', 'unknown')}")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 12 — ensure loaded memories from store respect scope
# ═══════════════════════════════════════════════════════════════════════════

def test_store_scope_filtering():
    section("Test 12: filter loaded store memories by agent access")
    all_memories = load_memories()
    if len(all_memories) == 0:
        print("  SKIP: no memories in store yet")
        return True

    ok = True
    # Filter for paper_agent
    paper_visible = filter_accessible(all_memories, "paper_agent")
    total = len(all_memories)
    ok &= check(len(paper_visible) <= total,
                f"paper_agent sees {len(paper_visible)}/{total} records")

    # Every record visible to paper_agent should pass check_access
    for r in paper_visible:
        ok &= check(check_access(r, "paper_agent"),
                    f"  {_rf(r, 'memory_id', '?')[:12]}... accessible to paper_agent ✓")

    print(f"  Total records in store: {total}")
    print(f"  Visible to paper_agent: {len(paper_visible)}")
    print(f"  Visible to coordinator: {len(filter_accessible(all_memories, 'coordinator'))}")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Helper for dataclass field access
# ═══════════════════════════════════════════════════════════════════════════

def _rf(record, key, default=None):
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 60)
    print("  Memory Privacy & Scope — Test Suite")
    print("=" * 60)

    results = {}
    results["classify_by_type"] = test_classify_by_type()
    results["classify_by_tag"] = test_classify_by_tag()
    results["private_access"] = test_private_scope_access()
    results["shared_access"] = test_shared_scope_access()
    results["global_access"] = test_global_scope_access()
    results["filter"] = test_filter_accessible()
    results["getter_setter"] = test_scope_getter_setter()
    results["validate_write"] = test_validate_on_write()
    results["auto_classify"] = test_auto_classify()
    results["scope_summary"] = test_scope_summary()
    results["store_integration"] = test_store_integration()
    results["store_filter"] = test_store_scope_filtering()

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
