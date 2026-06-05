"""
Integration test: Memory Store ↔ Memory Retriever.

Verifies that:
1. ``retriever.load_memories_for_retrieval()`` uses store.load_memories().
2. get_retriever_backend_status() reports uses_store=True when store is available.
3. Records written via store.append_memory() are visible to retriever.
4. All retriever filter dimensions work on store-backed data.
5. The fallback path functions when the store is unavailable (simulated).
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.memory.store import (
    ensure_memory_store,
    append_memory,
    load_memories as store_load_memories,
)
from research_agent.memory.retriever import (
    load_memories_for_retrieval,
    retrieve_memories,
    retrieve_from_store,
    get_retriever_backend_status,
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


# ── Test 1: Backend status after initialisation ────────────────────

def test_store_is_primary_backend():
    print("\n── Test 1: store is primary backend ──")

    # Ensure store is initialised
    ensure_memory_store()

    # Load via retriever — should hit the store
    records = load_memories_for_retrieval()
    check(isinstance(records, list), "load_memories_for_retrieval returns list")

    status = get_retriever_backend_status()
    print(f"  Backend status: uses_store={status['uses_store']}, "
          f"record_count={status['record_count']}")

    check(status["uses_store"] is True,
          f"uses_store is True (got {status['uses_store']})")
    check(status["fallback_reason"] == "",
          f"fallback_reason is empty (got '{status['fallback_reason'][:40]}')")


# ── Test 2: Write via store, read via retriever ──────────────────

def test_write_store_read_retriever():
    print("\n── Test 2: write via store.append_memory, read via retriever ──")

    _now = datetime.now(timezone.utc)

    test_record = {
        "memory_id": f"mem_integration_{_now.strftime('%Y%m%d_%H%M%S')}",
        "memory_type": "experiment_result",
        "memory_level": "mid_term",
        "memory_scope": "private",
        "owner_agent": "experiment_agent",
        "source_module": "experiment_tool",
        "content": "Integration test: coco_val_n300_g1 rerun completed, hrs_v1=0.34.",
        "summary": "Integration test record",
        "source_title": "COCO Validation n300 g1 (rerun)",
        "source_path": "data/experiments/coco_val_n300_g1.md",
        "tags": ["integration_test", "COCO", "hrs_v1", "rerun"],
        "status": "active",
        "importance": 4,
        "visibility": "private",
        "metadata": {"test": True, "run_tag": "coco_val_n300_g1"},
        "created_at": _now.isoformat(),
        "updated_at": _now.isoformat(),
    }

    # Write
    result = append_memory(test_record, write_level_file=True, update_summary=True)
    check(result["ok"], f"append_memory succeeds (id={result['memory_id'][:30]}...)")
    memory_id = result["memory_id"]

    # Read back via store directly
    all_records = store_load_memories()
    found = [r for r in all_records if r.get("memory_id") == memory_id]
    check(len(found) == 1, f"store.load_memories finds the record")

    # Read back via retriever
    records = load_memories_for_retrieval()
    found2 = [r for r in records if r.get("memory_id") == memory_id]
    check(len(found2) == 1, f"retriever.load_memories_for_retrieval finds the record")

    # Verify fields
    if found2:
        r = found2[0]
        check(r["memory_type"] == "experiment_result", "memory_type correct")
        check(r["memory_level"] == "mid_term", "memory_level correct")
        check("coco_val_n300_g1" in r["content"], "content preserved")
        check("integration_test" in r["tags"], "tags preserved")
        check(r["importance"] == 4, "importance preserved")
        check(r["metadata"]["run_tag"] == "coco_val_n300_g1", "metadata preserved")


# ── Test 3: Filter dimensions on store-backed data ────────────────

def test_filters_on_store_data():
    print("\n── Test 3: filter dimensions on store-backed data ──")

    records = load_memories_for_retrieval()
    check(len(records) > 0, f"store has records ({len(records)})")

    # memory_type
    r1 = retrieve_memories(records, memory_type="experiment_result")
    check(len(r1) >= 1, f"memory_type filter works ({len(r1)} results)")

    # tags
    r2 = retrieve_memories(records, tags=["integration_test"])
    check(len(r2) >= 1, f"tags filter works ({len(r2)} results)")

    # keyword
    r3 = retrieve_memories(records, keyword="coco_val_n300_g1")
    check(len(r3) >= 1, f"keyword filter works ({len(r3)} results)")

    # Combined
    r4 = retrieve_memories(
        records,
        memory_type="experiment_result",
        tags=["integration_test"],
        keyword="hrs_v1",
    )
    check(len(r4) >= 1, f"combined filters work ({len(r4)} results)")

    # owner_agent
    r5 = retrieve_memories(records, owner_agent="experiment_agent")
    check(len(r5) >= 1, f"owner_agent filter works ({len(r5)} results)")


# ── Test 4: retrieve_from_store convenience ───────────────────────

def test_retrieve_from_store():
    print("\n── Test 4: retrieve_from_store convenience ──")

    results = retrieve_from_store(
        memory_type="experiment_result",
        tags=["integration_test"],
        limit=5,
    )
    check(len(results) >= 1,
          f"retrieve_from_store finds integration test records ({len(results)})")

    for r in results:
        check(r["memory_type"] == "experiment_result",
              f"  {r['memory_id'][:30]} memory_type correct")


# ── Test 5: Fallback to direct JSONL ──────────────────────────────

def test_fallback_direct_jsonl():
    print("\n── Test 5: fallback to direct JSONL (when explicit path given) ──")

    # Write a minimal test record to a temp JSONL
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    fallback_record = {
        "memory_id": "mem_fallback_test_001",
        "memory_type": "general_note",
        "memory_level": "short_term",
        "memory_scope": "private",
        "owner_agent": "general_agent",
        "content": "This is a fallback test record.",
        "summary": "Fallback test",
        "source_title": "",
        "source_path": "",
        "tags": ["fallback"],
        "status": "active",
        "importance": 2,
        "created_at": "2026-06-05T00:00:00+00:00",
        "updated_at": "2026-06-05T00:00:00+00:00",
        "metadata": {},
    }
    tmp.write(json.dumps(fallback_record, ensure_ascii=False) + "\n")
    tmp.close()

    try:
        # Load directly (bypasses store — explicit path)
        from research_agent.memory.retriever import load_memories
        records = load_memories(tmp.name)
        check(len(records) == 1, "direct load_memories reads fallback file")
        check(records[0]["memory_id"] == "mem_fallback_test_001",
              "fallback record id correct")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


# ── Main ──────────────────────────────────────────────────────────

def main():
    global PASS, FAIL

    print("=" * 60)
    print("  Memory Store <-> Retriever Integration Test")
    print("=" * 60)

    test_store_is_primary_backend()
    test_write_store_read_retriever()
    test_filters_on_store_data()
    test_retrieve_from_store()
    test_fallback_direct_jsonl()

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
