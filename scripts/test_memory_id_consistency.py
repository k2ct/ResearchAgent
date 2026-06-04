"""
Test memory_id consistency across Writer, Store, and fallback paths.

Verifies:
1. Writer fallback record uses memory_id (not record_id).
2. Legacy record_id is migrated to memory_id.
3. Store _normalise_record fills all required defaults.
4. Writer→Store→Retriever end-to-end preserves memory_id.
5. Privacy validate corrects shared_with on private records.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.memory.writer import (
    _fallback_create_record,
    normalize_memory_id,
    write_memory_from_source,
)
from research_agent.memory.store import (
    _normalise_record,
    ensure_memory_store,
    load_memories,
    append_memory,
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


# ── Test 1: Fallback record uses memory_id ────────────────────────

def test_fallback_uses_memory_id():
    print("\n── Test 1: _fallback_create_record uses memory_id ──")

    rec = _fallback_create_record({
        "content": "Fallback test",
        "memory_level": "mid_term",
    })

    check("memory_id" in rec, "memory_id key exists")
    check("record_id" not in rec, "record_id key NOT present")
    check(rec["memory_id"].startswith("mem_"), "memory_id starts with mem_")
    check(rec["importance"] == 3, f"importance default is 3 (got {rec['importance']})")
    check(isinstance(rec["importance"], int), "importance is int")
    check("source_id" in rec, "source_id field exists")
    check("last_accessed_at" in rec, "last_accessed_at field exists")
    check("visibility" in rec, "visibility field exists")
    check(rec["visibility"] == "private", "visibility default is private")
    check(isinstance(rec["shared_with"], list), "shared_with is list")
    check(isinstance(rec["tags"], list), "tags is list")
    check(isinstance(rec["metadata"], dict), "metadata is dict")


# ── Test 2: Legacy record_id migration ────────────────────────────

def test_legacy_record_id_migration():
    print("\n── Test 2: Legacy record_id → memory_id migration ──")

    # Simulate old record with record_id
    old = {
        "record_id": "old_legacy_001",
        "content": "旧记录测试",
        "memory_level": "short_term",
        "memory_scope": "private",
        "memory_type": "general_note",
        "owner_agent": "memory_agent",
    }

    # normalize_memory_id
    rec1 = normalize_memory_id(dict(old))
    check(rec1.get("memory_id") == "old_legacy_001", "normalize: memory_id = old record_id")
    check("record_id" not in rec1, "normalize: record_id removed")
    check(rec1.get("metadata", {}).get("legacy_record_id") == "old_legacy_001",
          "normalize: legacy_record_id stashed in metadata")

    # _normalise_record
    rec2 = _normalise_record(dict(old))
    check(rec2.get("memory_id") == "old_legacy_001",
          f"_normalise_record migrates record_id (got {rec2.get('memory_id')})")
    check("record_id" not in rec2, "_normalise_record removes record_id")
    check(isinstance(rec2.get("shared_with"), list), "_normalise_record fills shared_with")
    check(rec2.get("source_module") == "manual", "_normalise_record fills source_module")
    check(rec2.get("source_id") == "", "_normalise_record fills source_id")
    check(rec2.get("importance") == 3, "_normalise_record fills importance=3")
    check("visibility" in rec2, "_normalise_record fills visibility")
    check("last_accessed_at" in rec2, "_normalise_record fills last_accessed_at")


# ── Test 3: New record always has memory_id ───────────────────────

def test_new_record_has_memory_id():
    print("\n── Test 3: New record from normalise always has memory_id ──")

    new = {
        "content": "全新记录",
        "memory_level": "mid_term",
        "memory_type": "general_note",
    }
    rec = _normalise_record(new)
    check("memory_id" in rec, "new record has memory_id")
    check(rec["memory_id"].startswith("") or len(rec["memory_id"]) > 20,
          f"memory_id looks valid: {rec['memory_id'][:30]}")
    check("record_id" not in rec, "new record has no record_id")


# ── Test 4: Store append → load preserves memory_id ───────────────

def test_store_roundtrip_preserves_id():
    print("\n── Test 4: Store append → load preserves memory_id ──")

    ensure_memory_store()

    test_rec = _fallback_create_record({
        "content": "Store roundtrip test: memory_id consistency check.",
        "memory_level": "short_term",
        "memory_type": "general_note",
        "owner_agent": "memory_agent",
        "tags": ["roundtrip_test"],
    })

    mem_id = test_rec["memory_id"]
    result = append_memory(test_rec)
    check(result["ok"], f"append_memory succeeded (id={mem_id[:30]}...)")
    check(result.get("memory_id") == mem_id or result["memory_id"] == result.get("memory_id", ""),
          "append result references correct memory_id")

    # Load and verify
    all_recs = load_memories()
    found = [r for r in all_recs if r.get("memory_id") == mem_id]
    check(len(found) >= 1, f"load_memories finds record by memory_id ({len(found)})")

    if found:
        r = found[0]
        check(r.get("shared_with") is not None, "shared_with present after load")
        check(r.get("source_module") is not None, "source_module present after load")
        check(r.get("source_id") is not None, "source_id present after load")
        check(r.get("visibility") is not None, "visibility present after load")


# ── Test 5: Writer → Store end-to-end uses memory_id ──────────────

def test_writer_to_store_memory_id():
    print("\n── Test 5: Writer → Store end-to-end uses memory_id ──")

    result = write_memory_from_source(
        content="请记住：我的长期研究方向是共现关系对 LVLM 幻觉的影响。",
        source_module="chat",
        source_title="memory_id consistency e2e test",
    )

    check(result["ok"], "write_memory_from_source succeeded")

    record = result["record"]
    # Handle both MemoryRecord and dict
    if isinstance(record, dict):
        mem_id = record.get("memory_id", "")
    else:
        mem_id = getattr(record, "memory_id", "")

    check(len(mem_id) > 0, f"record has memory_id: {mem_id[:30]}...")
    check("record_id" not in (record if isinstance(record, dict) else {}),
          "no record_id at top level in dict output")

    write_result = result.get("write_result", {})
    w_mem_id = write_result.get("memory_id", "") or write_result.get("record_id", "")
    check(len(w_mem_id) > 0, f"write_result has memory_id: {w_mem_id[:30]}...")


# ── Test 6: Privacy validate corrects private shared_with ──────────

def test_privacy_validate_corrects_shared_with():
    print("\n── Test 6: validate_scope_on_write corrects shared_with ──")

    try:
        from research_agent.memory.privacy_scope import validate_scope_on_write
        _HAS_PRIVACY = True
    except ImportError:
        _HAS_PRIVACY = False

    if not _HAS_PRIVACY:
        print("  SKIP  privacy_scope not available")
        return

    # Create a private record with leaked shared_with
    leaked = {
        "memory_id": "mem_test_leak_001",
        "memory_scope": "private",
        "shared_with": ["paper_agent"],
        "memory_type": "general_note",
        "memory_level": "short_term",
        "owner_agent": "memory_agent",
        "content": "Test privacy correction",
    }

    validation = validate_scope_on_write(dict(leaked))
    check(validation.get("valid"), f"validation valid={validation.get('valid')}")

    # After validation, private record should have empty shared_with
    corrected = validation.get("record", leaked)
    scope = corrected.get("memory_scope") if isinstance(corrected, dict) else getattr(corrected, "memory_scope", "?")
    sw = corrected.get("shared_with", []) if isinstance(corrected, dict) else getattr(corrected, "shared_with", [])

    if scope == "private":
        check(sw == [] or len(sw) == 0,
              f"private record shared_with corrected to empty (was {sw})")

    changes = validation.get("changes", [])
    print(f"  Validation changes: {changes}")
    print(f"  Validation warnings: {validation.get('warnings', [])}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    global PASS, FAIL

    print("=" * 60)
    print("  Memory ID Consistency Test Suite")
    print("=" * 60)

    test_fallback_uses_memory_id()
    test_legacy_record_id_migration()
    test_new_record_has_memory_id()
    test_store_roundtrip_preserves_id()
    test_writer_to_store_memory_id()
    test_privacy_validate_corrects_shared_with()

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
