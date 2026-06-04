"""
Memory Core Integration Test — end-to-end pipeline.

Tests the full chain: Writer → Store → Retriever → Privacy → Consolidation.

Writes real records to data/memory/*.jsonl (git-ignored).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.memory.store import (
    ensure_memory_store,
    load_memories as store_load_memories,
    load_memories_by_level,
    load_shared_memories,
    query_memories,
    update_memory_summary,
)
from research_agent.memory.writer import write_memory_from_source
from research_agent.memory.retriever import (
    load_memories_for_retrieval,
    retrieve_memories,
    get_retriever_backend_status,
)
from research_agent.memory.privacy_scope import (
    check_access,
    filter_accessible,
    validate_scope_on_write,
    scope_summary,
)
from research_agent.memory.consolidation import (
    run_consolidation_safely,
    preview_consolidation_plan,
    backup_memory_store,
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


# ── Test 1: Write through Writer → verify in Store ────────────────

def test_write_to_store():
    print("\n── Test 1: Writer → Store write path ──")

    ensure_memory_store()

    content = "请记住：我的长期研究方向是共现关系对 LVLM 幻觉的影响。"

    result = write_memory_from_source(
        content=content,
        source_module="chat",
        source_title="Research Direction Declaration",
    )

    check(result["ok"], f"write_memory_from_source succeeded")
    record = result["record"]

    # Handle both MemoryRecord dataclass and plain dict
    def _get(rec, key, default=""):
        if isinstance(rec, dict):
            return rec.get(key, default)
        return getattr(rec, key, default)

    mem_id = _get(record, "memory_id") or _get(record, "record_id", "")
    check(len(mem_id) > 0, f"record has ID: {mem_id[:30]}...")

    mem_level = _get(record, "memory_level", "")
    check(mem_level in ("long_term", "mid_term", "short_term"),
          f"memory_level is valid: {mem_level}")

    mem_scope = _get(record, "memory_scope", "")
    check(mem_scope in ("private", "shared", "global"),
          f"memory_scope is valid: {mem_scope}")

    # Verify in store
    all_records = store_load_memories()
    found = [r for r in all_records
             if r.get("memory_id") == mem_id or r.get("record_id") == mem_id]
    check(len(found) >= 1, f"record found in memory_store.jsonl ({len(found)})")

    return mem_id, mem_level, mem_scope, record


# ── Test 2: Verify level files ────────────────────────────────────

def test_level_files(mem_id: str, mem_level: str):
    print("\n── Test 2: Level-specific files ──")

    if mem_level in ("long_term", "mid_term", "short_term"):
        level_records = load_memories_by_level(mem_level)
        found = [r for r in level_records
                 if r.get("memory_id") == mem_id or r.get("record_id") == mem_id]
        check(len(found) >= 1,
              f"record found in {mem_level}_memory.jsonl ({len(found)})")
    else:
        check(True, f"skipped — unknown level: {mem_level}")


# ── Test 3: Shared memory file ────────────────────────────────────

def test_shared_file(mem_id: str, mem_scope: str):
    print("\n── Test 3: Shared memory file (if applicable) ──")

    if mem_scope in ("shared", "global"):
        shared = load_shared_memories()
        found = [r for r in shared
                 if r.get("memory_id") == mem_id or r.get("record_id") == mem_id]
        check(len(found) >= 1,
              f"record found in shared_memory.jsonl ({len(found)})")
    else:
        print(f"  INFO  scope={mem_scope} — not expected in shared_memory.jsonl")


# ── Test 4: Retriever finds the record ────────────────────────────

def test_retriever_finds(mem_id: str):
    print("\n── Test 4: Retriever finds the record ──")

    # Via store
    records = load_memories_for_retrieval()
    found = [r for r in records
             if r.get("memory_id") == mem_id or r.get("record_id") == mem_id]
    check(len(found) >= 1,
          f"retriever.load_memories_for_retrieval finds record ({len(found)})")

    # Retrieve by keyword
    results = retrieve_memories(records, keyword="共现关系")
    check(len(results) >= 1,
          f"retrieve_memories(keyword='共现关系') finds {len(results)} records")

    # Retrieve by memory_type
    mtype = found[0].get("memory_type", "") if found else ""
    if mtype:
        results2 = retrieve_memories(records, memory_type=mtype)
        check(len(results2) >= 1,
              f"retrieve by memory_type='{mtype}' finds {len(results2)} records")

    # Backend status
    status = get_retriever_backend_status()
    print(f"  Backend: uses_store={status['uses_store']}, "
          f"record_count={status['record_count']}")


# ── Test 5: Privacy access control ────────────────────────────────

def test_privacy_access(mem_id: str, record: dict):
    print("\n── Test 5: Privacy access control ──")

    # Get fresh record from store
    all_records = store_load_memories()
    target = next((r for r in all_records
                   if r.get("memory_id") == mem_id or r.get("record_id") == mem_id), None)
    check(target is not None, "target record loaded for privacy check")

    if target is None:
        return

    owner = target.get("owner_agent", "")
    scope = target.get("memory_scope", "private")

    # Owner can always access
    if owner:
        check(check_access(target, owner),
              f"owner '{owner}' can access own record")

    # Coordinator can always access
    check(check_access(target, "coordinator"),
          "coordinator can access (universal agent)")

    # Non-owner non-shared agent blocked for private
    if scope == "private" and owner and owner != "report_agent":
        check(not check_access(target, "report_agent"),
              f"unrelated agent 'report_agent' blocked for private record")

    # filter_accessible
    filtered = filter_accessible([target], owner if owner else "coordinator")
    check(len(filtered) == 1, "filter_accessible returns record for owner")

    # validate_scope_on_write
    validation = validate_scope_on_write(target)
    check(validation["valid"], f"validate_scope_on_write: valid={validation['valid']}")

    # scope_summary
    summary = scope_summary(all_records)
    check(summary["total"] >= 1, f"scope_summary covers {summary['total']} records")
    check("private" in summary["by_scope"], "scope_summary has private count")


# ── Test 6: Consolidation dry-run safety ──────────────────────────

def test_consolidation_dry_run():
    print("\n── Test 6: Consolidation dry-run safety ──")

    # Preview (dry run by default)
    preview = preview_consolidation_plan()
    check(preview["ok"], f"preview_consolidation_plan ok")
    check(preview["mode"] == "preview", f"preview mode: {preview['mode']}")

    # run_consolidation_safely (apply=False)
    safe = run_consolidation_safely(apply=False)
    check(safe["mode"] == "dry_run",
          f"run_consolidation_safely mode: {safe['mode']}")
    check(safe["consolidation_result"] is None,
          "apply=False → consolidation_result is None (no writes)")

    # Backup
    bk = backup_memory_store()
    check(bk["ok"], f"backup_memory_store succeeded")
    if bk["ok"]:
        print(f"  Backup created: {bk['backup_dir']}")
        print(f"  Files copied: {bk['files_copied']}")


# ── Test 7: Schema field presence check ───────────────────────────

def test_schema_field_presence(mem_id: str):
    print("\n── Test 7: Schema field presence ──")

    all_records = store_load_memories()
    target = next((r for r in all_records
                   if r.get("memory_id") == mem_id or r.get("record_id") == mem_id), None)

    if target is None:
        check(False, "target record not found for schema check")
        return

    # Required schema fields
    required_fields = [
        "memory_id", "memory_level", "memory_scope", "memory_type",
        "owner_agent", "content", "summary", "tags", "created_at", "updated_at",
    ]
    # Optional but expected
    expected_fields = [
        "shared_with", "source_module", "source_path", "source_title",
        "importance", "status", "metadata",
    ]
    # Schema fields that may be missing in fallback path
    optional_fields = [
        "source_id", "last_accessed_at", "visibility",
    ]

    for field in required_fields:
        # Fallback uses record_id instead of memory_id
        if field == "memory_id":
            has = bool(target.get("memory_id") or target.get("record_id"))
        else:
            has = field in target and target[field] not in (None, "")
        check(has, f"field '{field}' present")

    for field in expected_fields:
        has = field in target
        if not has:
            print(f"  WARN  optional field '{field}' missing from record")
        # Don't fail — optional

    for field in optional_fields:
        has = field in target
        if not has:
            print(f"  NOTE  schema field '{field}' not in record (fallback may omit)")


# ── Main ──────────────────────────────────────────────────────────

def main():
    global PASS, FAIL

    print("=" * 60)
    print("  Memory Core Integration Test")
    print("  (Writer → Store → Retriever → Privacy → Consolidation)")
    print("=" * 60)

    mem_id, mem_level, mem_scope, record = test_write_to_store()
    test_level_files(mem_id, mem_level)
    test_shared_file(mem_id, mem_scope)
    test_retriever_finds(mem_id)
    test_privacy_access(mem_id, record)
    test_consolidation_dry_run()
    test_schema_field_presence(mem_id)

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    if FAIL == 0:
        print("\n  Memory Core Integration PASS")
    else:
        print("\n  Memory Core Integration: SOME CHECKS FAILED")

    if FAIL > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
