"""
Test script for Memory Store v1.

Usage:
    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_memory_store.py

Tests:
1. ensure_memory_store() creates directory and files
2. Create 3 test memory records and append them
3. load_memories() reads all records
4. load_memories_by_level() filters by level
5. load_shared_memories() reads shared records
6. query_memories() with tag filter
7. query_memories() with keyword filter
8. update_memory_summary() generates markdown
9. clear_short_term_memory(confirm=True)
10. archive_memory with confirm=True

Test data is written to data/memory/ and NOT cleaned up so you can
inspect it manually.  Note: data/memory/*.jsonl and *.md are git-ignored.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.memory.store import (
    ensure_memory_store,
    append_memory,
    load_memories,
    load_memories_by_level,
    load_shared_memories,
    query_memories,
    update_memory_summary,
    clear_short_term_memory,
    archive_memory,
    _make_record_dict,
    MEMORY_DIR,
    MEMORY_SUMMARY_PATH,
    SHORT_TERM_PATH,
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


# ── Test 1: ensure_memory_store ─────────────────────────────────


def test_ensure_memory_store():
    section("Test 1: ensure_memory_store()")

    result = ensure_memory_store()
    check(result["ok"], f"ensure_memory_store ok: {result['ok']}")
    check(MEMORY_DIR.exists(), "data/memory/ directory exists")
    check((MEMORY_DIR / ".gitkeep").exists(), ".gitkeep exists")

    # Verify all files exist
    files_to_check = [
        "memory_store.jsonl",
        "long_term_memory.jsonl",
        "mid_term_memory.jsonl",
        "short_term_memory.jsonl",
        "shared_memory.jsonl",
        "memory_summary.md",
    ]
    for fname in files_to_check:
        p = MEMORY_DIR / fname
        check(p.exists(), f"  {fname} exists")

    print(f"  Paths created: {result.get('paths_created')}")


# ── Test 2: Create & append test records ─────────────────────────


def test_append_memories():
    section("Test 2: Create and append 3 test memory records")

    # Record 1: long_term, shared, research_direction
    r1 = _make_record_dict(
        memory_type="research_direction",
        memory_level="long_term",
        memory_scope="shared",
        owner_agent="research_agent",
        content="Project aims to evaluate multimodal bias in LVLMs through "
                "stereotype libraries, co-occurrence pattern analysis, and "
                "guardrail-agnostic evaluation methods.",
        summary="Core research: multimodal bias evaluation via stereotype library "
                "and hallucination screening.",
        source_title="Research Plan 2026",
        source_path="docs/research_plan.md",
        tags=["bias_evaluation", "hallucination", "stereotype_library", "VLM"],
    )

    # Record 2: mid_term, private, todo
    r2 = _make_record_dict(
        memory_type="todo",
        memory_level="mid_term",
        memory_scope="private",
        owner_agent="research_agent",
        content="Complete paper reading pipeline implementation. "
                "Integrate MinerU for PDF parsing. "
                "Build stereotype library with 200 attribute pairs.",
        summary="TODO: complete paper reading pipeline, MinerU integration, "
                "stereotype library expansion",
        source_title="Weekly Plan 2026-06-05",
        source_path="docs/weekly_plan.md",
        tags=["todo", "paper_reading", "mineru"],
    )

    # Record 3: short_term, private, general_note
    r3 = _make_record_dict(
        memory_type="general_note",
        memory_level="short_term",
        memory_scope="private",
        owner_agent="research_agent",
        content="共现关系分析显示 LVLM 对 gender-occupation 刻板印象的放大效应。"
                "coco_val_n300_g1 实验中 mean_extra_object_rate 为 0.23。"
                "需要进一步验证该效应在不同数据集和模型上的鲁棒性。",
        summary="共现关系可能诱发 LVLM 幻觉的初步实验证据",
        source_title="Experiment coco_val_n300_g1",
        source_path="data/experiments/coco_val_n300_g1.md",
        tags=["hallucination", "experiment", "共现关系", "bias"],
    )

    result1 = append_memory(r1)
    result2 = append_memory(r2)
    result3 = append_memory(r3)

    check(result1["ok"], "r1 appended ok")
    check(result2["ok"], "r2 appended ok")
    check(result3["ok"], "r3 appended ok")

    print(f"  r1 memory_id: {result1['memory_id'][:8]}...")
    print(f"  r2 memory_id: {result2['memory_id'][:8]}...")
    print(f"  r3 memory_id: {result3['memory_id'][:8]}...")

    print(f"  r1 paths: {[Path(p).name for p in result1['paths_written']]}")
    print(f"  r2 paths: {[Path(p).name for p in result2['paths_written']]}")
    print(f"  r3 paths: {[Path(p).name for p in result3['paths_written']]}")

    # Verify r1 was written to long_term and shared
    check(
        any("long_term" in p for p in result1["paths_written"]),
        "r1 written to long_term file"
    )
    check(
        any("shared" in p for p in result1["paths_written"]),
        "r1 (shared scope) written to shared file"
    )

    # Verify r2 was NOT written to shared
    check(
        not any("shared" in p for p in result2["paths_written"]),
        "r2 (private) NOT written to shared file"
    )

    return {
        "r1_id": result1["memory_id"],
        "r2_id": result2["memory_id"],
        "r3_id": result3["memory_id"],
    }


# ── Test 3: load_memories ────────────────────────────────────────


def test_load_memories():
    section("Test 3: load_memories()")

    records = load_memories()
    check(len(records) >= 3, f"at least 3 records loaded (got {len(records)})")

    for r in records:
        check("memory_id" in r, "record has memory_id")
        check("memory_type" in r, "record has memory_type")
        check("memory_level" in r, "record has memory_level")
        check("tags" in r, "record has tags")

    print(f"  Loaded {len(records)} records from memory_store.jsonl")


# ── Test 4: load_memories_by_level ───────────────────────────────


def test_load_by_level():
    section("Test 4: load_memories_by_level()")

    long_term = load_memories_by_level("long_term")
    check(len(long_term) >= 1, f"long_term has >=1 record (got {len(long_term)})")
    if long_term:
        check(
            long_term[0].get("memory_level") == "long_term",
            "record level is long_term"
        )
        print(f"  long_term record: {long_term[0].get('summary', '')[:80]}")

    mid_term = load_memories_by_level("mid_term")
    check(len(mid_term) >= 1, f"mid_term has >=1 record (got {len(mid_term)})")

    short_term = load_memories_by_level("short_term")
    check(len(short_term) >= 1, f"short_term has >=1 record (got {len(short_term)})")


# ── Test 5: load_shared_memories ─────────────────────────────────


def test_load_shared():
    section("Test 5: load_shared_memories()")

    shared = load_shared_memories()
    check(len(shared) >= 1, f"shared has >=1 record (got {len(shared)})")
    if shared:
        check(
            shared[0].get("memory_scope") in ("shared", "global"),
            f"shared record has shared/global scope: {shared[0].get('memory_scope')}"
        )
        print(f"  shared record: {shared[0].get('summary', '')[:80]}")


# ── Test 6: query_memories with tags ─────────────────────────────


def test_query_by_tags():
    section("Test 6: query_memories(tags=[...])")

    results = query_memories(tags=["hallucination"])
    check(len(results) >= 1, f"tag 'hallucination' matched >=1 (got {len(results)})")
    if results:
        record_tags = results[0].get("tags", [])
        check(
            "hallucination" in record_tags,
            f"matched record has 'hallucination' tag: {record_tags}"
        )
        print(f"  Found {len(results)} records with 'hallucination' tag")

    # Test multiple tags (OR)
    results2 = query_memories(tags=["todo"])
    check(len(results2) >= 1, f"tag 'todo' matched >=1 (got {len(results2)})")


# ── Test 7: query_memories with keyword ──────────────────────────


def test_query_by_keyword():
    section("Test 7: query_memories(keyword=...)")

    results = query_memories(keyword="共现关系")
    check(len(results) >= 1, f"keyword '共现关系' matched >=1 (got {len(results)})")
    if results:
        print(f"  Found {len(results)} records containing '共现关系'")
        print(f"  Summary: {results[0].get('summary', '')[:80]}")

    results2 = query_memories(keyword="stereotype")
    check(len(results2) >= 1, f"keyword 'stereotype' matched >=1 (got {len(results2)})")


# ── Test 8: update_memory_summary ────────────────────────────────


def test_update_summary():
    section("Test 8: update_memory_summary()")

    summary = update_memory_summary()
    check(len(summary) > 0, "summary is non-empty")
    check(MEMORY_SUMMARY_PATH.exists(), "memory_summary.md exists")

    # Check key sections present
    check("## Long-term Memory" in summary, "Long-term Memory section")
    check("## Mid-term Memory" in summary, "Mid-term Memory section")
    check("## Short-term Memory" in summary, "Short-term Memory section")
    check("## Shared Memory" in summary, "Shared Memory section")
    check("# ResearchAgent Memory Summary" in summary, "title present")

    print(f"  Summary length: {len(summary)} chars")
    print(f"  Written to: {MEMORY_SUMMARY_PATH}")

    # Show a preview
    print(f"\n  --- Summary Preview (first 800 chars) ---")
    print(summary[:800])


# ── Test 9: clear_short_term_memory ──────────────────────────────


def test_clear_short_term():
    section("Test 9: clear_short_term_memory()")

    # Without confirm should fail
    result_no = clear_short_term_memory(confirm=False)
    check(not result_no["ok"], "clear without confirm returns ok=False")

    # With confirm should succeed
    result_yes = clear_short_term_memory(confirm=True)
    check(result_yes["ok"], "clear with confirm=True returns ok=True")

    # Short-term file should be empty
    short_records = load_memories_by_level("short_term")
    check(len(short_records) == 0,
          f"short_term file empty after clear (got {len(short_records)})")

    # But main store should still have all records
    all_records = load_memories()
    check(len(all_records) >= 3,
          f"memory_store.jsonl still has >=3 records (got {len(all_records)})")


# ── Test 10: archive_memory ──────────────────────────────────────
# (uses an ID from test 2)

def test_archive_memory(ids: dict):
    section("Test 10: archive_memory()")

    memory_id = ids.get("r2_id", "")
    if not memory_id:
        check(False, "No memory ID to archive")
        return

    # Without confirm should fail
    result_no = archive_memory(memory_id, confirm=False)
    check(not result_no["ok"], "archive without confirm returns ok=False")

    # With confirm should succeed
    result_yes = archive_memory(memory_id, confirm=True)
    check(result_yes["ok"], f"archive with confirm returns ok=True: {result_yes}")

    # Reload and check status
    all_records = load_memories()
    archived = [r for r in all_records if r.get("memory_id") == memory_id]
    if archived:
        check(
            archived[0].get("status") == "archived",
            f"record status changed to 'archived': {archived[0].get('status')}"
        )
        print(f"  Archived record: [{memory_id[:8]}...] status={archived[0].get('status')}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    global PASS, FAIL

    print("=" * 60)
    print("Memory Store v1 — Test Suite")
    print("=" * 60)
    print(f"  Data directory: {MEMORY_DIR}")
    print(f"  Schema available: {_check_schema()}")

    test_ensure_memory_store()
    ids = test_append_memories()
    test_load_memories()
    test_load_by_level()
    test_load_shared()
    test_query_by_tags()
    test_query_by_keyword()
    test_update_summary()
    test_clear_short_term()
    test_archive_memory(ids)

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    print(f"\n  Note: Test data left in {MEMORY_DIR} for manual inspection.")
    print(f"  data/memory/*.jsonl and *.md are git-ignored.")

    if FAIL > 0:
        print("\nSome tests FAILED. Check output for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


def _check_schema() -> str:
    try:
        from research_agent.memory.schema import MemoryRecord  # noqa: F401
        return "memory.schema imported OK"
    except ImportError:
        return "schema.py not available (using dict fallback)"


if __name__ == "__main__":
    main()
