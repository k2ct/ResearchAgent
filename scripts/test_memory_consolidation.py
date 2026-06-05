"""
Test script for Memory Consolidation v1.

Usage:
    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_memory_consolidation.py

Tests:
1. Seed test memories (duplicates, long, old)
2. merge_duplicates — detect and merge near-duplicates
3. compress_long_memories — compress overly long records
4. mark_expired_memories — expire stale records
5. generate_stage_summary — weekly and monthly digests
6. run_consolidation — run all operations in sequence
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.memory.store import (
    ensure_memory_store,
    append_memory,
    load_memories,
    load_memories_by_level,
    update_memory_summary,
    clear_short_term_memory,
    _rewrite_jsonl,
    _make_record_dict,
    MEMORY_STORE_PATH,
    MEMORY_DIR,
)
from research_agent.memory.consolidation import (
    merge_duplicates,
    compress_long_memories,
    mark_expired_memories,
    generate_stage_summary,
    run_consolidation,
    _content_similarity,
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


# ── Helper: seed test data ───────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _days_ago_iso(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def seed_test_memories():
    """Clear store and seed with test records for consolidation."""
    # Clear existing
    _rewrite_jsonl(MEMORY_STORE_PATH, [])

    # 1a + 1b: near-duplicate pair (same topic, slightly different wording)
    r1a = _make_record_dict(
        memory_type="experiment_result",
        memory_level="mid_term",
        memory_scope="private",
        owner_agent="experiment_agent",
        content="coco_val_n300_g1 experiment shows mean_extra_object_rate of 0.23. "
                "The hallucination screening pipeline identified 20 high-risk images "
                "using the hrs_v1 composite score. Gender-swapped SD3.5 images showed "
                "higher hallucination in male→female direction.",
        summary="COCO hallucination screening: n300/g1 results",
        source_title="Experiment coco_val_n300_g1",
        tags=["hallucination", "experiment", "COCO"],
    )
    r1a["created_at"] = _days_ago_iso(2)

    r1b = _make_record_dict(
        memory_type="experiment_result",
        memory_level="mid_term",
        memory_scope="private",
        owner_agent="experiment_agent",
        content="coco_val_n300_g1 experiment shows mean extra object rate 0.23. "
                "The screening pipeline found 20 high-risk images using hrs_v1. "
                "Gender swap (male→female) had more extra objects than the reverse.",
        summary="COCO hallucination screening n300 g1 findings",
        source_title="Experiment coco_val_n300_g1",
        tags=["hallucination", "experiment", "COCO", "gender_swap"],
    )
    r1b["created_at"] = _days_ago_iso(1)

    # 2: Long content (should be compressed)
    long_content = (
        "This is a very long experiment report. " * 80 +
        "It contains detailed methodology descriptions. " * 80 +
        "And extensive analysis of results. " * 80 +
        "With many figures and tables referenced. " * 80
    )
    r2 = _make_record_dict(
        memory_type="experiment_result",
        memory_level="mid_term",
        content=long_content,
        summary="",
        tags=["long", "experiment"],
    )
    r2["created_at"] = _days_ago_iso(5)

    # 3: Old short-term memory (should expire)
    r3 = _make_record_dict(
        memory_type="general_note",
        memory_level="short_term",
        content="Temporary note about port conflict on port 8503.",
        summary="Port conflict note",
        tags=["debug"],
    )
    r3["created_at"] = _days_ago_iso(10)
    r3["updated_at"] = _days_ago_iso(10)

    # 4: Old mid-term memory (should expire)
    r4 = _make_record_dict(
        memory_type="todo",
        memory_level="mid_term",
        content="TODO: Set up CI pipeline for automated testing.",
        summary="CI pipeline setup",
        tags=["todo"],
    )
    r4["created_at"] = _days_ago_iso(90)
    r4["updated_at"] = _days_ago_iso(90)

    # 5: Active long-term memory (should NOT expire)
    r5 = _make_record_dict(
        memory_type="research_direction",
        memory_level="long_term",
        memory_scope="shared",
        content="Core research direction: multimodal bias evaluation in LVLMs. "
                "Build stereotype library with intersectional attributes. "
                "Integrate MinerU for PDF paper ingestion.",
        summary="Core research direction",
        tags=["research_direction", "bias", "stereotype_library"],
    )
    r5["created_at"] = _days_ago_iso(120)

    # 6: Progress update for weekly summary
    r6 = _make_record_dict(
        memory_type="progress_update",
        memory_level="mid_term",
        content="Completed paper reading pipeline implementation. "
                "MinerU integration for PDF parsing is working. "
                "Stereotype library now has 100 attribute pairs.",
        summary="Weekly progress: paper pipeline + MinerU + 100 stereotypes",
        tags=["progress", "paper_reading", "mineru"],
    )
    r6["created_at"] = _days_ago_iso(1)

    # 7: Another recent memory for summary
    r7 = _make_record_dict(
        memory_type="claim_support",
        memory_level="mid_term",
        memory_scope="shared",
        content="Found evidence supporting the claim that co-occurrence patterns "
                "in LVLM training data may induce object hallucination. "
                "Multiple papers (guardrail_agnostic, VIGNETTE) provide "
                "methodological support.",
        summary="Evidence found for co-occurrence → hallucination claim",
        tags=["claim_support", "hallucination", "共现关系"],
    )
    r7["created_at"] = _days_ago_iso(2)

    # Write all
    for r in [r1a, r1b, r2, r3, r4, r5, r6, r7]:
        append_memory(r, write_level_file=True, update_summary=False)

    print(f"  Seeded 8 test memories into memory_store.jsonl")


# ── Test 1: Text similarity ──────────────────────────────────────


def test_similarity():
    section("Test 1: _content_similarity()")

    sim_high = _content_similarity(
        "coco_val_n300_g1 experiment shows mean extra object rate 0.23.",
        "coco_val_n300_g1 experiment shows mean_extra_object_rate of 0.23.",
    )
    check(sim_high > 0.5, f"similar content → high similarity: {sim_high:.3f}")

    sim_low = _content_similarity(
        "coco_val_n300_g1 experiment shows mean extra object rate 0.23.",
        "Set up CI pipeline for automated testing with GitHub Actions.",
    )
    check(sim_low < 0.3, f"different content → low similarity: {sim_low:.3f}")

    sim_identical = _content_similarity("identical text", "identical text")
    check(sim_identical > 0.9, f"identical → very high: {sim_identical:.3f}")


# ── Test 2: Merge duplicates ─────────────────────────────────────


def test_merge_duplicates():
    section("Test 2: merge_duplicates()")

    # Dry run first
    dry = merge_duplicates(similarity_threshold=0.50, dry_run=True)
    check(dry["ok"], "dry run ok")
    print(f"  Dry run: {dry['merged_count']} merges detected")
    for m in dry.get("merges", []):
        print(f"    keep={m['kept'][:8]}... merge_away={m['merged_away'][:8]}... sim={m['similarity']}")

    check(dry["merged_count"] >= 1, "at least 1 duplicate detected")

    # Real merge
    result = merge_duplicates(similarity_threshold=0.50, dry_run=False)
    check(result["ok"], "merge ok")
    check(result["merged_count"] >= 1, f"merged >=1 records (merged {result['merged_count']})")

    # Verify: merged-away record should have status "merged"
    all_records = load_memories()
    merged_records = [r for r in all_records if r.get("status") == "merged"]
    check(len(merged_records) >= 1, f"found >=1 merged record ({len(merged_records)})")

    # Verify: kept record should have merged_from in metadata
    kept = [r for r in all_records if r.get("status") == "active"
            and "merged_from" in r.get("metadata", {})]
    if kept:
        check(len(kept) >= 1, f"found >=1 record with merged_from metadata ({len(kept)})")
        merged_tags = kept[0].get("tags", [])
        check("gender_swap" in merged_tags,
              f"kept record has merged tags (gender_swap): {merged_tags}")


# ── Test 3: Compress long memories ───────────────────────────────


def test_compress():
    section("Test 3: compress_long_memories()")

    dry = compress_long_memories(threshold=3000, target_length=2000, dry_run=True)
    check(dry["ok"], "dry run ok")
    print(f"  Dry run: {dry['compressed_count']} records to compress")

    result = compress_long_memories(threshold=3000, target_length=2000, dry_run=False)
    check(result["ok"], "compress ok")
    check(result["compressed_count"] >= 1, f"compressed >=1 records ({result['compressed_count']})")

    for c in result.get("compressed", []):
        print(f"    {c['memory_id'][:8]}... {c['original_len']} → {c['new_len']} chars")

    # Verify: compressed records should be shorter
    all_records = load_memories()
    for c in result.get("compressed", []):
        rec = next((r for r in all_records if r.get("memory_id") == c["memory_id"]), None)
        if rec:
            actual_len = len(rec.get("content", ""))
            check(actual_len <= c["original_len"],
                  f"compressed content <= original ({actual_len} <= {c['original_len']})")
            # Should have metadata tracking original length
            has_orig = rec.get("metadata", {}).get("original_content_length")
            check(has_orig is not None, "original_content_length recorded in metadata")


# ── Test 4: Mark expired ──────────────────────────────────────────


def test_expire():
    section("Test 4: mark_expired_memories()")

    dry = mark_expired_memories(
        short_term_days=7, mid_term_days=60,
        expire_long_term=False, dry_run=True,
    )
    check(dry["ok"], "dry run ok")
    print(f"  Dry run: {dry['expired_count']} records to expire")
    for e in dry.get("expired", []):
        print(f"    {e['memory_id'][:8]}... level={e['level']} days={e['days_since_update']}")

    check(dry["expired_count"] >= 2, "at least 2 records to expire (short 10d + mid 90d)")

    result = mark_expired_memories(
        short_term_days=7, mid_term_days=60,
        expire_long_term=False, dry_run=False,
    )
    check(result["ok"], "expire ok")
    check(result["expired_count"] >= 2, f"expired >=2 records ({result['expired_count']})")

    # Verify: short-term old record is expired
    all_records = load_memories()
    expired = [r for r in all_records if r.get("status") == "expired"]
    check(len(expired) >= 2, f"found >=2 expired records ({len(expired)})")

    # Verify: long-term record is NOT expired
    long_records = [r for r in all_records
                    if r.get("memory_level") == "long_term"
                    and r.get("status") == "active"]
    check(len(long_records) >= 1,
          f"long-term active records remain: {len(long_records)}")


# ── Test 5: Stage summary (weekly) ────────────────────────────────


def test_stage_summary_weekly():
    section("Test 5: generate_stage_summary (weekly, 7 days)")

    result = generate_stage_summary(window_days=7, label="Weekly", write_to_store=True)
    check(result["ok"], f"weekly summary ok: {result['ok']}")
    check(result["records_covered"] >= 1,
          f"covers >=1 records ({result['records_covered']})")
    check(len(result["summary_text"]) > 0, "summary_text is non-empty")

    # Should have key sections
    text = result["summary_text"]
    check("# " in text, "summary has title")
    check("Window:" in text, "summary has window info")
    check("Records covered:" in text, "summary has record count")

    print(f"  Records covered: {result['records_covered']}")
    print(f"  Written memory_id: {result['written_memory_id'][:20]}...")
    print(f"  Summary length: {len(text)} chars")
    print(f"\n  --- Preview (first 800 chars) ---")
    print(text[:800])


# ── Test 6: Stage summary (monthly) ───────────────────────────────


def test_stage_summary_monthly():
    section("Test 6: generate_stage_summary (monthly, 30 days)")

    result = generate_stage_summary(window_days=30, label="Monthly", write_to_store=True)
    check(result["ok"], f"monthly summary ok: {result['ok']}")
    check(result["records_covered"] >= 1,
          f"covers >=1 records ({result['records_covered']})")

    print(f"  Records covered: {result['records_covered']}")
    print(f"  Written memory_id: {result['written_memory_id'][:20]}...")
    print(f"  Summary length: {len(result['summary_text'])} chars")


# ── Test 7: run_consolidation (all-in-one) ────────────────────────


def test_run_consolidation():
    section("Test 7: run_consolidation() — all operations")

    # Re-seed with fresh test data
    seed_test_memories()

    result = run_consolidation(
        merge_similarity=0.50,
        compress_threshold=3000,
        compress_target=2000,
        short_term_expiry_days=7,
        mid_term_expiry_days=60,
        stage_window_days=30,
        stage_label="Monthly",
        dry_run=False,
    )

    # Check all sub-results
    merge_ok = result["merge_result"]["ok"]
    compress_ok = result["compress_result"]["ok"]
    expire_ok = result["expire_result"]["ok"]
    stage_ok = result["stage_summary_result"]["ok"]

    check(merge_ok, f"merge sub-result ok: {merge_ok}")
    check(compress_ok, f"compress sub-result ok: {compress_ok}")
    check(expire_ok, f"expire sub-result ok: {expire_ok}")
    check(stage_ok, f"stage sub-result ok: {stage_ok}")
    check(result["ok"], f"overall ok: {result['ok']}")

    print(f"  Merge:     {result['merge_result']['merged_count']} merged")
    print(f"  Compress:  {result['compress_result']['compressed_count']} compressed")
    print(f"  Expire:    {result['expire_result']['expired_count']} expired")
    print(f"  Stage:     {result['stage_summary_result']['records_covered']} records covered")

    # Verify memory_summary.md was updated
    summary_path = MEMORY_DIR / "memory_summary.md"
    check(summary_path.exists(), "memory_summary.md exists after consolidation")


# ── Test 8: Dry run does not modify ───────────────────────────────


def test_dry_run_safety():
    section("Test 8: dry_run=True does not modify store")

    # seed fresh
    seed_test_memories()
    records_before = load_memories()
    count_before = len(records_before)

    for r in records_before:
        r["status"] = "active"  # reset any prior changes

    _rewrite_jsonl(MEMORY_STORE_PATH, records_before)

    run_consolidation(
        merge_similarity=0.50,
        compress_threshold=3000,
        short_term_expiry_days=7,
        mid_term_expiry_days=60,
        stage_window_days=7,
        dry_run=True,
    )

    records_after = load_memories()
    count_after = len(records_after)

    check(count_before == count_after,
          f"record count unchanged: {count_before} → {count_after}")

    # All statuses should be "active"
    for r in records_after:
        if r.get("status") != "active":
            check(False, f"record status changed in dry_run: {r.get('status')}")
            break
    else:
        check(True, "all record statuses still active after dry_run")


# ── Test 9: backup_memory_store ─────────────────────────────────


def test_backup():
    section("Test 9: backup_memory_store()")

    # Import the backup function
    from research_agent.memory.consolidation import backup_memory_store

    result = backup_memory_store()
    check(result["ok"], f"backup ok: {result['ok']}")
    check(len(result["files_copied"]) >= 1,
          f"at least 1 file copied (got {len(result['files_copied'])})")
    check("backup_dir" in result, "backup_dir in result")

    backup_path = Path(result["backup_dir"])
    check(backup_path.exists(), f"backup dir exists: {backup_path}")

    # Verify copied files
    for fname in result["files_copied"]:
        copied = backup_path / fname
        check(copied.exists(), f"  {fname} exists in backup")

    print(f"  Backup dir: {backup_path}")
    print(f"  Files: {result['files_copied']}")


# ── Test 10: preview_consolidation_plan ──────────────────────────


def test_preview_plan():
    section("Test 10: preview_consolidation_plan()")

    from research_agent.memory.consolidation import preview_consolidation_plan

    # Count records before
    records_before = load_memories()
    count_before = len(records_before)

    plan = preview_consolidation_plan(
        merge_similarity=0.50,
        compress_threshold=3000,
        short_term_expiry_days=7,
        mid_term_expiry_days=60,
        stage_window_days=30,
    )

    check(plan["ok"], f"preview ok: {plan['ok']}")
    check(plan["mode"] == "preview", f"mode=preview: {plan['mode']}")
    check("duplicates_to_merge" in plan, "has duplicates_to_merge")
    check("long_memories_to_compress" in plan, "has long_memories_to_compress")
    check("memories_to_expire" in plan, "has memories_to_expire")
    check("estimated_changes" in plan, "has estimated_changes")

    # Verify NO files were modified
    records_after = load_memories()
    count_after = len(records_after)
    check(count_before == count_after,
          f"record count unchanged after preview: {count_before} → {count_after}")

    print(f"  Duplicates to merge: {plan['duplicates_to_merge']}")
    print(f"  Long mems to compress: {plan['long_memories_to_compress']}")
    print(f"  Memories to expire: {plan['memories_to_expire']}")
    print(f"  Stage records: {plan['stage_summary_records']}")


# ── Test 11: run_consolidation_safely(apply=False) ───────────────


def test_safe_dry_run():
    section("Test 11: run_consolidation_safely(apply=False)")

    from research_agent.memory.consolidation import run_consolidation_safely

    records_before = load_memories()
    count_before = len(records_before)

    result = run_consolidation_safely(
        merge_similarity=0.50,
        compress_threshold=3000,
        short_term_expiry_days=7,
        mid_term_expiry_days=60,
        stage_window_days=7,
        apply=False,  # explicitly dry-run
    )

    check(result["ok"], f"ok: {result['ok']}")
    check(result["mode"] == "dry_run", f"mode=dry_run: {result['mode']}")
    check(result["consolidation_result"] is None,
          "consolidation_result is None (not applied)")
    check("preview" in result, "has preview")

    # Verify NO files modified
    records_after = load_memories()
    check(len(records_before) == len(records_after),
          f"record count unchanged ({count_before})")

    print(f"  Mode: {result['mode']}")


# ── Test 12: run_consolidation_safely(apply=True) ────────────────


def test_safe_apply():
    section("Test 12: run_consolidation_safely(apply=True, backup=True)")

    from research_agent.memory.consolidation import run_consolidation_safely

    seed_test_memories()

    result = run_consolidation_safely(
        merge_similarity=0.50,
        compress_threshold=3000,
        short_term_expiry_days=7,
        mid_term_expiry_days=60,
        stage_window_days=7,
        stage_label="Weekly",
        apply=True,
        backup=True,
    )

    check(result["ok"], f"ok: {result['ok']}")
    check(result["mode"] == "apply", f"mode=apply: {result['mode']}")

    # Backup must have run
    bk = result.get("backup_result")
    check(bk is not None, "backup_result is not None")
    if bk:
        check(bk["ok"], f"backup ok: {bk['ok']}")
        check(len(bk["files_copied"]) >= 1,
              f"backup copied >=1 files ({len(bk['files_copied'])})")
        print(f"  Backup: {bk['backup_dir']}")
        print(f"  Files: {bk['files_copied']}")

    # Consolidation must have run
    cons = result.get("consolidation_result")
    check(cons is not None, "consolidation_result is not None")
    if cons:
        check(cons.get("dry_run") == False,
              "consolidation dry_run=False")

    print(f"  Merge count: {cons.get('merge_result', {}).get('merged_count', '?')}")
    print(f"  Compress count: {cons.get('compress_result', {}).get('compressed_count', '?')}")
    print(f"  Expire count: {cons.get('expire_result', {}).get('expired_count', '?')}")


# ── Test 13: CLI dry-run (via subprocess) ────────────────────────


def test_cli_dry_run():
    section("Test 13: CLI — default dry-run via subprocess")

    import subprocess

    python_exe = PROJECT_ROOT / ".conda" / "python.exe"
    script = PROJECT_ROOT / "scripts" / "run_memory_consolidation.py"

    result = subprocess.run(
        [str(python_exe), str(script), "--expire", "--weekly"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=30,
    )

    output = result.stdout + result.stderr

    check(result.returncode == 0, f"CLI exits 0 (got {result.returncode})")
    check("This is a dry run" in output or "DRY RUN" in output,
          "output contains dry-run notice")
    check("No files were modified" in output or "No files" in output,
          "output confirms no files modified")
    check("--apply" in output,
          "output mentions --apply flag")

    print(f"  CLI return code: {result.returncode}")
    # Show a few key lines
    for line in output.splitlines():
        if "dry run" in line.lower() or "apply" in line.lower() or "Duplicates" in line:
            print(f"    {line.strip()}")


# ── Cleanup ───────────────────────────────────────────────────────


def cleanup():
    """Reset short_term file (cleaned by expire tests)."""
    from research_agent.memory.store import SHORT_TERM_PATH
    SHORT_TERM_PATH.write_text("", encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────


def main():
    global PASS, FAIL

    print("=" * 60)
    print("Memory Consolidation v1 — Test Suite")
    print("=" * 60)

    ensure_memory_store()
    seed_test_memories()

    test_similarity()
    test_merge_duplicates()
    seed_test_memories()  # re-seed so compress has fresh data
    test_compress()
    test_expire()
    test_stage_summary_weekly()
    test_stage_summary_monthly()
    test_run_consolidation()
    test_dry_run_safety()
    test_backup()
    test_preview_plan()
    test_safe_dry_run()
    test_safe_apply()
    test_cli_dry_run()

    cleanup()
    update_memory_summary()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    print(f"\n  Note: Test data left in {MEMORY_DIR} for manual inspection.")

    if FAIL > 0:
        print("\nSome tests FAILED. Check output for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
