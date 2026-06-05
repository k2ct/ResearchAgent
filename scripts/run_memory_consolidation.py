"""
Memory Consolidation CLI — safe production runner.

**Default: dry-run only.**  No files are modified unless you pass ``--apply``.

Usage::

    # Preview what would change (safe — no writes)
    python scripts/run_memory_consolidation.py

    # Actually apply changes (backs up first)
    python scripts/run_memory_consolidation.py --apply

    # Apply with custom settings
    python scripts/run_memory_consolidation.py --apply --weekly --merge-threshold 0.5

    # Apply without backup (NOT recommended)
    python scripts/run_memory_consolidation.py --apply --no-backup
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.memory.store import ensure_memory_store, MEMORY_DIR
from research_agent.memory.consolidation import (
    preview_consolidation_plan,
    run_consolidation_safely,
    STAGE_WEEKLY_DAYS,
    STAGE_MONTHLY_DAYS,
)


def main():
    parser = argparse.ArgumentParser(
        description="Memory Consolidation — safe CLI (default: dry-run only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_memory_consolidation.py                # dry-run preview
  python scripts/run_memory_consolidation.py --apply        # apply with backup
  python scripts/run_memory_consolidation.py --apply --weekly --expire
        """,
    )

    # ── Safety flags ──
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually write changes to memory JSONL files. "
             "Without this flag, only a dry-run preview is shown.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        default=False,
        help="Skip automatic backup before applying. NOT recommended.",
    )

    # ── Summary mode ──
    summary_group = parser.add_mutually_exclusive_group()
    summary_group.add_argument(
        "--weekly",
        action="store_const",
        const=STAGE_WEEKLY_DAYS,
        dest="stage_days",
        default=STAGE_WEEKLY_DAYS,
        help="Generate a weekly summary (7-day window, default).",
    )
    summary_group.add_argument(
        "--monthly",
        action="store_const",
        const=STAGE_MONTHLY_DAYS,
        dest="stage_days",
        help="Generate a monthly summary (30-day window).",
    )

    # ── Tuning ──
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.60,
        help="Jaccard similarity threshold for duplicate merge (0.0–1.0, default: 0.60).",
    )
    parser.add_argument(
        "--compress-threshold",
        type=int,
        default=3000,
        help="Content length (chars) above which memories are compressed (default: 3000).",
    )
    parser.add_argument(
        "--expire",
        action="store_true",
        default=False,
        help="Enable expiry marking for stale memories. "
             "Without this flag, expiry is skipped even in --apply mode.",
    )

    args = parser.parse_args()

    # ── Ensure store exists ──
    ensure_memory_store()

    # ── Stage label ──
    stage_label = ""
    if args.stage_days == STAGE_MONTHLY_DAYS:
        stage_label = "Monthly"
    # else: weekly (default), label stays "" for auto-detection

    # ═══════════════════════════════════════════════════════════════
    # DRY-RUN MODE (default)
    # ═══════════════════════════════════════════════════════════════
    if not args.apply:
        print("=" * 60)
        print("  Memory Consolidation — DRY RUN")
        print("=" * 60)
        print()
        print("  This is a dry run. No files were modified.")
        print("  To apply changes, run with: --apply")
        print()

        preview = preview_consolidation_plan(
            merge_similarity=args.merge_threshold,
            compress_threshold=args.compress_threshold,
            short_term_expiry_days=7 if args.expire else 9999,
            mid_term_expiry_days=60 if args.expire else 9999,
            stage_window_days=args.stage_days,
            stage_label=stage_label,
        )

        if not preview["ok"]:
            print(f"  Error: {preview['error']}")
            sys.exit(1)

        print(f"  Duplicates to merge:      {preview['duplicates_to_merge']}")
        print(f"  Long memories to compress: {preview['long_memories_to_compress']}")
        print(f"  Memories to expire:       {preview['memories_to_expire']}")
        print(f"  Stage summary records:    {preview['stage_summary_records']}")
        print()

        if preview["estimated_changes"]:
            print("  --- Change Details ---")
            for change in preview["estimated_changes"]:
                op = change["operation"]
                count = change["count"]
                print(f"  [{op}] x{count}")
                details = change.get("details", [])
                if isinstance(details, list):
                    for d in details[:5]:
                        if isinstance(d, dict):
                            if "kept" in d:
                                print(f"    merge: keep={d['kept'][:12]}... "
                                      f"away={d.get('merged_away', '')[:12]}... "
                                      f"sim={d.get('similarity', '?')}")
                            elif "memory_id" in d:
                                print(f"    {d['memory_id'][:12]}... "
                                      f"level={d.get('level', '?')} "
                                      f"days={d.get('days_since_update', '?')}")
                            elif "original_len" in d:
                                print(f"    {d['memory_id'][:12]}... "
                                      f"{d['original_len']} → {d['new_len']} chars")
                elif isinstance(details, dict):
                    print(f"    label={details.get('label', '?')} "
                          f"window={details.get('window_days', '?')}d")
        else:
            print("  No changes needed — memory store is clean.")

        print()
        print("=" * 60)
        print("  Dry run complete. Use --apply to execute these changes.")
        print("=" * 60)
        sys.exit(0)

    # ═══════════════════════════════════════════════════════════════
    # APPLY MODE
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  Memory Consolidation — APPLY MODE")
    print("=" * 60)
    print()

    do_backup = not args.no_backup
    if do_backup:
        print("  Backup: enabled (use --no-backup to skip)")
    else:
        print("  Backup: DISABLED (--no-backup specified)")

    # Set expiry thresholds
    if args.expire:
        st_exp = 7
        mt_exp = 60
    else:
        st_exp = 9999
        mt_exp = 9999
        print("  Expiry: skipped (use --expire to enable)")

    result = run_consolidation_safely(
        merge_similarity=args.merge_threshold,
        compress_threshold=args.compress_threshold,
        short_term_expiry_days=st_exp,
        mid_term_expiry_days=mt_exp,
        stage_window_days=args.stage_days,
        stage_label=stage_label,
        apply=True,
        backup=do_backup,
    )

    # ── Backup result ──
    if result.get("backup_result"):
        bk = result["backup_result"]
        if bk["ok"]:
            print(f"  Backup created: {bk['backup_dir']}")
            print(f"  Files backed up: {', '.join(bk['files_copied'])}")
        else:
            print(f"  Backup FAILED: {bk['error']}")
            print("  Aborting.")
            sys.exit(1)
    print()

    # ── Preview ──
    preview = result.get("preview", {})
    print(f"  Duplicates merged:    {preview.get('duplicates_to_merge', '?')}")
    print(f"  Memories compressed:  {preview.get('long_memories_to_compress', '?')}")
    print(f"  Memories expired:     {preview.get('memories_to_expire', '?')}")
    print(f"  Stage records covered:{preview.get('stage_summary_records', '?')}")
    print()

    # ── Consolidation result ──
    cons = result.get("consolidation_result", {})
    if cons.get("ok"):
        print("  Consolidation completed successfully.")
    else:
        print("  Consolidation completed with errors.")
        # Check sub-results for details
        for key in ("merge_result", "compress_result", "expire_result", "stage_summary_result"):
            sub = cons.get(key, {})
            if sub and not sub.get("ok"):
                print(f"    {key}: error={sub.get('error', 'unknown')}")

    print()
    print("=" * 60)
    print(f"  Mode: apply")
    print(f"  Store directory: {MEMORY_DIR}")
    print("  Data NOT committed to git (git-ignored).")
    print("=" * 60)


if __name__ == "__main__":
    main()
