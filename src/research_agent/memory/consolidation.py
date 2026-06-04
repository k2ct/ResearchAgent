"""
Memory Consolidation v1 — merge, compress, expire, and summarise.

Long-term memory maintenance operations that run on the JSONL store.

Operations:
1. **merge_duplicates** — find near-duplicate records and merge them.
2. **compress_long_memories** — truncate or summarise overly long content.
3. **mark_expired_memories** — set status=expired for stale records.
4. **generate_stage_summary** — create a weekly or monthly research digest.
5. **run_consolidation** — run all operations in sequence.

Design:
- Reads from ``store.py`` (load_memories, query_memories).
- Writes back via ``store.py`` internal helpers.
- Does NOT depend on schema.py or writer.py directly.
- Preserves original JSONL data; modifications are idempotent.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .store import (
    _read_jsonl,
    _rewrite_jsonl,
    _append_jsonl,
    load_memories,
    query_memories,
    update_memory_summary,
    MEMORY_STORE_PATH,
    MEMORY_DIR,
    _make_record_dict,
)


# ── Constants ────────────────────────────────────────────────────────

# Content length threshold for compression (characters)
DEFAULT_COMPRESS_THRESHOLD = 3000

# Default summary length for compressed records
DEFAULT_COMPRESS_TARGET_LENGTH = 2000

# Similarity threshold for merge (0.0–1.0)
DEFAULT_MERGE_SIMILARITY_THRESHOLD = 0.60

# Expiry thresholds (days since last update)
SHORT_TERM_EXPIRY_DAYS = 7
MID_TERM_EXPIRY_DAYS = 60
LONG_TERM_NEVER_EXPIRES = True

# Stage summary windows
STAGE_WEEKLY_DAYS = 7
STAGE_MONTHLY_DAYS = 30


# ── 1. Text similarity ───────────────────────────────────────────────


def _token_set(text: str) -> Set[str]:
    """Tokenise text into a set of normalised words."""
    # Simple tokenisation: split on whitespace + punctuation, lowercase
    tokens = re.findall(r'[a-zA-Z0-9一-鿿]+', text.lower())
    return set(tokens)


def _content_similarity(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard similarity between two texts based on token sets.

    Returns a float in [0.0, 1.0].
    """
    set_a = _token_set(text_a)
    set_b = _token_set(text_b)

    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# ── 2. Merge duplicates ──────────────────────────────────────────────


def merge_duplicates(
    similarity_threshold: float = DEFAULT_MERGE_SIMILARITY_THRESHOLD,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Find and merge near-duplicate memory records.

    Two records are considered duplicates when:
    - They have the same ``memory_type`` AND ``memory_level``.
    - Their content Jaccard similarity >= *similarity_threshold*.

    Merging strategy:
    - Keep the **older** record (earlier ``created_at``).
    - Append the younger record's ``memory_id`` to the older's metadata
      ``merged_from`` list.
    - Merge ``tags`` (union).
    - Merge ``source_path`` (append to metadata).
    - Update ``summary`` to the shorter of the two.
    - Set the younger record's ``status`` to ``"merged"``.
    - Update ``updated_at`` on the older record.

    Args:
        similarity_threshold: Jaccard threshold for merge (0.0–1.0).
        dry_run: If True, only report what would be merged without writing.

    Returns::

        {
            "ok": bool,
            "merged_count": int,
            "merges": [{"kept": str, "merged_away": str, "similarity": float}, ...],
            "dry_run": bool,
            "error": str,
        }
    """
    try:
        records = load_memories()
        active = [r for r in records if r.get("status") not in ("merged", "archived", "expired")]

        merges: List[Dict[str, Any]] = []
        merged_away_ids: Set[str] = set()

        for i in range(len(active)):
            if active[i]["memory_id"] in merged_away_ids:
                continue
            for j in range(i + 1, len(active)):
                if active[j]["memory_id"] in merged_away_ids:
                    continue

                r_a = active[i]
                r_b = active[j]

                # Require same type and level
                if r_a.get("memory_type") != r_b.get("memory_type"):
                    continue
                if r_a.get("memory_level") != r_b.get("memory_level"):
                    continue

                sim = _content_similarity(
                    r_a.get("content", ""),
                    r_b.get("content", ""),
                )

                if sim >= similarity_threshold:
                    # Keep older, merge away younger
                    created_a = r_a.get("created_at", "")
                    created_b = r_b.get("created_at", "")

                    if created_a <= created_b:
                        older, younger = r_a, r_b
                    else:
                        older, younger = r_b, r_a

                    merges.append({
                        "kept": older["memory_id"],
                        "merged_away": younger["memory_id"],
                        "similarity": round(sim, 3),
                    })
                    merged_away_ids.add(younger["memory_id"])

                    if not dry_run:
                        _apply_merge(older, younger)

        if not dry_run and merges:
            _rewrite_jsonl(MEMORY_STORE_PATH, records)
            update_memory_summary()

        return {
            "ok": True,
            "merged_count": len(merges),
            "merges": merges,
            "dry_run": dry_run,
            "error": "",
        }

    except Exception as e:
        return {
            "ok": False,
            "merged_count": 0,
            "merges": [],
            "dry_run": dry_run,
            "error": str(e),
        }


def _apply_merge(older: Dict[str, Any], younger: Dict[str, Any]) -> None:
    """
    Merge *younger* into *older* in-place.

    Does NOT write to disk — caller must rewrite the store.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Track merged IDs
    merged_from = older.setdefault("metadata", {}).get("merged_from", [])
    if isinstance(merged_from, list):
        merged_from.append(younger["memory_id"])
    else:
        merged_from = [younger["memory_id"]]
    older.setdefault("metadata", {})["merged_from"] = merged_from

    # Merge tags (union)
    older_tags = set(older.get("tags", []))
    younger_tags = set(younger.get("tags", []))
    older["tags"] = sorted(older_tags | younger_tags)

    # Merge source paths
    older_sources = older.setdefault("metadata", {}).get("merged_sources", [])
    if not older_sources:
        older_src = older.get("source_path", "")
        if older_src:
            older_sources = [older_src]
    younger_src = younger.get("source_path", "")
    if younger_src and younger_src not in older_sources:
        older_sources.append(younger_src)
    older.setdefault("metadata", {})["merged_sources"] = older_sources

    # Keep shorter summary
    older_summary = older.get("summary", "")
    younger_summary = younger.get("summary", "")
    if younger_summary and (not older_summary or len(younger_summary) < len(older_summary)):
        older["summary"] = younger_summary

    # Take the longer content
    if len(younger.get("content", "")) > len(older.get("content", "")):
        older["content"] = younger["content"]

    older["updated_at"] = now
    younger["status"] = "merged"
    younger["updated_at"] = now


# ── 3. Compress long memories ────────────────────────────────────────


def compress_long_memories(
    threshold: int = DEFAULT_COMPRESS_THRESHOLD,
    target_length: int = DEFAULT_COMPRESS_TARGET_LENGTH,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Compress memory records whose ``content`` exceeds *threshold* characters.

    Strategy:
    - If ``summary`` is empty or shorter than *target_length*, generate a
      summary from the beginning of ``content``.
    - Truncate ``content`` to *target_length* and append a note referencing
      the original length.
    - Preserve the original content length in metadata.

    Args:
        threshold: Content length that triggers compression.
        target_length: Target length after compression.
        dry_run: If True, only report without writing.

    Returns::

        {
            "ok": bool,
            "compressed_count": int,
            "compressed": [{"memory_id": str, "original_len": int, "new_len": int}, ...],
            "dry_run": bool,
            "error": str,
        }
    """
    try:
        records = load_memories()
        compressed: List[Dict[str, Any]] = []

        for r in records:
            content = r.get("content", "")
            if len(content) <= threshold:
                continue
            if r.get("status") in ("merged", "archived", "expired"):
                continue

            original_len = len(content)
            summary = r.get("summary", "")

            if not dry_run:
                # Generate summary if needed
                if not summary or len(summary) < 100:
                    summary = content[:target_length].strip()
                    # Clean up: remove mid-word breaks
                    last_period = max(
                        summary.rfind("。"),
                        summary.rfind(". "),
                        summary.rfind("\n\n"),
                    )
                    if last_period > target_length // 2:
                        summary = summary[:last_period + 1]
                    r["summary"] = summary

                # Truncate content
                truncated = content[:target_length].strip()
                last_period = max(
                    truncated.rfind("。"),
                    truncated.rfind(". "),
                    truncated.rfind("\n\n"),
                )
                if last_period > target_length // 2:
                    truncated = truncated[:last_period + 1]

                r["content"] = truncated + (
                    f"\n\n[Content compressed: original {original_len} chars → "
                    f"{len(truncated)} chars. Full content available in backup.]"
                )

                # Record original length
                r.setdefault("metadata", {})["original_content_length"] = original_len
                r["updated_at"] = datetime.now(timezone.utc).isoformat()

            compressed.append({
                "memory_id": r.get("memory_id", "?"),
                "original_len": original_len,
                "new_len": len(r["content"]),
            })

        if not dry_run and compressed:
            _rewrite_jsonl(MEMORY_STORE_PATH, records)
            update_memory_summary()

        return {
            "ok": True,
            "compressed_count": len(compressed),
            "compressed": compressed,
            "dry_run": dry_run,
            "error": "",
        }

    except Exception as e:
        return {
            "ok": False,
            "compressed_count": 0,
            "compressed": [],
            "dry_run": dry_run,
            "error": str(e),
        }


# ── 4. Mark expired memories ─────────────────────────────────────────


def _parse_iso_date(date_str: str) -> Optional[datetime]:
    """Parse an ISO-8601 date string, returning None on failure."""
    if not date_str:
        return None
    try:
        # Handle Z suffix
        s = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def mark_expired_memories(
    short_term_days: int = SHORT_TERM_EXPIRY_DAYS,
    mid_term_days: int = MID_TERM_EXPIRY_DAYS,
    expire_long_term: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Mark stale memory records as ``expired``.

    Expiry rules:
    - **short_term**: expired after *short_term_days* days without update.
    - **mid_term**: expired after *mid_term_days* days without update.
    - **long_term**: never expires by default (set ``expire_long_term=True``
      to enable, using *mid_term_days*).

    Only affects records with ``status == "active"``.

    Args:
        short_term_days: Days before short-term is expired.
        mid_term_days: Days before mid-term is expired.
        expire_long_term: Whether long-term memories can expire.
        dry_run: If True, only report without writing.

    Returns::

        {
            "ok": bool,
            "expired_count": int,
            "expired": [{"memory_id": str, "level": str, "days_since_update": int}, ...],
            "dry_run": bool,
            "error": str,
        }
    """
    try:
        records = load_memories()
        now = datetime.now(timezone.utc)
        expired: List[Dict[str, Any]] = []

        thresholds = {
            "short_term": short_term_days,
            "mid_term": mid_term_days,
            "long_term": mid_term_days if expire_long_term else float("inf"),
        }

        for r in records:
            if r.get("status") != "active":
                continue

            level = r.get("memory_level", "short_term")
            threshold_days = thresholds.get(level, short_term_days)

            if threshold_days == float("inf"):
                continue

            # Use updated_at if available, else created_at
            date_str = r.get("updated_at") or r.get("created_at", "")
            last_date = _parse_iso_date(date_str)

            if last_date is None:
                continue

            days_since = (now - last_date).days

            if days_since >= threshold_days:
                days_info = {
                    "memory_id": r.get("memory_id", "?"),
                    "level": level,
                    "days_since_update": days_since,
                }
                expired.append(days_info)

                if not dry_run:
                    r["status"] = "expired"
                    r["updated_at"] = now.isoformat()
                    r.setdefault("metadata", {})["expired_reason"] = (
                        f"No update for {days_since} days "
                        f"(threshold: {threshold_days} days for {level})"
                    )

        if not dry_run and expired:
            _rewrite_jsonl(MEMORY_STORE_PATH, records)
            update_memory_summary()

        return {
            "ok": True,
            "expired_count": len(expired),
            "expired": expired,
            "dry_run": dry_run,
            "error": "",
        }

    except Exception as e:
        return {
            "ok": False,
            "expired_count": 0,
            "expired": [],
            "dry_run": dry_run,
            "error": str(e),
        }


# ── 5. Stage summary generation ──────────────────────────────────────


def generate_stage_summary(
    window_days: int = STAGE_WEEKLY_DAYS,
    label: str = "",
    write_to_store: bool = True,
) -> Dict[str, Any]:
    """
    Generate a consolidated research progress summary from recent memories.

    Groups memories created within the last *window_days* and produces a
    structured markdown digest.  If *write_to_store* is True, the summary
    is also appended as a new ``report_summary`` memory record.

    Args:
        window_days: Look-back window in days (7 = weekly, 30 = monthly).
        label: Human-readable label for the summary
               (default: "Weekly" for 7 days, "Monthly" for 30).
        write_to_store: If True, write the summary as a new memory record.

    Returns::

        {
            "ok": bool,
            "summary_text": str,
            "window_days": int,
            "label": str,
            "records_covered": int,
            "written_memory_id": str,
            "error": str,
        }
    """
    try:
        if not label:
            label = "Weekly" if window_days <= 10 else "Monthly"

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=window_days)
        cutoff_str = cutoff.isoformat()

        # Load all records and filter by date
        all_records = load_memories()
        recent: List[Dict[str, Any]] = []
        for r in all_records:
            created = r.get("created_at", "")
            created_dt = _parse_iso_date(created)
            if created_dt and created_dt >= cutoff:
                recent.append(r)

        if not recent:
            summary_text = (
                f"# {label} Research Summary\n\n"
                f"*Generated: {now.isoformat()}*\n"
                f"*Window: last {window_days} days*\n\n"
                f"No memories recorded in this period.\n"
            )

            return {
                "ok": True,
                "summary_text": summary_text,
                "window_days": window_days,
                "label": label,
                "records_covered": 0,
                "written_memory_id": "",
                "error": "",
            }

        # Group by memory_type
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in recent:
            mt = r.get("memory_type", "general_note")
            by_type[mt].append(r)

        # Build summary
        lines: List[str] = [
            f"# {label} Research Summary",
            "",
            f"*Generated: {now.isoformat()}*",
            f"*Window: {cutoff_str[:10]} → {now.strftime('%Y-%m-%d')} ({window_days} days)*",
            f"*Records covered: {len(recent)}*",
            "",
            "---",
            "",
        ]

        type_labels = {
            "research_direction": "## Research Directions",
            "paper_note": "## Paper Reading",
            "claim_support": "## Claim Support & Arguments",
            "progress_update": "## Progress Updates",
            "experiment_result": "## Experiment Results",
            "report_summary": "## Report Summaries",
            "todo": "## Tasks & TODOs",
            "issue": "## Issues & Limitations",
            "meeting_note": "## Meeting Notes",
            "code_note": "## Code Notes",
            "project_decision": "## Project Decisions",
            "user_preference": "## User Preferences",
            "general_note": "## General Notes",
        }

        for mt in sorted(by_type.keys()):
            items = by_type[mt]
            heading = type_labels.get(mt, f"## {mt}")
            lines.append(heading)
            lines.append("")

            for r in items:
                summary = r.get("summary", "") or r.get("content", "")[:150]
                mid = r.get("memory_id", "?")[:8]
                source = r.get("source_title", "") or r.get("source_path", "")
                tags = ", ".join(r.get("tags", [])[:5]) or "(none)"
                level = r.get("memory_level", "?")

                lines.append(f"- **[{mid}]** ({level}) {summary}")
                if source:
                    lines.append(f"  - source: {source}")
                if tags != "(none)":
                    lines.append(f"  - tags: {tags}")
                lines.append("")

            lines.append("")

        # Add tag cloud
        all_tags: List[str] = []
        for r in recent:
            all_tags.extend(r.get("tags", []))
        tag_counts = Counter(all_tags)
        if tag_counts:
            lines.append("## Tag Cloud")
            lines.append("")
            top_tags = tag_counts.most_common(15)
            lines.append(", ".join(f"`{t}` ({c})" for t, c in top_tags))
            lines.append("")

        summary_text = "\n".join(lines)

        # Write to store as a new memory
        written_id = ""
        if write_to_store:
            stage_record = _make_record_dict(
                memory_type="report_summary",
                memory_level="long_term",
                memory_scope="shared",
                owner_agent="memory_agent",
                content=summary_text,
                summary=f"{label} research summary covering {len(recent)} records",
                source_title=f"{label} Summary {now.strftime('%Y-%m-%d')}",
                source_path="",
                tags=["summary", label.lower(), "consolidation"],
                status="active",
            )
            _append_jsonl(MEMORY_STORE_PATH, stage_record)
            written_id = stage_record["memory_id"]

            # Also write to level files
            from .store import _LEVEL_FILE_MAP
            level_path = _LEVEL_FILE_MAP.get(stage_record["memory_level"])
            if level_path:
                _append_jsonl(level_path, stage_record)

            # Update the markdown summary
            update_memory_summary()

        return {
            "ok": True,
            "summary_text": summary_text,
            "window_days": window_days,
            "label": label,
            "records_covered": len(recent),
            "written_memory_id": written_id,
            "error": "",
        }

    except Exception as e:
        return {
            "ok": False,
            "summary_text": "",
            "window_days": window_days,
            "label": label,
            "records_covered": 0,
            "written_memory_id": "",
            "error": str(e),
        }


# ── 6. Run all consolidation ─────────────────────────────────────────


def run_consolidation(
    merge_similarity: float = DEFAULT_MERGE_SIMILARITY_THRESHOLD,
    compress_threshold: int = DEFAULT_COMPRESS_THRESHOLD,
    compress_target: int = DEFAULT_COMPRESS_TARGET_LENGTH,
    short_term_expiry_days: int = SHORT_TERM_EXPIRY_DAYS,
    mid_term_expiry_days: int = MID_TERM_EXPIRY_DAYS,
    stage_window_days: int = STAGE_WEEKLY_DAYS,
    stage_label: str = "",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Run all consolidation operations in sequence.

    Order:
    1. Merge duplicates
    2. Compress long memories
    3. Mark expired memories
    4. Generate stage summary

    Args:
        merge_similarity: Jaccard threshold for merge.
        compress_threshold: Content length that triggers compression.
        compress_target: Target content length after compression.
        short_term_expiry_days: Days before short-term expires.
        mid_term_expiry_days: Days before mid-term expires.
        stage_window_days: Days in the stage summary window.
        stage_label: Label for the stage summary.
        dry_run: If True, no writes are performed.

    Returns::

        {
            "ok": bool,
            "merge_result": {...},
            "compress_result": {...},
            "expire_result": {...},
            "stage_summary_result": {...},
            "dry_run": bool,
        }
    """
    results: Dict[str, Any] = {
        "ok": True,
        "dry_run": dry_run,
    }

    # 1. Merge
    merge_result = merge_duplicates(
        similarity_threshold=merge_similarity,
        dry_run=dry_run,
    )
    results["merge_result"] = merge_result
    if not merge_result["ok"]:
        results["ok"] = False

    # 2. Compress
    compress_result = compress_long_memories(
        threshold=compress_threshold,
        target_length=compress_target,
        dry_run=dry_run,
    )
    results["compress_result"] = compress_result
    if not compress_result["ok"]:
        results["ok"] = False

    # 3. Expire
    expire_result = mark_expired_memories(
        short_term_days=short_term_expiry_days,
        mid_term_days=mid_term_expiry_days,
        dry_run=dry_run,
    )
    results["expire_result"] = expire_result
    if not expire_result["ok"]:
        results["ok"] = False

    # 4. Stage summary
    stage_result = generate_stage_summary(
        window_days=stage_window_days,
        label=stage_label,
        write_to_store=not dry_run,
    )
    results["stage_summary_result"] = stage_result
    if not stage_result["ok"]:
        results["ok"] = False

    # Final summary update (non-dry-run only)
    if not dry_run and results["ok"]:
        try:
            update_memory_summary()
        except Exception:
            pass

    return results


# ── Helpers from collections (import here to avoid circular imports) ──

from collections import Counter  # noqa: E402
