"""
Memory Retriever v2 — multi-dimensional search over memory records.

Reads memory records through the Memory Store (:mod:`research_agent.memory.store`)
as the primary backend. Falls back to direct JSONL file reading if the store
module is unavailable.

Retrieval supports filtering by type, tags, source module, time range,
keyword, memory level, agent role, importance, and status.

Backward-compatible with v1 — all existing function signatures and
return types are preserved.
"""

from __future__ import annotations

import json
import sys as _sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ── Internal backend state ─────────────────────────────────────────

_backend_state: Dict = {
    "uses_store": False,
    "fallback_reason": "",
    "record_count": 0,
}


def get_retriever_backend_status() -> Dict:
    """
    Return the current retriever backend status.

    Useful for debugging and integration tests::

        >>> status = get_retriever_backend_status()
        >>> print(status["uses_store"])   # True if store is active
        >>> print(status["fallback_reason"])
        >>> print(status["record_count"])
    """
    return dict(_backend_state)


# ── 1. Load (primary: store, fallback: direct JSONL) ───────────────


def load_memories(jsonl_path: str | Path) -> List[Dict]:
    """
    Load all memory records from a single JSONL file on disk.

    This is the **direct file reader** — it does NOT go through the store.
    Use ``load_memories_for_retrieval()`` for the store-backed path.

    Each line must be a valid JSON object. Malformed lines are
    skipped with a warning printed to stderr.

    Returns a list of dicts; never raises on bad lines.
    """
    path = Path(jsonl_path)
    records: List[Dict] = []

    if not path.exists():
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if isinstance(record, dict):
                    records.append(record)
            except json.JSONDecodeError:
                print(
                    f"[memory_retriever] skipping malformed JSONL "
                    f"line {line_no} in {jsonl_path}",
                    file=_sys.stderr,
                )

    return records


def load_memories_from_dir(dir_path: str | Path) -> List[Dict]:
    """
    Load all memory records from all ``*.jsonl`` files in a directory.

    Files are processed in sorted order for deterministic results.
    Direct file reader — does not go through the store.
    """
    directory = Path(dir_path)
    if not directory.is_dir():
        return []

    all_records: List[Dict] = []
    for jsonl_file in sorted(directory.glob("*.jsonl")):
        all_records.extend(load_memories(jsonl_file))

    return all_records


def load_memories_for_retrieval(
    jsonl_path: Optional[str | Path] = None,
    prefer_store: bool = True,
) -> List[Dict]:
    """
    Load all memory records, preferring the Memory Store backend.

    Priority order:
    1. If ``jsonl_path`` is explicitly provided AND the file exists,
       load from that path directly (used for standalone/test scenarios).
    2. If ``prefer_store=True``, try ``research_agent.memory.store.load_memories()``.
    3. Fall back to the default store path ``data/memory/memory_store.jsonl``.

    Updates ``get_retriever_backend_status()`` with the outcome.
    """
    # --- Explicit path (test / standalone) ---
    if jsonl_path is not None:
        explicit = Path(jsonl_path)
        if explicit.exists():
            records = load_memories(explicit)
            _backend_state["uses_store"] = False
            _backend_state["fallback_reason"] = "explicit jsonl_path provided"
            _backend_state["record_count"] = len(records)
            return records

    # --- Try store.load_memories() ---
    if prefer_store:
        try:
            from research_agent.memory.store import load_memories as _store_load

            records = _store_load()
            _backend_state["uses_store"] = True
            _backend_state["fallback_reason"] = ""
            _backend_state["record_count"] = len(records)
            return records

        except ImportError as e:
            _backend_state["uses_store"] = False
            _backend_state["fallback_reason"] = f"ImportError: {e}"
        except Exception as e:
            _backend_state["uses_store"] = False
            _backend_state["fallback_reason"] = f"{type(e).__name__}: {e}"

    # --- Fallback: default store path ---
    try:
        project_root = Path(__file__).resolve().parents[3]
        default_store = project_root / "data" / "memory" / "memory_store.jsonl"
        records = load_memories(default_store)
        _backend_state["record_count"] = len(records)
        return records
    except Exception as e:
        _backend_state["fallback_reason"] += f" | Default path failed: {e}"
        _backend_state["record_count"] = 0
        return []


# ── 2. Filter predicates ──────────────────────────────────────────


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    """Parse an ISO-8601 string to a timezone-aware datetime, or None."""
    if not value:
        return None
    try:
        s = value.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _match_keyword(record: Dict, keyword: str) -> bool:
    """Check if keyword appears (case-insensitive) in content, summary, or source_title."""
    kw = keyword.lower()
    for field in ("content", "summary", "source_title"):
        text = record.get(field, "")
        if isinstance(text, str) and kw in text.lower():
            return True
    return False


def _match_tag_any(record: Dict, query_tags: List[str]) -> bool:
    """Check if ANY of the query_tags appear in the record's tags (OR logic)."""
    record_tags: List[str] = record.get("tags", [])
    if not record_tags:
        return False
    record_set = {t.lower() for t in record_tags}
    return any(qt.lower() in record_set for qt in query_tags)


# ── 3. Main retrieval ─────────────────────────────────────────────


def retrieve_memories(
    records: List[Dict],
    *,
    # Exact-match filters
    memory_type: Optional[str] = None,
    memory_level: Optional[str] = None,
    memory_scope: Optional[str] = None,
    source_module: Optional[str] = None,
    owner_agent: Optional[str] = None,
    status: Optional[str] = None,
    # Multi-value filter
    tags: Optional[List[str]] = None,
    # Time range
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    updated_after: Optional[str] = None,
    updated_before: Optional[str] = None,
    # Keyword (case-insensitive, searches content + summary + source_title)
    keyword: Optional[str] = None,
    # Importance range
    importance_min: Optional[int] = None,
    importance_max: Optional[int] = None,
    # Result control
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Retrieve memory records matching the given filter criteria.

    All filter parameters are optional and combined with AND logic.
    ``tags`` uses OR logic — a record matches if ANY query tag appears.

    Parameters
    ----------
    records : List[Dict]
        Pre-loaded memory records (from ``load_memories`` or
        ``load_memories_for_retrieval``).
    memory_type : str, optional
        Exact match on ``memory_type`` field.
    memory_level : str, optional
        Exact match on ``memory_level`` field.
    memory_scope : str, optional
        Exact match on ``memory_scope`` field.
    source_module : str, optional
        Exact match on ``source_module`` field.
    owner_agent : str, optional
        Exact match on ``owner_agent`` field.
    status : str, optional
        Exact match on ``status`` field.
    tags : List[str], optional
        Match if ANY tag in the list appears in the record's tags.
    created_after : str, optional
        ISO-8601 timestamp — keep records created at or after this time.
    created_before : str, optional
        ISO-8601 timestamp — keep records created at or before this time.
    updated_after : str, optional
        ISO-8601 timestamp for updated_at.
    updated_before : str, optional
        ISO-8601 timestamp for updated_at.
    keyword : str, optional
        Case-insensitive substring search in content, summary, source_title.
    importance_min : int, optional
        Minimum importance (inclusive).
    importance_max : int, optional
        Maximum importance (inclusive).
    limit : int, optional
        Return at most this many records.

    Returns
    -------
    List[Dict]
        Matching records, preserving original order among matches.
    """
    ca = _parse_iso_datetime(created_after) if created_after else None
    cb = _parse_iso_datetime(created_before) if created_before else None
    ua = _parse_iso_datetime(updated_after) if updated_after else None
    ub = _parse_iso_datetime(updated_before) if updated_before else None

    results: List[Dict] = []

    for rec in records:
        if memory_type is not None and rec.get("memory_type") != memory_type:
            continue
        if memory_level is not None and rec.get("memory_level") != memory_level:
            continue
        if memory_scope is not None and rec.get("memory_scope") != memory_scope:
            continue
        if source_module is not None and rec.get("source_module") != source_module:
            continue
        if owner_agent is not None and rec.get("owner_agent") != owner_agent:
            continue
        if status is not None and rec.get("status") != status:
            continue

        if tags and not _match_tag_any(rec, tags):
            continue

        if keyword and not _match_keyword(rec, keyword):
            continue

        if ca is not None:
            ct = _parse_iso_datetime(rec.get("created_at", ""))
            if ct is None or ct < ca:
                continue
        if cb is not None:
            ct = _parse_iso_datetime(rec.get("created_at", ""))
            if ct is None or ct > cb:
                continue
        if ua is not None:
            ut = _parse_iso_datetime(rec.get("updated_at", ""))
            if ut is None or ut < ua:
                continue
        if ub is not None:
            ut = _parse_iso_datetime(rec.get("updated_at", ""))
            if ut is None or ut > ub:
                continue

        imp = rec.get("importance", 3)
        if importance_min is not None and imp < importance_min:
            continue
        if importance_max is not None and imp > importance_max:
            continue

        results.append(rec)

    if limit is not None:
        results = results[:limit]

    return results


# ── 4. Convenience: load + retrieve in one call ───────────────────


def retrieve_from_store(
    jsonl_path: Optional[str | Path] = None,
    *,
    memory_type: Optional[str] = None,
    memory_level: Optional[str] = None,
    memory_scope: Optional[str] = None,
    source_module: Optional[str] = None,
    owner_agent: Optional[str] = None,
    status: Optional[str] = None,
    tags: Optional[List[str]] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    updated_after: Optional[str] = None,
    updated_before: Optional[str] = None,
    keyword: Optional[str] = None,
    importance_min: Optional[int] = None,
    importance_max: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Load records via the store (or fallback) and retrieve matching entries.

    If ``jsonl_path`` is provided and the file exists, reads from that file
    directly (useful for standalone test scenarios).  Otherwise delegates
    to ``load_memories_for_retrieval`` (which prefers ``store.load_memories()``).

    This is the recommended entry point for most callers.
    """
    records = load_memories_for_retrieval(
        jsonl_path=jsonl_path,
        prefer_store=(jsonl_path is None),  # only prefer store when no explicit path
    )
    return retrieve_memories(
        records,
        memory_type=memory_type,
        memory_level=memory_level,
        memory_scope=memory_scope,
        source_module=source_module,
        owner_agent=owner_agent,
        status=status,
        tags=tags,
        created_after=created_after,
        created_before=created_before,
        updated_after=updated_after,
        updated_before=updated_before,
        keyword=keyword,
        importance_min=importance_min,
        importance_max=importance_max,
        limit=limit,
    )
