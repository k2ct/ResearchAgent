"""
Memory Store v1 — JSONL + Markdown summary.

Stores structured memory records to JSONL files organised by memory level
(long-term / mid-term / short-term / shared) and generates a human-readable
markdown summary.

Design:
- JSONL is the authoritative data store (append-only).
- Markdown summary is regenerated on demand.
- No Chroma / vector search in v1.
- No semantic retrieval in v1.
- Does NOT decide "whether to write a memory" — that is the caller's job.

Schema compatibility:
- Tries to import ``MemoryRecord`` from ``.schema``.
- If ``schema.py`` is not yet merged, falls back to a lightweight dict-based
  approach that is fully compatible with the expected schema shape.
- The fallback can be removed once the schema module is merged.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Try to import schema (may not exist yet) ─────────────────────────

try:
    from .schema import (
        MemoryRecord,
        create_memory_record,
        memory_record_to_dict,
    )
    _HAS_SCHEMA = True
except ImportError:
    _HAS_SCHEMA = False


# ── Constants ────────────────────────────────────────────────────────

# Use project-root-relative paths — callers should chdir to project root
# or pass absolute paths.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEMORY_DIR = _PROJECT_ROOT / "data" / "memory"

MEMORY_STORE_PATH = MEMORY_DIR / "memory_store.jsonl"
LONG_TERM_PATH = MEMORY_DIR / "long_term_memory.jsonl"
MID_TERM_PATH = MEMORY_DIR / "mid_term_memory.jsonl"
SHORT_TERM_PATH = MEMORY_DIR / "short_term_memory.jsonl"
SHARED_MEMORY_PATH = MEMORY_DIR / "shared_memory.jsonl"
MEMORY_SUMMARY_PATH = MEMORY_DIR / "memory_summary.md"

_VALID_LEVELS = {"long_term", "mid_term", "short_term"}
_LEVEL_FILE_MAP: Dict[str, Path] = {
    "long_term": LONG_TERM_PATH,
    "mid_term": MID_TERM_PATH,
    "short_term": SHORT_TERM_PATH,
}


# ── 1. Initialisation ────────────────────────────────────────────────


def ensure_memory_store() -> Dict[str, Any]:
    """
    Create the memory store directory and all required files.

    Idempotent — safe to call multiple times.

    Returns::

        {"ok": bool, "paths_created": [...], "error": str}
    """
    try:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        # .gitkeep so the directory can be committed
        gitkeep = MEMORY_DIR / ".gitkeep"
        gitkeep.touch(exist_ok=True)

        paths: List[Path] = [
            MEMORY_STORE_PATH,
            LONG_TERM_PATH,
            MID_TERM_PATH,
            SHORT_TERM_PATH,
            SHARED_MEMORY_PATH,
            MEMORY_SUMMARY_PATH,
        ]

        created: List[str] = []
        for p in paths:
            if not p.exists():
                p.touch(exist_ok=True)
                created.append(str(p))

        return {"ok": True, "paths_created": created, "error": ""}

    except Exception as e:
        return {"ok": False, "paths_created": [], "error": str(e)}


# ── 2. Schema fallback helpers ───────────────────────────────────────


def _make_record_dict(
    memory_type: str = "general_note",
    memory_level: str = "short_term",
    memory_scope: str = "private",
    owner_agent: str = "research_agent",
    content: str = "",
    summary: str = "",
    source_title: str = "",
    source_path: str = "",
    tags: Optional[List[str]] = None,
    status: str = "active",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a memory record dict (fallback when schema.py is not available).

    Produces the same dict shape that ``memory_record_to_dict()`` would.
    """
    now = datetime.now(timezone.utc).isoformat()
    return {
        "memory_id": str(uuid.uuid4()),
        "memory_type": memory_type,
        "memory_level": memory_level,
        "memory_scope": memory_scope,
        "owner_agent": owner_agent,
        "content": content,
        "summary": summary,
        "source_title": source_title,
        "source_path": source_path,
        "tags": tags or [],
        "status": status,
        "metadata": metadata or {},
        "created_at": now,
        "updated_at": now,
    }


def _normalise_record(record: Any) -> Dict[str, Any]:
    """
    Normalise a memory record to a plain dict.

    Accepts:
    - A dict (used as-is, with defaults filled)
    - A ``MemoryRecord`` object (converted via ``memory_record_to_dict``)
    """
    if _HAS_SCHEMA and isinstance(record, MemoryRecord):
        d = memory_record_to_dict(record)
    elif isinstance(record, dict):
        d = dict(record)
    else:
        raise TypeError(
            f"Expected dict or MemoryRecord, got {type(record).__name__}"
        )

    # Migrate legacy record_id → memory_id
    if "memory_id" not in d and "record_id" in d:
        d["memory_id"] = d.pop("record_id")
        d.setdefault("metadata", {})["_migrated_from_record_id"] = True

    # Ensure required keys exist
    d.setdefault("memory_id", str(uuid.uuid4()))
    d.setdefault("memory_type", "general_note")
    d.setdefault("memory_level", "short_term")
    d.setdefault("memory_scope", "private")
    d.setdefault("owner_agent", "research_agent")
    d.setdefault("content", "")
    d.setdefault("summary", "")
    d.setdefault("source_title", "")
    d.setdefault("source_path", "")
    d.setdefault("source_module", "manual")
    d.setdefault("source_id", "")
    d.setdefault("tags", [])
    d.setdefault("shared_with", [])
    d.setdefault("status", "active")
    d.setdefault("visibility", "private")
    d.setdefault("importance", 3)
    d.setdefault("last_accessed_at", None)
    d.setdefault("metadata", {})
    now = datetime.now(timezone.utc).isoformat()
    d.setdefault("created_at", now)
    d.setdefault("updated_at", now)

    return d


# ── 3. Write / Read ──────────────────────────────────────────────────


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read all records from a JSONL file. Returns empty list if file missing."""
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _rewrite_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Overwrite a JSONL file with the given records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── 4. Append memory ─────────────────────────────────────────────────


def append_memory(
    record: Any,
    write_level_file: bool = True,
    update_summary: bool = True,
) -> Dict[str, Any]:
    """
    Append a memory record to the store.

    Args:
        record: A dict (or ``MemoryRecord`` if schema.py is available).
        write_level_file: If True, also write to the level-specific JSONL
            (long_term / mid_term / short_term).
        update_summary: If True, regenerate the markdown summary after writing.

    Returns::

        {"ok": bool, "memory_id": str, "paths_written": [...], "error": str}
    """
    try:
        d = _normalise_record(record)
        memory_id = d["memory_id"]

        paths_written: List[str] = []

        # Always write to the main store
        _append_jsonl(MEMORY_STORE_PATH, d)
        paths_written.append(str(MEMORY_STORE_PATH))

        # Write to level-specific file
        if write_level_file:
            level = d.get("memory_level", "short_term")
            level_path = _LEVEL_FILE_MAP.get(level)
            if level_path:
                _append_jsonl(level_path, d)
                paths_written.append(str(level_path))

        # Write to shared memory if scope is shared or global
        scope = d.get("memory_scope", "private")
        if scope in ("shared", "global"):
            _append_jsonl(SHARED_MEMORY_PATH, d)
            paths_written.append(str(SHARED_MEMORY_PATH))

        # Regenerate summary
        if update_summary:
            update_memory_summary()

        return {
            "ok": True,
            "memory_id": memory_id,
            "paths_written": paths_written,
            "error": "",
        }

    except Exception as e:
        return {
            "ok": False,
            "memory_id": "",
            "paths_written": [],
            "error": str(e),
        }


# ── 5. Load memories ─────────────────────────────────────────────────


def load_memories(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load all records from memory_store.jsonl (or a custom path)."""
    return _read_jsonl(path or MEMORY_STORE_PATH)


def load_memories_by_level(memory_level: str) -> List[Dict[str, Any]]:
    """Load records for a specific memory level."""
    level_path = _LEVEL_FILE_MAP.get(memory_level)
    if level_path is None:
        raise ValueError(
            f"Invalid memory_level: {memory_level!r}. "
            f"Valid levels: {sorted(_VALID_LEVELS)}"
        )
    return _read_jsonl(level_path)


def load_shared_memories() -> List[Dict[str, Any]]:
    """Load all shared / global memories."""
    return _read_jsonl(SHARED_MEMORY_PATH)


# ── 6. Query ─────────────────────────────────────────────────────────


def query_memories(
    memory_type: Optional[str] = None,
    memory_level: Optional[str] = None,
    owner_agent: Optional[str] = None,
    tags: Optional[List[str]] = None,
    keyword: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Query memories with optional filters.

    All filters are ANDed together.  ``tags`` matches if *any* of the
    requested tags are present in the record's tags.

    ``keyword`` does a case-insensitive substring search across
    ``content``, ``summary``, and ``source_title``.

    Results are returned in insertion order (oldest first), truncated to
    *limit*.
    """
    records = load_memories()
    keyword_lower = keyword.lower() if keyword else None
    tag_set = set(tags) if tags else None

    matched: List[Dict[str, Any]] = []

    for r in records:
        if memory_type is not None and r.get("memory_type") != memory_type:
            continue
        if memory_level is not None and r.get("memory_level") != memory_level:
            continue
        if owner_agent is not None and r.get("owner_agent") != owner_agent:
            continue

        if tag_set is not None:
            record_tags = set(r.get("tags", []))
            if not tag_set.intersection(record_tags):
                continue

        if keyword_lower is not None:
            haystack = " ".join([
                r.get("content", ""),
                r.get("summary", ""),
                r.get("source_title", ""),
            ]).lower()
            if keyword_lower not in haystack:
                continue

        matched.append(r)

    return matched[:limit]


# ── 7. Markdown summary ──────────────────────────────────────────────


def update_memory_summary() -> str:
    """
    Regenerate the human-readable memory summary markdown.

    Groups records by memory_level, then by memory_type within each level.
    Shows the most recent records (up to 30 per level).

    Does NOT include full ``content`` — only metadata + summary + tags.

    Returns the generated markdown string.
    """
    all_records = load_memories()

    # Build grouped structure
    by_level: Dict[str, List[Dict[str, Any]]] = {
        "long_term": [],
        "mid_term": [],
        "short_term": [],
    }
    for r in all_records:
        level = r.get("memory_level", "short_term")
        if level in by_level:
            by_level[level].append(r)

    lines: List[str] = [
        "# ResearchAgent Memory Summary",
        "",
        f"*Generated: {datetime.now(timezone.utc).isoformat()}*",
        f"*Total records: {len(all_records)}*",
        "",
        "---",
        "",
    ]

    level_labels = {
        "long_term": "## Long-term Memory",
        "mid_term": "## Mid-term Memory",
        "short_term": "## Short-term Memory",
    }

    for level_key in ("long_term", "mid_term", "short_term"):
        records = by_level.get(level_key, [])
        lines.append(level_labels[level_key])
        lines.append("")

        if not records:
            lines.append("*No records.*")
            lines.append("")
            continue

        # Group by memory_type
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            mt = r.get("memory_type", "general_note")
            by_type.setdefault(mt, []).append(r)

        for mt, items in sorted(by_type.items()):
            lines.append(f"### {mt} ({len(items)} records)")
            lines.append("")
            for r in items[-5:]:  # most recent 5 per type
                mid = r.get("memory_id", "?")[:8]
                owner = r.get("owner_agent", "?")
                tags = ", ".join(r.get("tags", [])) or "(none)"
                summary = r.get("summary", "") or r.get("content", "")[:120]
                source = r.get("source_title", "") or r.get("source_path", "")
                scope = r.get("memory_scope", "private")

                lines.append(f"- **[{mid}]** `{scope}` {summary}")
                if source:
                    lines.append(f"  - source: {source}")
                lines.append(f"  - tags: {tags}  |  owner: {owner}")
                lines.append("")

        lines.append("")

    # Shared memory section
    shared = _read_jsonl(SHARED_MEMORY_PATH)
    lines.append("## Shared Memory")
    lines.append("")
    if not shared:
        lines.append("*No shared memories.*")
    else:
        for r in shared[-10:]:
            mid = r.get("memory_id", "?")[:8]
            summary = r.get("summary", "") or r.get("content", "")[:120]
            source = r.get("source_title", "") or r.get("source_path", "")
            lines.append(f"- **[{mid}]** {summary}")
            if source:
                lines.append(f"  - source: {source}")
        lines.append("")

    summary_text = "\n".join(lines)

    # Write to file
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_SUMMARY_PATH.write_text(summary_text, encoding="utf-8")

    return summary_text


# ── 8. Clear short-term ──────────────────────────────────────────────


def clear_short_term_memory(confirm: bool = False) -> Dict[str, Any]:
    """
    Clear the short_term_memory.jsonl file.

    Only acts if ``confirm=True``.  Does NOT touch memory_store.jsonl
    (historical records are preserved in the main store).
    """
    if not confirm:
        return {
            "ok": False,
            "error": "clear_short_term_memory requires confirm=True",
        }

    try:
        SHORT_TERM_PATH.write_text("", encoding="utf-8")
        return {"ok": True, "error": ""}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── 9. Archive (optional, lightweight) ───────────────────────────────


def archive_memory(memory_id: str, confirm: bool = False) -> Dict[str, Any]:
    """
    Mark a memory record as ``archived`` in memory_store.jsonl.

    Note: This rewrites the entire main store file.  For large stores this
    could be slow; v2 should use a database or indexed file.
    """
    if not confirm:
        return {
            "ok": False,
            "error": "archive_memory requires confirm=True",
        }

    records = load_memories()
    updated = False

    for r in records:
        if r.get("memory_id") == memory_id:
            r["status"] = "archived"
            r["updated_at"] = datetime.now(timezone.utc).isoformat()
            updated = True
            break

    if not updated:
        return {
            "ok": False,
            "error": f"Memory not found: {memory_id}",
        }

    _rewrite_jsonl(MEMORY_STORE_PATH, records)
    update_memory_summary()

    return {"ok": True, "error": ""}
