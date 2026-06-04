"""
Unified Memory Schema for ResearchAgent.

Defines the canonical data format for long-term, mid-term, and short-term
research memories. This module provides types, a factory function,
validation, and serialisation helpers.

Schema-only — no storage I/O, no writer logic, no LangGraph integration.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional


# ── 1. Enumerated Value Sets ──────────────────────────────────────

MemoryLevel = Literal[
    "long_term",
    "mid_term",
    "short_term",
]

MemoryScope = Literal[
    "private",
    "shared",
    "global",
]

AgentRole = Literal[
    "coordinator",
    "paper_agent",
    "experiment_agent",
    "report_agent",
    "progress_agent",
    "claim_agent",
    "code_agent",
    "memory_agent",
    "general_agent",
    "user",
]

MemoryType = Literal[
    "research_direction",
    "claim_support",
    "paper_note",
    "progress_update",
    "experiment_result",
    "report_summary",
    "code_note",
    "project_decision",
    "user_preference",
    "todo",
    "issue",
    "meeting_note",
    "general_note",
]

SourceModule = Literal[
    "claim_support",
    "paper_reading",
    "ppt_progress",
    "experiment_tool",
    "report_writer",
    "manual",
    "chat",
    "code_assistant",
    "system",
]

MemoryStatus = Literal[
    "active",
    "archived",
    "expired",
]

Visibility = Literal[
    "private",
    "internal",
    "public",
]

# ── 2. Validator containers (for fast membership checks) ──────────

_VALID_MEMORY_LEVELS: set = {
    "long_term", "mid_term", "short_term",
}
_VALID_MEMORY_SCOPES: set = {
    "private", "shared", "global",
}
_VALID_AGENT_ROLES: set = {
    "coordinator", "paper_agent", "experiment_agent", "report_agent",
    "progress_agent", "claim_agent", "code_agent", "memory_agent",
    "general_agent", "user",
}
_VALID_MEMORY_TYPES: set = {
    "research_direction", "claim_support", "paper_note", "progress_update",
    "experiment_result", "report_summary", "code_note", "project_decision",
    "user_preference", "todo", "issue", "meeting_note", "general_note",
}
_VALID_SOURCE_MODULES: set = {
    "claim_support", "paper_reading", "ppt_progress", "experiment_tool",
    "report_writer", "manual", "chat", "code_assistant", "system",
}


# ── 3. MemoryRecord Dataclass ─────────────────────────────────────


@dataclass
class MemoryRecord:
    """
    Unified memory record for ResearchAgent.

    Field glossary
    --------------
    memory_id : str
        Unique identifier, e.g. ``mem_20260605_a1b2c3d4``.
    memory_level : MemoryLevel
        long_term | mid_term | short_term
    memory_scope : MemoryScope
        private | shared | global
    memory_type : MemoryType
        Semantic category of the memory content.
    owner_agent : AgentRole
        The agent that originally created this memory.
    shared_with : List[str]
        Agent roles permitted to read this memory (meaningful when scope=shared).

    content : str
        Full memory text.
    summary : str
        Short abstract, auto-generated from ``content`` if omitted.
    tags : List[str]
        Free-form keyword tags.

    source_module : SourceModule
        Which subsystem produced this memory.
    source_path : str
        File path or resource identifier of the source material.
    source_id : str
        Internal identifier from the source system (may be empty).
    source_title : str
        Human-readable title of the source.

    created_at : str
        ISO-8601 creation timestamp (UTC).
    updated_at : str
        ISO-8601 last-modification timestamp (UTC).
    last_accessed_at : Optional[str]
        ISO-8601 timestamp of last read access.

    importance : int
        1 (trivial) – 5 (critical). Default 3.
    status : MemoryStatus
        active | archived | expired
    visibility : Visibility
        private | internal | public — who outside the agent ecosystem can see this.

    metadata : Dict
        Arbitrary extension key-value pairs.
    """

    memory_id: str = ""
    memory_level: MemoryLevel = "short_term"
    memory_scope: MemoryScope = "private"
    memory_type: MemoryType = "general_note"
    owner_agent: AgentRole = "general_agent"
    shared_with: List[str] = field(default_factory=list)

    content: str = ""
    summary: str = ""
    tags: List[str] = field(default_factory=list)

    source_module: SourceModule = "manual"
    source_path: str = ""
    source_id: str = ""
    source_title: str = ""

    created_at: str = ""
    updated_at: str = ""
    last_accessed_at: Optional[str] = None

    importance: int = 3
    status: MemoryStatus = "active"
    visibility: Visibility = "private"

    metadata: Dict = field(default_factory=dict)


# ── 4. Factory Function ───────────────────────────────────────────


def _generate_memory_id() -> str:
    """Generate a unique memory ID: mem_YYYYMMDD_hex8."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    short_hex = uuid.uuid4().hex[:8]
    return f"mem_{today}_{short_hex}"


def _utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def create_memory_record(
    content: str,
    memory_level: MemoryLevel,
    memory_scope: MemoryScope,
    memory_type: MemoryType,
    owner_agent: AgentRole,
    source_module: SourceModule = "manual",
    summary: str = "",
    tags: Optional[List[str]] = None,
    source_path: str = "",
    source_title: str = "",
    source_id: str = "",
    shared_with: Optional[List[str]] = None,
    importance: int = 3,
    status: MemoryStatus = "active",
    visibility: Visibility = "private",
    metadata: Optional[Dict] = None,
) -> MemoryRecord:
    """
    Create a fully populated MemoryRecord with sensible defaults.

    ``memory_id``, ``created_at``, and ``updated_at`` are auto-generated.
    ``summary`` falls back to the first 120 characters of ``content``.
    """

    now = _utc_now_iso()

    if not summary:
        summary = content[:120].replace("\n", " ").strip()

    return MemoryRecord(
        memory_id=_generate_memory_id(),
        memory_level=memory_level,
        memory_scope=memory_scope,
        memory_type=memory_type,
        owner_agent=owner_agent,
        shared_with=shared_with or [],
        content=content,
        summary=summary,
        tags=tags or [],
        source_module=source_module,
        source_path=source_path,
        source_id=source_id,
        source_title=source_title,
        created_at=now,
        updated_at=now,
        last_accessed_at=None,
        importance=importance,
        status=status,
        visibility=visibility,
        metadata=metadata or {},
    )


# ── 5. Validation ─────────────────────────────────────────────────


def validate_memory_record(record: MemoryRecord) -> List[str]:
    """
    Validate a MemoryRecord and return a list of error messages.

    Returns an empty list if the record is valid.
    """
    errors: List[str] = []

    # memory_id
    if not record.memory_id or not record.memory_id.strip():
        errors.append("memory_id is empty")

    # memory_level
    if record.memory_level not in _VALID_MEMORY_LEVELS:
        errors.append(
            f"invalid memory_level '{record.memory_level}'; "
            f"must be one of {sorted(_VALID_MEMORY_LEVELS)}"
        )

    # memory_scope
    if record.memory_scope not in _VALID_MEMORY_SCOPES:
        errors.append(
            f"invalid memory_scope '{record.memory_scope}'; "
            f"must be one of {sorted(_VALID_MEMORY_SCOPES)}"
        )

    # memory_type
    if record.memory_type not in _VALID_MEMORY_TYPES:
        errors.append(
            f"invalid memory_type '{record.memory_type}'; "
            f"must be one of {sorted(_VALID_MEMORY_TYPES)}"
        )

    # owner_agent
    if record.owner_agent not in _VALID_AGENT_ROLES:
        errors.append(
            f"invalid owner_agent '{record.owner_agent}'; "
            f"must be one of {sorted(_VALID_AGENT_ROLES)}"
        )

    # source_module
    if record.source_module not in _VALID_SOURCE_MODULES:
        errors.append(
            f"invalid source_module '{record.source_module}'; "
            f"must be one of {sorted(_VALID_SOURCE_MODULES)}"
        )

    # importance range
    if not (1 <= record.importance <= 5):
        errors.append(
            f"importance must be 1-5, got {record.importance}"
        )

    # content must not be empty
    if not record.content or not record.content.strip():
        errors.append("content is empty")

    return errors


# ── 6. Serialisation ──────────────────────────────────────────────


def memory_record_to_dict(record: MemoryRecord) -> Dict:
    """
    Convert a MemoryRecord to a plain dictionary.

    Lists and dicts are deep-copied to avoid shared references.
    """
    d = asdict(record)
    # Ensure mutable containers are fresh copies
    d["shared_with"] = list(record.shared_with)
    d["tags"] = list(record.tags)
    d["metadata"] = dict(record.metadata)
    return d


def memory_record_from_dict(data: Dict) -> MemoryRecord:
    """
    Reconstruct a MemoryRecord from a dictionary.

    Missing keys fall back to the dataclass defaults.
    """
    # Extract only the fields that belong to MemoryRecord
    field_names = {f.name for f in MemoryRecord.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return MemoryRecord(**filtered)


def memory_record_to_jsonl(record: MemoryRecord) -> str:
    """
    Serialise a MemoryRecord to a single JSON line (JSONL).

    Uses ``ensure_ascii=False`` so CJK characters are preserved.
    """
    d = memory_record_to_dict(record)
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


def memory_record_from_jsonl(line: str) -> MemoryRecord:
    """
    Deserialise a JSONL line back into a MemoryRecord.
    """
    data = json.loads(line)
    return memory_record_from_dict(data)
