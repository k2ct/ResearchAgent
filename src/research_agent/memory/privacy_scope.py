"""
Memory Privacy & Scope — access control for research agent memories.

Scope levels (per schema):
- ``private`` — only the ``owner_agent`` may access.
- ``shared``  — ``owner_agent`` + agents listed in ``shared_with``.
- ``global``  — all agents may access.

This module provides:
1. Rule-based scope classification from memory metadata.
2. Access-check predicates (can agent X read record Y?).
3. Scope getter / setter / lister on loaded records.
4. Write-time scope validation and auto-correction.
5. Filter helpers to produce agent-accessible subsets.

Design: pure functions operating on MemoryRecord dataclass instances
(or plain dicts that conform to the same shape).  No side-effects
except when explicitly calling ``apply_scope_update_to_store()``,
which delegates to ``memory.store.append_memory()``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Lazy imports — store / schema may be called at runtime
_STORE_AVAILABLE = False
try:
    from research_agent.memory.store import append_memory  # type: ignore[import-untyped]
    _STORE_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Field-access helper (works on dataclass AND dict records)
# ═══════════════════════════════════════════════════════════════════════════

def _rf(record: Any, key: str, default: Any = None) -> Any:
    """Read a field from a MemoryRecord dataclass or a plain dict."""
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _wf(record: Any, key: str, value: Any) -> Any:
    """Write a field on a MemoryRecord dataclass or plain dict (mutates)."""
    if isinstance(record, dict):
        record[key] = value
    else:
        setattr(record, key, value)
    return record


# ═══════════════════════════════════════════════════════════════════════════
# 1. Scope classification rules
# ═══════════════════════════════════════════════════════════════════════════

# memory_type → default (scope, shared_with)
_TYPE_SCOPE_RULES: Dict[str, Tuple[str, List[str]]] = {
    # ── global scope (all agents) ──
    "research_direction": ("global", []),
    "project_decision": ("global", []),
    # ── shared scope ──
    "claim_support": ("shared", ["paper_agent", "report_agent", "progress_agent"]),
    "paper_note": ("shared", ["claim_agent", "report_agent"]),
    "progress_update": ("shared", ["experiment_agent", "report_agent", "coordinator"]),
    "experiment_result": ("shared", ["claim_agent", "report_agent", "progress_agent"]),
    "report_summary": ("shared", ["progress_agent", "coordinator"]),
    # ── private scope (default) ──
    "todo": ("private", []),
    "code_note": ("private", []),
    "user_preference": ("private", []),
    "general_note": ("private", []),
}

# Tags that promote scope to shared/global
_TAG_SCOPE_PROMOTIONS: Dict[str, str] = {
    "architecture": "shared",
    "pipeline": "shared",
    "workflow": "shared",
    "integration": "shared",
    "cross-agent": "shared",
    "shared_knowledge": "global",
    "common_reference": "global",
}

# memory_level + specific conditions → scope elevation
_LEVEL_SCOPE_OVERRIDES = {
    # long_term + research-like types → global
    ("long_term", "research_direction"): "global",
    ("long_term", "project_decision"): "global",
    # mid_term + shared types → shared
    ("mid_term", "progress_update"): "shared",
    ("mid_term", "experiment_result"): "shared",
}


def classify_scope(
    memory_type: str,
    memory_level: str = "mid_term",
    tags: Optional[List[str]] = None,
    owner_agent: str = "",
) -> Dict[str, Any]:
    """
    Determine the appropriate memory scope from metadata.

    Returns::

        {
            "memory_scope": "private" | "shared" | "global",
            "shared_with": [str, ...],
            "reason": str,
        }
    """
    tags_lower = [t.lower() for t in (tags or [])]

    # 1. Level × type override (strongest)
    override_key = (memory_level, memory_type)
    if override_key in _LEVEL_SCOPE_OVERRIDES:
        scope = _LEVEL_SCOPE_OVERRIDES[override_key]
        shared_with = _TYPE_SCOPE_RULES.get(memory_type, ("private", []))[1]
        return {
            "memory_scope": scope,
            "shared_with": shared_with,
            "reason": f"Level+type override: {memory_level} {memory_type} → {scope}",
        }

    # 2. Tag-based promotion (overrides type default)
    for tag in tags_lower:
        if tag in _TAG_SCOPE_PROMOTIONS:
            promoted = _TAG_SCOPE_PROMOTIONS[tag]
            return {
                "memory_scope": promoted,
                "shared_with": [],
                "reason": f"Tag promotion: '{tag}' → {promoted}",
            }

    # 3. Type-based rules
    if memory_type in _TYPE_SCOPE_RULES:
        scope, shared_with = _TYPE_SCOPE_RULES[memory_type]
        return {
            "memory_scope": scope,
            "shared_with": list(shared_with),
            "reason": f"Type rule: {memory_type} → {scope}",
        }

    # 4. Default
    return {
        "memory_scope": "private",
        "shared_with": [],
        "reason": "Default → private scope",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Access check
# ═══════════════════════════════════════════════════════════════════════════

# Special agents with universal access
_UNIVERSAL_AGENTS: Set[str] = {"coordinator", "admin"}


def check_access(
    record: Any,
    requesting_agent: str,
) -> bool:
    """
    Return True if *requesting_agent* is permitted to read *record*.

    Rules (first match wins):
    1. Universal agents (coordinator / admin) → always True.
    2. ``global`` scope → True.
    3. Owner agent → True.
    4. ``shared`` scope AND requesting_agent in ``shared_with`` → True.
    5. Otherwise → False.
    """
    agent = requesting_agent.strip().lower()
    if agent in _UNIVERSAL_AGENTS:
        return True

    scope = _rf(record, "memory_scope", "private")
    if scope == "global":
        return True

    owner = (_rf(record, "owner_agent", "") or "").strip().lower()
    if owner == agent:
        return True

    if scope == "shared":
        shared_with: List[str] = _rf(record, "shared_with", []) or []
        shared_lower = [a.strip().lower() for a in shared_with]
        if agent in shared_lower:
            return True

    return False


def filter_accessible(
    records: List[Any],
    requesting_agent: str,
) -> List[Any]:
    """Return the subset of *records* accessible to *requesting_agent*."""
    return [r for r in records if check_access(r, requesting_agent)]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Scope getter / setter / lister
# ═══════════════════════════════════════════════════════════════════════════

VALID_SCOPES = {"private", "shared", "global"}


def get_scope(record: Any) -> Dict[str, Any]:
    """Extract scope metadata from a record."""
    return {
        "memory_scope": _rf(record, "memory_scope", "private"),
        "shared_with": list(_rf(record, "shared_with", []) or []),
        "owner_agent": _rf(record, "owner_agent", ""),
        "memory_id": _rf(record, "memory_id", ""),
    }


def set_scope(
    record: Any,
    scope: str,
    shared_with: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Update the scope of *record* (mutates and returns a result dict).

    Returns::

        {"ok": bool, "record": record, "previous_scope": str, "error": str}
    """
    scope = scope.strip().lower()
    if scope not in VALID_SCOPES:
        return {
            "ok": False,
            "record": record,
            "previous_scope": _rf(record, "memory_scope", "private"),
            "error": f"Invalid scope '{scope}'. Must be one of {VALID_SCOPES}.",
        }

    previous = _rf(record, "memory_scope", "private")
    _wf(record, "memory_scope", scope)

    if scope == "shared" and shared_with is not None:
        _wf(record, "shared_with", list(shared_with))
    elif scope != "shared":
        _wf(record, "shared_with", [])

    return {
        "ok": True,
        "record": record,
        "previous_scope": previous,
        "error": "",
    }


def list_shared_with(record: Any) -> List[str]:
    """Return the list of agents this record is shared with."""
    scope = _rf(record, "memory_scope", "private")
    if scope == "global":
        return ["*"]  # all agents
    if scope == "shared":
        return list(_rf(record, "shared_with", []) or [])
    return []  # private


# ═══════════════════════════════════════════════════════════════════════════
# 4. Write-time validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_scope_on_write(record: Any) -> Dict[str, Any]:
    """
    Validate and auto-correct scope before writing to the store.

    Checks:
    - scope is one of {private, shared, global}
    - shared scope must have non-empty shared_with (warns, doesn't block)
    - private scope should have empty shared_with (auto-corrects)
    - global scope auto-clears shared_with (all agents have access)

    Returns::

        {"valid": bool, "record": record, "warnings": [str, ...], "changes": [str, ...]}
    """
    warnings: List[str] = []
    changes: List[str] = []

    scope = _rf(record, "memory_scope", "private")
    shared_with: List[str] = list(_rf(record, "shared_with", []) or [])

    # Validate scope value
    if scope not in VALID_SCOPES:
        old = scope
        # Try to re-classify
        classification = classify_scope(
            memory_type=_rf(record, "memory_type", "general_note"),
            memory_level=_rf(record, "memory_level", "mid_term"),
            tags=list(_rf(record, "tags", []) or []),
            owner_agent=_rf(record, "owner_agent", ""),
        )
        _wf(record, "memory_scope", classification["memory_scope"])
        _wf(record, "shared_with", classification["shared_with"])
        changes.append(f"Invalid scope '{old}' → auto-classified as '{classification['memory_scope']}' "
                       f"({classification['reason']})")
        scope = classification["memory_scope"]
        shared_with = classification["shared_with"]

    # Global → clear shared_with
    if scope == "global" and shared_with:
        _wf(record, "shared_with", [])
        changes.append("global scope: cleared shared_with (all agents have access)")
        shared_with = []

    # Private → ensure empty shared_with
    if scope == "private" and shared_with:
        _wf(record, "shared_with", [])
        changes.append("private scope: cleared shared_with (only owner may access)")
        shared_with = []

    # Shared → warn if empty
    if scope == "shared" and not shared_with:
        warnings.append("shared scope with empty shared_with — no other agents can access")

    return {
        "valid": True,
        "record": record,
        "warnings": warnings,
        "changes": changes,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Bulk scope operations
# ═══════════════════════════════════════════════════════════════════════════

def auto_classify_record_scope(record: Any) -> Dict[str, Any]:
    """
    Re-classify the scope of *record* using :func:`classify_scope`
    and apply the result.  Returns the same shape as :func:`set_scope`.
    """
    classification = classify_scope(
        memory_type=_rf(record, "memory_type", "general_note"),
        memory_level=_rf(record, "memory_level", "mid_term"),
        tags=list(_rf(record, "tags", []) or []),
        owner_agent=_rf(record, "owner_agent", ""),
    )
    return set_scope(
        record,
        scope=classification["memory_scope"],
        shared_with=classification["shared_with"],
    )


def auto_classify_all(records: List[Any]) -> List[Dict[str, Any]]:
    """Run :func:`auto_classify_record_scope` on every record."""
    return [auto_classify_record_scope(r) for r in records]


# ═══════════════════════════════════════════════════════════════════════════
# 6. Store integration
# ═══════════════════════════════════════════════════════════════════════════

def apply_scope_update_to_store(
    record: Any,
    new_scope: str,
    shared_with: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Update a record's scope and persist via the Memory Store.

    Because the store uses append-only JSONL, this appends an updated
    copy.  The old record remains in the file but can be filtered out
    by ``status`` or ``updated_at``.

    Returns::

        {"ok": bool, "record": updated_record, "write_result": dict, "error": str}
    """
    result = set_scope(deepcopy(record) if not isinstance(record, dict) else dict(record),
                       scope=new_scope, shared_with=shared_with)
    if not result["ok"]:
        return {**result, "write_result": {}, "error": result["error"]}

    updated = result["record"]

    # Validate before writing
    validation = validate_scope_on_write(updated)

    if _STORE_AVAILABLE:
        write_result = append_memory(updated)
        return {
            "ok": write_result.get("ok", False),
            "record": updated,
            "write_result": write_result,
            "error": "",
            "validation": validation,
        }
    else:
        return {
            "ok": True,
            "record": updated,
            "write_result": {"written": False, "reason": "store.py not available"},
            "error": "",
            "validation": validation,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 7. Scope summary / audit
# ═══════════════════════════════════════════════════════════════════════════

def scope_summary(records: List[Any]) -> Dict[str, Any]:
    """
    Produce an audit summary of scope distribution across *records*.

    Returns::

        {
            "total": int,
            "by_scope": {"private": int, "shared": int, "global": int},
            "by_owner": {agent: count, ...},
            "shared_with_counts": {agent: count, ...},
            "private_count": int,
            "global_count": int,
        }
    """
    by_scope: Dict[str, int] = {"private": 0, "shared": 0, "global": 0}
    by_owner: Dict[str, int] = {}
    shared_with_counts: Dict[str, int] = {}

    for r in records:
        scope = _rf(r, "memory_scope", "private")
        by_scope[scope] = by_scope.get(scope, 0) + 1

        owner = _rf(r, "owner_agent", "unknown")
        by_owner[owner] = by_owner.get(owner, 0) + 1

        for agent in (_rf(r, "shared_with", []) or []):
            shared_with_counts[agent] = shared_with_counts.get(agent, 0) + 1

    return {
        "total": len(records),
        "by_scope": by_scope,
        "by_owner": dict(sorted(by_owner.items(), key=lambda x: -x[1])),
        "shared_with_counts": dict(sorted(shared_with_counts.items(), key=lambda x: -x[1])),
        "private_count": by_scope.get("private", 0),
        "global_count": by_scope.get("global", 0),
    }
