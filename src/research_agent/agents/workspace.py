"""
Shared Workspace with Section-level Permissions v1.

Multiple specialist agents collaborate on a single Markdown workspace document.
All agents have full read access.  Each agent may only write to sections
they own or have explicit write permission for.  Unauthorised writes are
automatically converted to suggestions.  The Coordinator has universal
write access and can approve suggestions.

Design:
- Workspace is a task-scoped draft — NOT long-term memory.
- Stable final summaries may be persisted to Memory Store separately.
- All writes are logged to a JSONL patch log.
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Constants ──────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_DIR = _PROJECT_ROOT / "data" / "workspaces"

_COORDINATOR = "coordinator_agent"

_DEFAULT_SECTIONS: List[Dict[str, Any]] = [
    {"section_id": "paper_evidence",     "title": "Paper Evidence",
     "owner_agent": "paper_agent",       "write_permission": ["paper_agent", _COORDINATOR]},
    {"section_id": "experiment_evidence","title": "Experiment Evidence",
     "owner_agent": "experiment_agent",  "write_permission": ["experiment_agent", _COORDINATOR]},
    {"section_id": "claim_support",      "title": "Claim Support",
     "owner_agent": "claim_agent",       "write_permission": ["claim_agent", _COORDINATOR]},
    {"section_id": "progress_summary",   "title": "Progress Summary",
     "owner_agent": "progress_agent",    "write_permission": ["progress_agent", _COORDINATOR]},
    {"section_id": "report_draft",       "title": "Report Draft",
     "owner_agent": "report_agent",      "write_permission": ["report_agent", _COORDINATOR]},
    {"section_id": "implementation_notes","title": "Implementation Notes",
     "owner_agent": "code_agent",        "write_permission": ["code_agent", _COORDINATOR]},
    {"section_id": "memory_notes",       "title": "Memory Notes",
     "owner_agent": "memory_agent",      "write_permission": ["memory_agent", _COORDINATOR]},
    {"section_id": "final_summary",      "title": "Coordinator Final Summary",
     "owner_agent": _COORDINATOR,        "write_permission": [_COORDINATOR]},
]

_AGENT_DEFAULT_SECTION: Dict[str, str] = {s["owner_agent"]: s["section_id"] for s in _DEFAULT_SECTIONS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = "ws") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WorkspaceSection:
    section_id: str = ""
    title: str = ""
    owner_agent: str = ""
    write_permission: List[str] = field(default_factory=list)
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkspaceSection":
        return cls(
            section_id=d.get("section_id", ""),
            title=d.get("title", ""),
            owner_agent=d.get("owner_agent", ""),
            write_permission=list(d.get("write_permission", [])),
            content=d.get("content", ""),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class WorkspaceDocument:
    workspace_id: str = ""
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    status: str = "active"
    coordinator: str = _COORDINATOR
    allowed_agents: List[str] = field(default_factory=list)
    sections: List[WorkspaceSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sections"] = [s.to_dict() for s in self.sections]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkspaceDocument":
        return cls(
            workspace_id=d.get("workspace_id", ""),
            title=d.get("title", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            status=d.get("status", "active"),
            coordinator=d.get("coordinator", _COORDINATOR),
            allowed_agents=list(d.get("allowed_agents", [])),
            sections=[WorkspaceSection.from_dict(s) for s in d.get("sections", [])],
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class WorkspacePatch:
    patch_id: str = ""
    workspace_id: str = ""
    agent_id: str = ""
    target_section: str = ""
    operation: str = "replace_section"   # replace_section | append_section | suggest_edit
    content: str = ""
    reason: str = ""
    status: str = "pending"              # applied | rejected | suggestion | pending
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkspacePatch":
        return cls(
            patch_id=d.get("patch_id", ""),
            workspace_id=d.get("workspace_id", ""),
            agent_id=d.get("agent_id", ""),
            target_section=d.get("target_section", ""),
            operation=d.get("operation", "replace_section"),
            content=d.get("content", ""),
            reason=d.get("reason", ""),
            status=d.get("status", "pending"),
            created_at=d.get("created_at", ""),
            metadata=dict(d.get("metadata", {})),
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Default workspace builder
# ═══════════════════════════════════════════════════════════════════════════

def build_default_workspace(
    title: str,
    coordinator: str = _COORDINATOR,
    allowed_agents: Optional[List[str]] = None,
) -> WorkspaceDocument:
    now = _utc_now()
    agents = allowed_agents or [s["owner_agent"] for s in _DEFAULT_SECTIONS]

    sections = [
        WorkspaceSection(
            section_id=s["section_id"],
            title=s["title"],
            owner_agent=s["owner_agent"],
            write_permission=list(s["write_permission"]),
            content=f"*({s['title']} — awaiting input)*",
        )
        for s in _DEFAULT_SECTIONS
    ]

    return WorkspaceDocument(
        workspace_id=_new_id("ws"),
        title=title,
        created_at=now,
        updated_at=now,
        coordinator=coordinator,
        allowed_agents=agents,
        sections=sections,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Markdown serialisation
# ═══════════════════════════════════════════════════════════════════════════

def workspace_to_markdown(ws: WorkspaceDocument) -> str:
    lines: List[str] = [
        "---",
        f"workspace_id: {ws.workspace_id}",
        f"title: {ws.title}",
        f"created_at: {ws.created_at}",
        f"updated_at: {ws.updated_at}",
        f"status: {ws.status}",
        f"coordinator: {ws.coordinator}",
        "allowed_agents:",
    ]
    for a in ws.allowed_agents:
        lines.append(f"  - {a}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {ws.title}")
    lines.append("")

    for s in ws.sections:
        lines.append(f"## {s.title}")
        lines.append(f"<!-- section_id: {s.section_id} -->")
        lines.append(f"<!-- owner: {s.owner_agent} -->")
        lines.append(f"<!-- write_permission: {', '.join(s.write_permission)} -->")
        lines.append("")
        lines.append(s.content.strip())
        lines.append("")

    return "\n".join(lines)


def workspace_from_markdown(text: str) -> WorkspaceDocument:
    ws = WorkspaceDocument()
    lines = text.splitlines()
    in_front_matter = False
    in_allowed = False
    current_section: Optional[WorkspaceSection] = None
    content_lines: List[str] = []

    for line in lines:
        stripped = line.strip()

        # Front matter
        if stripped == "---":
            if not in_front_matter:
                in_front_matter = True
                continue
            else:
                in_front_matter = False
                in_allowed = False
                continue

        if in_front_matter:
            if stripped.startswith("workspace_id:"):
                ws.workspace_id = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("title:"):
                ws.title = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("created_at:"):
                ws.created_at = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("updated_at:"):
                ws.updated_at = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("status:"):
                ws.status = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("coordinator:"):
                ws.coordinator = stripped.split(":", 1)[1].strip()
            elif stripped == "allowed_agents:":
                in_allowed = True
            elif in_allowed and stripped.startswith("- "):
                ws.allowed_agents.append(stripped[2:].strip())
            continue

        # Section comment markers (fill metadata on current section, don't create new)
        if stripped.startswith("<!-- section_id:"):
            if current_section:
                current_section.section_id = stripped.replace("<!-- section_id:", "").replace("-->", "").strip()
            continue
        if stripped.startswith("<!-- owner:"):
            if current_section:
                current_section.owner_agent = stripped.replace("<!-- owner:", "").replace("-->", "").strip()
            continue
        if stripped.startswith("<!-- write_permission:"):
            if current_section:
                raw = stripped.replace("<!-- write_permission:", "").replace("-->", "").strip()
                current_section.write_permission = [a.strip() for a in raw.split(",") if a.strip()]
            continue

        # H1: title line (skip)
        if stripped.startswith("# ") and not current_section:
            continue

        # H2: section title
        if stripped.startswith("## ") and not stripped.startswith("### "):
            # Flush previous section
            if current_section:
                current_section.content = "\n".join(content_lines).strip()
                ws.sections.append(current_section)
            # Start new section (title only, metadata filled by comments)
            sec_title = stripped[3:].strip()
            current_section = WorkspaceSection(title=sec_title)
            content_lines = []
            continue

        # Regular content
        if current_section is not None:
            content_lines.append(line)

    # Flush last section
    if current_section is not None:
        current_section.content = "\n".join(content_lines).strip()
        ws.sections.append(current_section)

    return ws


# ═══════════════════════════════════════════════════════════════════════════
# 4. File I/O
# ═══════════════════════════════════════════════════════════════════════════

def ensure_workspace_dir() -> Path:
    _WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return _WORKSPACE_DIR


def save_workspace(ws: WorkspaceDocument, path: Optional[Path] = None) -> Dict[str, Any]:
    try:
        ws.updated_at = _utc_now()
        target = path or (_WORKSPACE_DIR / f"{ws.workspace_id}.md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(workspace_to_markdown(ws), encoding="utf-8")
        return {"ok": True, "path": str(target), "error": ""}
    except Exception as e:
        return {"ok": False, "path": "", "error": str(e)}


def load_workspace(path: Path) -> WorkspaceDocument:
    text = path.read_text(encoding="utf-8")
    return workspace_from_markdown(text)


def _patch_log_path(workspace_id: str) -> Path:
    return _WORKSPACE_DIR / f"{workspace_id}_patch_log.jsonl"


def append_patch_log(patch: WorkspacePatch) -> Dict[str, Any]:
    try:
        log_path = _patch_log_path(patch.workspace_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(patch.to_dict(), ensure_ascii=False) + "\n")
        return {"ok": True, "path": str(log_path), "error": ""}
    except Exception as e:
        return {"ok": False, "path": "", "error": str(e)}


def load_patch_log(workspace_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    log_path = _patch_log_path(workspace_id)
    if not log_path.exists():
        return []
    records: List[Dict] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-limit:]


# ═══════════════════════════════════════════════════════════════════════════
# 5. Permissions
# ═══════════════════════════════════════════════════════════════════════════

def can_agent_write_section(ws: WorkspaceDocument, agent_id: str, section_id: str) -> bool:
    if agent_id == _COORDINATOR:
        return True
    for s in ws.sections:
        if s.section_id == section_id:
            return agent_id in s.write_permission
    return False


def get_readable_workspace_for_agent(ws: WorkspaceDocument, agent_id: str) -> str:
    """v1: all agents have full read access."""
    return workspace_to_markdown(ws)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Patch creation & application
# ═══════════════════════════════════════════════════════════════════════════

def create_workspace_patch(
    workspace_id: str,
    agent_id: str,
    target_section: str,
    content: str,
    operation: str = "replace_section",
    reason: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> WorkspacePatch:
    return WorkspacePatch(
        patch_id=_new_id("patch"),
        workspace_id=workspace_id,
        agent_id=agent_id,
        target_section=target_section,
        operation=operation,
        content=content,
        reason=reason,
        status="pending",
        created_at=_utc_now(),
        metadata=metadata or {},
    )


def apply_workspace_patch(
    ws: WorkspaceDocument,
    patch: WorkspacePatch,
    log: bool = True,
) -> Dict[str, Any]:
    """Apply a patch to a workspace.  Unauthorised writes become suggestions."""
    # Find section
    section = None
    for s in ws.sections:
        if s.section_id == patch.target_section:
            section = s
            break

    if section is None:
        patch.status = "rejected"
        if log:
            append_patch_log(patch)
        return {"ok": False, "applied": False, "suggestion": False,
                "patch": patch.to_dict(), "workspace": ws.to_dict(),
                "error": f"Section '{patch.target_section}' not found"}

    # Check permission
    if not can_agent_write_section(ws, patch.agent_id, patch.target_section):
        # Unauthorised → suggestion
        patch.operation = "suggest_edit"
        patch.status = "suggestion"
        if log:
            append_patch_log(patch)
        return {"ok": True, "applied": False, "suggestion": True,
                "patch": patch.to_dict(), "workspace": ws.to_dict(),
                "error": f"Agent '{patch.agent_id}' lacks write permission for '{patch.target_section}' — saved as suggestion"}

    # Apply
    if patch.operation in ("replace_section", "suggest_edit"):
        section.content = patch.content
    elif patch.operation == "append_section":
        section.content = section.content.rstrip() + "\n\n" + patch.content
    else:
        section.content = patch.content  # default: replace

    patch.status = "applied"
    ws.updated_at = _utc_now()

    if log:
        append_patch_log(patch)

    return {"ok": True, "applied": True, "suggestion": False,
            "patch": patch.to_dict(), "workspace": ws.to_dict(),
            "error": ""}


def approve_suggestion_patch(
    ws: WorkspaceDocument,
    patch: WorkspacePatch,
    coordinator_agent: str = _COORDINATOR,
) -> Dict[str, Any]:
    """Approve a suggestion patch — coordinator only."""
    if patch.agent_id != coordinator_agent and patch.status != "suggestion":
        return {"ok": False, "applied": False, "suggestion": True,
                "patch": patch.to_dict(), "workspace": ws.to_dict(),
                "error": "Only suggestion patches can be approved"}

    # Force-apply as coordinator
    patch.agent_id = coordinator_agent
    patch.operation = "replace_section"
    return apply_workspace_patch(ws, patch, log=True)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Handoff integration
# ═══════════════════════════════════════════════════════════════════════════

def default_section_for_agent(agent_id: str) -> str:
    return _AGENT_DEFAULT_SECTION.get(agent_id, "final_summary")


def create_patch_from_handoff_result(
    ws: WorkspaceDocument,
    agent_id: str,
    target_section: str,
    result_text: str,
    handoff_id: str = "",
) -> WorkspacePatch:
    section_id = target_section or default_section_for_agent(agent_id)
    return create_workspace_patch(
        workspace_id=ws.workspace_id,
        agent_id=agent_id,
        target_section=section_id,
        content=result_text,
        operation="replace_section",
        reason=f"Handoff result from {agent_id}",
        metadata={"handoff_id": handoff_id},
    )


# ═══════════════════════════════════════════════════════════════════════════
# 8. Memory integration
# ═══════════════════════════════════════════════════════════════════════════

def save_workspace_summary_to_memory(
    ws: WorkspaceDocument,
    save_memory: bool = False,
) -> Dict[str, Any]:
    if not save_memory:
        return {"ok": True, "written": False, "memory_id": "", "reason": "save_memory=False"}

    try:
        from research_agent.memory.writer import write_memory_from_source

        final_section = next((s for s in ws.sections if s.section_id == "final_summary"), None)
        final_text = final_section.content if final_section else "(no final summary)"

        section_summaries = "\n".join(
            f"- **{s.title}** ({s.owner_agent}): {s.content[:150]}..."
            for s in ws.sections if s.content.strip()
        )

        content = (
            f"# Workspace Summary: {ws.title}\n\n"
            f"## Final Summary\n{final_text}\n\n"
            f"## Section Summaries\n{section_summaries}\n"
        )

        result = write_memory_from_source(
            content=content,
            source_module="system",
            source_title=ws.title,
            metadata={
                "workspace_id": ws.workspace_id,
                "section_count": len(ws.sections),
                "allowed_agents": ws.allowed_agents,
            },
        )
        mem_id = ""
        if result.get("ok"):
            rec = result.get("record", {})
            if isinstance(rec, dict):
                mem_id = rec.get("memory_id", "")
            elif hasattr(rec, "memory_id"):
                mem_id = rec.memory_id or ""

        return {"ok": result.get("ok", False), "written": result.get("ok", False),
                "memory_id": mem_id, "error": result.get("error", "")}

    except Exception as e:
        return {"ok": False, "written": False, "memory_id": "",
                "error": f"{type(e).__name__}: {e}"}
