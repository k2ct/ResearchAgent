"""
Agent Handoff / Communication primitives.

Defines data structures and functions for:
1. Creating and validating handoff requests between specialised agents.
2. Building handoff plans from task type and query analysis.
3. Aggregating results from multiple agents.
4. Logging handoff events to a local JSONL file.

Does NOT define agent profiles — imports from ``.profiles`` when available.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Try to import profiles (may not exist yet)
# ═══════════════════════════════════════════════════════════════════════════

_PROFILES_AVAILABLE = False
_list_agent_profiles = None
_get_agent_profile = None

try:
    from .profiles import list_agent_profiles as _lap
    from .profiles import get_agent_profile as _gap
    _list_agent_profiles = _lap
    _get_agent_profile = _gap
    _PROFILES_AVAILABLE = True
except ImportError:
    pass

# Default known agent IDs (fallback when profiles.py is unavailable)
_DEFAULT_AGENT_IDS: List[str] = [
    "coordinator_agent",
    "paper_agent",
    "experiment_agent",
    "claim_agent",
    "progress_agent",
    "report_agent",
    "code_agent",
    "memory_agent",
    "general_agent",
]

# Known valid handoff targets per agent (fallback)
_DEFAULT_HANDOFF_TARGETS: Dict[str, List[str]] = {
    "coordinator_agent": [
        "paper_agent", "experiment_agent", "claim_agent",
        "progress_agent", "report_agent", "code_agent",
        "memory_agent", "general_agent",
    ],
    "paper_agent": ["claim_agent", "report_agent", "memory_agent", "coordinator_agent"],
    "experiment_agent": ["claim_agent", "report_agent", "progress_agent", "coordinator_agent"],
    "claim_agent": ["paper_agent", "experiment_agent", "report_agent", "memory_agent", "coordinator_agent"],
    "progress_agent": ["report_agent", "memory_agent", "coordinator_agent"],
    "report_agent": ["memory_agent", "coordinator_agent"],
    "code_agent": ["coordinator_agent"],
    "memory_agent": ["coordinator_agent", "paper_agent", "claim_agent", "report_agent", "progress_agent"],
    "general_agent": ["coordinator_agent"],
}


def _known_agent_ids() -> List[str]:
    """Return the list of known agent IDs."""
    if _PROFILES_AVAILABLE and _list_agent_profiles:
        try:
            profiles = _list_agent_profiles()
            return [p.get("agent_id", "") for p in profiles if p.get("agent_id")]
        except Exception:
            pass
    return list(_DEFAULT_AGENT_IDS)


def _known_handoff_targets(agent_id: str) -> List[str]:
    """Return the list of handoff targets for an agent."""
    if _PROFILES_AVAILABLE and _get_agent_profile:
        try:
            profile = _get_agent_profile(agent_id)
            if profile:
                targets = profile.get("handoff_targets", [])
                if targets:
                    return targets
        except Exception:
            pass
    return _DEFAULT_HANDOFF_TARGETS.get(agent_id, [])


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data structures
# ═══════════════════════════════════════════════════════════════════════════

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = "ho") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"


@dataclass
class HandoffRequest:
    """A request from one agent to another to perform a specific task."""

    handoff_id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    task: str = ""
    input_text: str = ""
    task_type: str = "general"
    memory_scope: List[str] = field(default_factory=list)
    required_memory_types: List[str] = field(default_factory=list)
    expected_output: str = ""
    priority: int = 3
    reason: str = ""
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandoffResult:
    """The result of a handoff request."""

    handoff_id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    status: str = "pending"      # pending | completed | failed | skipped
    result_text: str = ""
    confidence: float = 0.0
    sources: List[Dict[str, Any]] = field(default_factory=list)
    memory_ids: List[str] = field(default_factory=list)
    created_at: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandoffPlan:
    """A plan consisting of multiple handoff requests."""

    plan_id: str = ""
    root_task: str = ""
    root_query: str = ""
    coordinator_agent: str = "coordinator_agent"
    handoffs: List[HandoffRequest] = field(default_factory=list)
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Create handoff request
# ═══════════════════════════════════════════════════════════════════════════

def create_handoff_request(
    from_agent: str,
    to_agent: str,
    task: str,
    input_text: str,
    task_type: str = "general",
    memory_scope: Optional[List[str]] = None,
    required_memory_types: Optional[List[str]] = None,
    expected_output: str = "",
    priority: int = 3,
    reason: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> HandoffRequest:
    """
    Create a new ``HandoffRequest`` with auto-generated ID and timestamp.
    """
    return HandoffRequest(
        handoff_id=_new_id("ho"),
        from_agent=from_agent,
        to_agent=to_agent,
        task=task,
        input_text=input_text,
        task_type=task_type,
        memory_scope=memory_scope or [],
        required_memory_types=required_memory_types or [],
        expected_output=expected_output,
        priority=priority,
        reason=reason,
        created_at=_utc_now(),
        metadata=metadata or {},
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_handoff_request(request: HandoffRequest) -> List[str]:
    """
    Validate a handoff request. Returns a list of error messages.

    An empty list means the request is valid.

    Checks:
    - from_agent and to_agent are non-empty
    - from_agent ≠ to_agent
    - task and input_text are non-empty
    - priority is in [1, 5]
    - to_agent is a known agent (warning, not error)
    - to_agent is in from_agent's handoff_targets (warning, not error)
    """
    errors: List[str] = []

    if not request.from_agent or not request.from_agent.strip():
        errors.append("from_agent is empty")
    if not request.to_agent or not request.to_agent.strip():
        errors.append("to_agent is empty")
    if request.from_agent.strip() == request.to_agent.strip():
        errors.append("from_agent must differ from to_agent")
    if not request.task or not request.task.strip():
        errors.append("task is empty")
    if not request.input_text or not request.input_text.strip():
        errors.append("input_text is empty")
    if not (1 <= request.priority <= 5):
        errors.append(f"priority must be 1-5, got {request.priority}")

    # Profile-based checks (warnings only — don't block handoff)
    known_ids = _known_agent_ids()
    if known_ids and request.to_agent not in known_ids:
        errors.append(
            f"WARNING: to_agent '{request.to_agent}' is not a known agent. "
            f"Known agents: {known_ids}"
        )

    targets = _known_handoff_targets(request.from_agent)
    if targets and request.to_agent not in targets:
        errors.append(
            f"WARNING: '{request.to_agent}' is not in {request.from_agent}'s "
            f"usual handoff targets: {targets}"
        )

    return errors


# ═══════════════════════════════════════════════════════════════════════════
# 4. Serialisation round-trips
# ═══════════════════════════════════════════════════════════════════════════

def handoff_request_to_dict(req: HandoffRequest) -> Dict[str, Any]:
    """Convert a HandoffRequest to a plain dict."""
    return asdict(req)


def handoff_request_from_dict(data: Dict[str, Any]) -> HandoffRequest:
    """Reconstruct a HandoffRequest from a dict."""
    return HandoffRequest(
        handoff_id=data.get("handoff_id", ""),
        from_agent=data.get("from_agent", ""),
        to_agent=data.get("to_agent", ""),
        task=data.get("task", ""),
        input_text=data.get("input_text", ""),
        task_type=data.get("task_type", "general"),
        memory_scope=data.get("memory_scope", []),
        required_memory_types=data.get("required_memory_types", []),
        expected_output=data.get("expected_output", ""),
        priority=data.get("priority", 3),
        reason=data.get("reason", ""),
        created_at=data.get("created_at", ""),
        metadata=data.get("metadata", {}),
    )


def handoff_result_to_dict(res: HandoffResult) -> Dict[str, Any]:
    """Convert a HandoffResult to a plain dict."""
    return asdict(res)


def handoff_result_from_dict(data: Dict[str, Any]) -> HandoffResult:
    """Reconstruct a HandoffResult from a dict."""
    return HandoffResult(
        handoff_id=data.get("handoff_id", ""),
        from_agent=data.get("from_agent", ""),
        to_agent=data.get("to_agent", ""),
        status=data.get("status", "pending"),
        result_text=data.get("result_text", ""),
        confidence=float(data.get("confidence", 0.0)),
        sources=data.get("sources", []),
        memory_ids=data.get("memory_ids", []),
        created_at=data.get("created_at", ""),
        error=data.get("error", ""),
        metadata=data.get("metadata", {}),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Build handoff plan from query and task_type
# ═══════════════════════════════════════════════════════════════════════════

# Keywords that suggest multi-agent plans
_MULTI_AGENT_KEYWORDS = [
    "汇报", "组会", "大纲", "PPT", "总结", "进展",
    "论文支持", "找论文", "证据", "论证",
    "实验分析", "安排", "今天应该",
]


def build_handoff_plan(
    root_query: str,
    task_type: str,
    coordinator_agent: str = "coordinator_agent",
) -> HandoffPlan:
    """
    Build a ``HandoffPlan`` from a root query and task type.

    Uses keyword heuristics to decide which agents should be involved.
    """
    query_lower = root_query.lower()
    handoffs: List[HandoffRequest] = []
    coordinator = coordinator_agent

    # ── paper_question ──────────────────────────────────────────
    if task_type == "paper_question":
        handoffs.append(create_handoff_request(
            from_agent=coordinator,
            to_agent="paper_agent",
            task="Find relevant paper notes and background for the question",
            input_text=root_query,
            task_type="paper_question",
            required_memory_types=["paper_note", "claim_support"],
            expected_output="Paper reading notes and related work analysis",
            priority=4,
            reason=f"Task type is paper_question",
        ))

    # ── experiment_analysis ─────────────────────────────────────
    elif task_type == "experiment_analysis":
        handoffs.append(create_handoff_request(
            from_agent=coordinator,
            to_agent="experiment_agent",
            task="Analyse experimental data and provide metrics summary",
            input_text=root_query,
            task_type="experiment_analysis",
            required_memory_types=["experiment_result"],
            expected_output="Experiment metrics and analysis",
            priority=4,
            reason=f"Task type is experiment_analysis",
        ))

    # ── claim_support ───────────────────────────────────────────
    elif task_type == "claim_support":
        handoffs.append(create_handoff_request(
            from_agent=coordinator,
            to_agent="claim_agent",
            task="Find evidence supporting or refuting the claim",
            input_text=root_query,
            task_type="claim_support",
            required_memory_types=["claim_support", "paper_note"],
            expected_output="Claim support report with evidence",
            priority=4,
            reason=f"Task type is claim_support",
        ))
        # Also ask paper_agent and experiment_agent for supporting context
        handoffs.append(create_handoff_request(
            from_agent=coordinator,
            to_agent="paper_agent",
            task="Search paper notes for related theoretical background",
            input_text=root_query,
            task_type="paper_question",
            required_memory_types=["paper_note"],
            expected_output="Related paper background",
            priority=3,
            reason="Claim support benefits from paper background",
        ))
        handoffs.append(create_handoff_request(
            from_agent=coordinator,
            to_agent="experiment_agent",
            task="Check for experimental evidence related to the claim",
            input_text=root_query,
            task_type="experiment_analysis",
            required_memory_types=["experiment_result"],
            expected_output="Experimental evidence if available",
            priority=3,
            reason="Claim support benefits from empirical evidence",
        ))

    # ── report_generation ───────────────────────────────────────
    elif task_type == "report_generation":
        handoffs.append(create_handoff_request(
            from_agent=coordinator,
            to_agent="report_agent",
            task="Generate a structured research report or presentation",
            input_text=root_query,
            task_type="report_generation",
            required_memory_types=["report_summary", "progress_update"],
            expected_output="Structured report or presentation outline",
            priority=4,
            reason=f"Task type is report_generation",
        ))
        handoffs.append(create_handoff_request(
            from_agent=coordinator,
            to_agent="progress_agent",
            task="Summarise recent research progress for the report",
            input_text=root_query,
            task_type="progress_update",
            required_memory_types=["progress_update", "meeting_note"],
            expected_output="Recent progress summary",
            priority=3,
            reason="Reports benefit from recent progress context",
        ))
        handoffs.append(create_handoff_request(
            from_agent=coordinator,
            to_agent="paper_agent",
            task="Find paper background for the report",
            input_text=root_query,
            task_type="paper_question",
            required_memory_types=["paper_note", "claim_support"],
            expected_output="Paper background for report",
            priority=2,
            reason="Reports may need paper context",
        ))

    # ── general / complex multi-agent queries ───────────────────
    elif any(kw in query_lower for kw in _MULTI_AGENT_KEYWORDS):
        # Multi-agent: progress + paper + report
        handoffs.append(create_handoff_request(
            from_agent=coordinator,
            to_agent="progress_agent",
            task="Summarise recent research progress and findings",
            input_text=root_query,
            task_type="progress_update",
            required_memory_types=["progress_update", "meeting_note", "todo"],
            expected_output="Recent progress summary",
            priority=4,
            reason="Multi-agent query detected: needs progress summary",
        ))
        handoffs.append(create_handoff_request(
            from_agent=coordinator,
            to_agent="paper_agent",
            task="Find relevant paper support and background",
            input_text=root_query,
            task_type="paper_question",
            required_memory_types=["paper_note", "claim_support"],
            expected_output="Paper support and background",
            priority=4,
            reason="Multi-agent query detected: needs paper support",
        ))
        handoffs.append(create_handoff_request(
            from_agent=coordinator,
            to_agent="report_agent",
            task="Generate presentation outline or report structure",
            input_text=root_query,
            task_type="report_generation",
            required_memory_types=["progress_update", "paper_note", "report_summary"],
            expected_output="Report or presentation outline",
            priority=4,
            reason="Multi-agent query detected: needs generated output",
        ))

    # ── fallback: general_agent ─────────────────────────────────
    else:
        handoffs.append(create_handoff_request(
            from_agent=coordinator,
            to_agent="general_agent",
            task="Handle the general query",
            input_text=root_query,
            task_type="general",
            expected_output="General answer or advice",
            priority=3,
            reason="No specific task_type matched — routing to general_agent",
        ))

    return HandoffPlan(
        plan_id=_new_id("plan"),
        root_task=task_type,
        root_query=root_query,
        coordinator_agent=coordinator,
        handoffs=handoffs,
        created_at=_utc_now(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Aggregate results
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_handoff_results(
    plan: HandoffPlan,
    results: List[HandoffResult],
) -> Dict[str, Any]:
    """
    Aggregate results from multiple handoffs into a combined answer.

    Returns a dict with keys: plan_id, root_task, completed, failed,
    combined_answer, agent_outputs, sources, memory_ids, confidence.
    """
    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status == "skipped"]
    pending = [r for r in results if r.status == "pending"]

    # Group by to_agent
    by_agent: Dict[str, List[HandoffResult]] = {}
    for r in completed:
        by_agent.setdefault(r.to_agent, []).append(r)

    # Build combined answer
    lines: List[str] = ["# Multi-Agent Result", ""]

    agent_labels = {
        "paper_agent": "Paper Agent",
        "experiment_agent": "Experiment Agent",
        "claim_agent": "Claim Agent",
        "progress_agent": "Progress Agent",
        "report_agent": "Report Agent",
        "code_agent": "Code Agent",
        "memory_agent": "Memory Agent",
        "general_agent": "General Agent",
        "coordinator_agent": "Coordinator",
    }

    for agent_id in sorted(by_agent.keys()):
        agent_results = by_agent[agent_id]
        label = agent_labels.get(agent_id, agent_id)
        lines.append(f"## {label}")
        lines.append("")
        for i, r in enumerate(agent_results, start=1):
            if r.result_text:
                lines.append(f"### Output {i}")
                lines.append(r.result_text[:2000])
                lines.append("")
            if r.sources:
                lines.append(f"**Sources ({len(r.sources)}):**")
                for s in r.sources[:5]:
                    path = s.get("path", "?")
                    lines.append(f"- `{path}`")
                lines.append("")

    # Failed / Skipped
    if failed or skipped:
        lines.append("## Failed / Skipped")
        lines.append("")
        for r in failed:
            lines.append(f"- **{r.to_agent}** (failed): {r.error or 'no error details'}")
        for r in skipped:
            lines.append(f"- **{r.to_agent}** (skipped)")
        lines.append("")

    if pending:
        lines.append(f"*{len(pending)} handoff(s) still pending.*")
        lines.append("")

    # Collect and deduplicate sources
    all_sources: List[Dict[str, Any]] = []
    seen_sources: set = set()
    for r in completed:
        for s in r.sources:
            key = s.get("path", id(s))
            if key not in seen_sources:
                seen_sources.add(key)
                all_sources.append(s)

    # Collect and deduplicate memory_ids
    all_memory_ids: List[str] = []
    seen_mids: set = set()
    for r in completed:
        for mid in r.memory_ids:
            if mid not in seen_mids:
                seen_mids.add(mid)
                all_memory_ids.append(mid)

    # Average confidence
    avg_confidence = (
        sum(r.confidence for r in completed) / len(completed)
        if completed else 0.0
    )

    return {
        "plan_id": plan.plan_id,
        "root_task": plan.root_task,
        "root_query": plan.root_query,
        "completed": len(completed),
        "failed": len(failed),
        "skipped": len(skipped),
        "pending": len(pending),
        "combined_answer": "\n".join(lines),
        "agent_outputs": [
            {
                "to_agent": r.to_agent,
                "status": r.status,
                "result_text": r.result_text[:500],
                "confidence": r.confidence,
            }
            for r in results
        ],
        "sources": all_sources,
        "memory_ids": all_memory_ids,
        "confidence": round(avg_confidence, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. Handoff log (simple JSONL, not Memory Store)
# ═══════════════════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_HANDOFF_LOG_PATH = _PROJECT_ROOT / "data" / "memory" / "handoff_log.jsonl"


def append_handoff_log(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Append a handoff event to the local JSONL log.

    Args:
        event: A dict describing the event (request, result, plan, etc.).

    Returns::

        {"ok": bool, "path": str, "error": str}
    """
    try:
        event.setdefault("logged_at", _utc_now())
        _HANDOFF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_HANDOFF_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return {"ok": True, "path": str(_HANDOFF_LOG_PATH), "error": ""}
    except Exception as e:
        return {"ok": False, "path": str(_HANDOFF_LOG_PATH), "error": str(e)}


def load_handoff_log(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Load the last *limit* entries from the handoff log.

    Returns an empty list if the log file does not exist.
    """
    if not _HANDOFF_LOG_PATH.exists():
        return []

    records: List[Dict[str, Any]] = []
    with open(_HANDOFF_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records[-limit:]
