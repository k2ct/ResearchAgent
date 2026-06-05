"""
Multi-Agent Trace & Evaluation.

Records orchestrator runs to a local JSONL trace file and provides
quality evaluation for each run.

Trace file: ``data/traces/multi_agent_traces.jsonl`` (git-ignored).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TRACES_DIR = _PROJECT_ROOT / "data" / "traces"
_TRACES_PATH = _TRACES_DIR / "multi_agent_traces.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════
# 1. MultiAgentTrace dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MultiAgentTrace:
    """A single orchestrator run trace."""

    trace_id: str = ""
    query: str = ""
    task_type: str = ""
    primary_agent: str = ""
    handoff_plan: Dict[str, Any] = field(default_factory=dict)
    handoff_results: List[Dict[str, Any]] = field(default_factory=list)
    memory_ids: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    final_answer_preview: str = ""
    memory_written: bool = False
    created_at: str = ""
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Create trace from orchestrator result
# ═══════════════════════════════════════════════════════════════════════════

def create_trace_from_orchestrator_result(
    result: Dict[str, Any],
    query: str = "",
    task_type: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a trace dict from a ``run_multi_agent_pipeline()`` result.

    Args:
        result: The orchestrator result dict.
        query: Original user query.
        task_type: Task type used.
        metadata: Optional extra metadata.

    Returns:
        A plain dict conforming to MultiAgentTrace.
    """
    # Collect errors
    errors: List[str] = []
    if result.get("memory_write_error"):
        errors.append(f"memory_write: {result['memory_write_error']}")

    handoff_results = result.get("handoff_results", [])
    for hr in handoff_results:
        if hr.get("status") == "failed":
            errors.append(
                f"handoff to {hr.get('to_agent', '?')} failed"
            )

    trace = MultiAgentTrace(
        trace_id=f"trace_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}",
        query=query,
        task_type=task_type,
        primary_agent=result.get("primary_agent", ""),
        handoff_plan=result.get("handoff_plan") or {},
        handoff_results=handoff_results,
        memory_ids=result.get("handoff_memory_ids", [])[:30],
        sources=result.get("handoff_sources", [])[:30],
        final_answer_preview=(result.get("combined_answer", "") or "")[:500],
        memory_written=result.get("memory_written", False),
        created_at=_utc_now(),
        errors=errors,
        metadata=metadata or {},
    )

    return asdict(trace)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Append / Load traces
# ═══════════════════════════════════════════════════════════════════════════

def append_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    """
    Append a trace record to the JSONL trace file.

    Returns::

        {"ok": bool, "path": str, "trace_id": str, "error": str}
    """
    try:
        _TRACES_DIR.mkdir(parents=True, exist_ok=True)
        with open(_TRACES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        return {
            "ok": True,
            "path": str(_TRACES_PATH),
            "trace_id": trace.get("trace_id", ""),
            "error": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "path": str(_TRACES_PATH),
            "trace_id": trace.get("trace_id", ""),
            "error": str(e),
        }


def load_traces(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Load the most recent *limit* traces from the JSONL file.
    """
    if not _TRACES_PATH.exists():
        return []

    records: List[Dict[str, Any]] = []
    with open(_TRACES_PATH, "r", encoding="utf-8") as f:
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
# 4. Summarise trace
# ═══════════════════════════════════════════════════════════════════════════

def summarize_trace(trace: Dict[str, Any]) -> str:
    """
    Produce a human-readable Markdown summary of a trace.

    Args:
        trace: A dict from ``create_trace_from_orchestrator_result()``
               or loaded from JSONL.

    Returns:
        Markdown string.
    """
    tid = trace.get("trace_id", "?")[:16]
    query = trace.get("query", "?")
    task_type = trace.get("task_type", "?")
    primary = trace.get("primary_agent", "?")
    plan = trace.get("handoff_plan", {}) or {}
    results = trace.get("handoff_results", []) or []
    errors = trace.get("errors", []) or []
    memory_ids = trace.get("memory_ids", []) or []
    sources = trace.get("sources", []) or []
    memory_written = trace.get("memory_written", False)
    answer = trace.get("final_answer_preview", "") or ""

    lines: List[str] = [
        "# Multi-Agent Trace",
        "",
        f"**Trace ID:** `{tid}...`",
        f"**Created:** {trace.get('created_at', '?')}",
        "",
        "## Query",
        "",
        query,
        "",
        "## Task",
        f"- **Type:** {task_type}",
        f"- **Primary Agent:** {primary}",
        "",
        "## Handoff Plan",
        f"- Plan ID: `{plan.get('plan_id', '?')[:16]}...`",
        f"- Handoffs: {plan.get('handoff_count', 0)}",
        f"- Targets: {', '.join(plan.get('targets', []))}",
        "",
        "## Agent Results",
        "",
    ]

    for i, r in enumerate(results, start=1):
        status = r.get("status", "?")
        agent = r.get("to_agent", "?")
        conf = r.get("confidence", 0.0)
        text = (r.get("result_text", "") or "")[:200]
        icon = "✅" if status == "completed" else ("❌" if status == "failed" else "⏳")

        lines.append(f"### {icon} {i}. {agent} ({status})")
        if status == "completed":
            lines.append(f"- Confidence: {conf:.2f}")
            if text:
                lines.append(f"- Output: {text}")
        elif status == "failed":
            lines.append(f"- Error: {r.get('error', 'no details')}")
        lines.append("")

    lines.append("## Memory Used")
    lines.append("")
    if memory_ids:
        for mid in memory_ids[:10]:
            lines.append(f"- `{mid[:20]}...`")
    else:
        lines.append("*No memories used.*")
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    if sources:
        for s in sources[:10]:
            path = s.get("path", "?")
            lines.append(f"- `{path}`")
    else:
        lines.append("*No sources.*")
    lines.append("")

    lines.append("## Errors")
    lines.append("")
    if errors:
        for e in errors:
            lines.append(f"- {e}")
    else:
        lines.append("*No errors.*")
    lines.append("")

    lines.append("## Metadata")
    lines.append(f"- Memory written: {memory_written}")
    lines.append(f"- Final answer length: {len(answer)} chars")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Evaluate handoff quality
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_handoff_quality(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the quality of a multi-agent orchestrator result.

    Labels: ``"strong"`` / ``"medium"`` / ``"weak"``.

    Rules:
    - completed >= 2 AND avg_confidence >= 0.7 → strong
    - completed >= 1 → medium
    - else → weak
    - Warnings for: no sources, no memory, any failures.
    """
    handoff_results = result.get("handoff_results", [])
    sources = result.get("handoff_sources", [])
    memory_ids = result.get("handoff_memory_ids", [])

    completed = [r for r in handoff_results if r.get("status") == "completed"]
    failed = [r for r in handoff_results if r.get("status") == "failed"]

    completed_count = len(completed)
    failed_count = len(failed)

    confidences = [r.get("confidence", 0.0) for r in completed]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    warnings: List[str] = []

    # Quality label
    if completed_count >= 2 and avg_confidence >= 0.7:
        label = "strong"
    elif completed_count >= 1:
        label = "medium"
    else:
        label = "weak"

    # Warnings
    if not sources:
        warnings.append("No sources provided by any agent.")
    if not memory_ids:
        warnings.append("No memory IDs referenced.")
    if failed_count > 0:
        warnings.append(f"{failed_count} handoff(s) failed.")
    if avg_confidence < 0.5 and completed_count > 0:
        warnings.append(f"Average confidence is low ({avg_confidence:.2f}).")

    return {
        "completed_count": completed_count,
        "failed_count": failed_count,
        "avg_confidence": round(avg_confidence, 3),
        "source_count": len(sources),
        "memory_count": len(memory_ids),
        "quality_label": label,
        "warnings": warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Convenience: trace + evaluate
# ═══════════════════════════════════════════════════════════════════════════

def trace_and_evaluate(
    orchestrator_result: Dict[str, Any],
    query: str = "",
    task_type: str = "",
    save_trace: bool = True,
) -> Dict[str, Any]:
    """
    Create a trace, evaluate quality, and optionally save to disk.

    Returns::

        {"trace": dict, "quality": dict, "saved": bool}
    """
    trace = create_trace_from_orchestrator_result(
        orchestrator_result, query=query, task_type=task_type
    )
    quality = evaluate_handoff_quality(orchestrator_result)

    saved = False
    if save_trace:
        result = append_trace(trace)
        saved = result.get("ok", False)

    return {"trace": trace, "quality": quality, "saved": saved}
