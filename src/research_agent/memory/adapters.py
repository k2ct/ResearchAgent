"""
Memory Integration Adapters.

Bridge between Phase-2 research modules (Claim Support, Paper Reading,
PPT Progress, Report Writer, Experiment Analysis) and Phase-3 Memory
System (Writer / Store).

Each ``save_*_result()`` function:
1. Extracts core content from a module's result dict.
2. Sets ``source_module``, ``source_path``, ``source_title``, ``metadata``.
3. Calls ``memory.writer.write_memory_from_source()`` (if available).
4. Returns a standardised result dict.

Graceful fallback: if the Memory Writer is unavailable, adapters return
``ok=False`` with a clear error message — they never raise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Helpers ───────────────────────────────────────────────────────────


def _safe_get(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely read a key from a dict, returning *default* on missing."""
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


def _join_nonempty(parts: List[str], sep: str = "\n\n") -> str:
    """Join non-empty strings with *sep*."""
    return sep.join(p for p in parts if p and p.strip())


def _extract_sources_text(sources: Any) -> str:
    """Convert a sources list/dict into a short text summary."""
    if not sources:
        return "(no sources)"
    if not isinstance(sources, list):
        return str(sources)[:300]

    lines: List[str] = []
    for s in sources[:8]:
        if isinstance(s, dict):
            path = s.get("path", "") or s.get("source_path", "")
            stype = s.get("source_type", "")
            title = s.get("title", "") or s.get("source_title", "")
            if title:
                lines.append(f"- {title} ({stype}) `{path}`")
            else:
                lines.append(f"- `{path}` ({stype})")
        elif isinstance(s, str):
            lines.append(f"- {s}")
    return "\n".join(lines) if lines else "(no sources)"


# ── Writer import (graceful fallback) ────────────────────────────────

_WRITER_AVAILABLE = False
_write_memory_fn = None

try:
    from research_agent.memory.writer import write_memory_from_source as _wmfs
    _write_memory_fn = _wmfs
    _WRITER_AVAILABLE = True
except ImportError:
    pass


def _try_write(
    content: str,
    source_module: str,
    source_path: str = "",
    source_title: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    auto_write: bool = True,
) -> Dict[str, Any]:
    """
    Build a record and optionally write it.

    When *auto_write=False*, only builds the record via
    ``build_memory_record_from_source()`` — does NOT touch the store.

    Returns a dict with keys: ok, memory_id, record, write_result, error.
    """
    result: Dict[str, Any] = {
        "ok": False,
        "memory_id": "",
        "record": None,
        "write_result": None,
        "error": "",
    }

    if not _WRITER_AVAILABLE:
        result["error"] = "Memory writer unavailable — writer.py not importable."
        return result

    try:
        # When auto_write=False, only build the record — don't call writer
        if not auto_write:
            try:
                from research_agent.memory.writer import build_memory_record_from_source
                record = build_memory_record_from_source(
                    content=content,
                    source_module=source_module,
                    source_path=source_path,
                    source_title=source_title,
                    metadata=metadata or {},
                )
                result["ok"] = True
                result["record"] = record
                result["write_result"] = None
                if isinstance(record, dict):
                    result["memory_id"] = record.get("memory_id", "") or record.get("record_id", "")
                return result
            except ImportError:
                result["error"] = "build_memory_record_from_source not available."
                return result

        # auto_write=True: use the full write pipeline
        write_result = _write_memory_fn(  # type: ignore[misc]
            content=content,
            source_module=source_module,
            source_path=source_path,
            source_title=source_title,
            metadata=metadata or {},
        )

        record = _safe_get(write_result, "record") or _safe_get(write_result, "build_result")

        result["ok"] = _safe_get(write_result, "ok", False)
        result["record"] = record
        result["write_result"] = write_result

        # Extract memory_id from record
        if isinstance(record, dict):
            result["memory_id"] = record.get("memory_id", "") or record.get("record_id", "")
        elif hasattr(record, "memory_id"):
            result["memory_id"] = record.memory_id or ""

        if not result["memory_id"]:
            wr = _safe_get(write_result, "write_result", {}) or {}
            result["memory_id"] = _safe_get(wr, "memory_id", "") or _safe_get(wr, "record_id", "")

        if not result["ok"]:
            result["error"] = _safe_get(write_result, "error", "Write failed")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 1. Claim Support adapter
# ═══════════════════════════════════════════════════════════════════════════

def save_claim_support_result(
    result: Dict[str, Any],
    auto_write: bool = True,
) -> Dict[str, Any]:
    """
    Save a ``generate_claim_support()`` result as a memory record.

    Expected keys in *result*: claim, claim_type, report, sources,
    evidence_count, grouped_evidence, used_llm, llm_error.
    """
    claim = _safe_get(result, "claim", "Unknown claim")
    claim_type = _safe_get(result, "claim_type", "general")
    report_text = _safe_get(result, "report", "")
    sources = _safe_get(result, "sources", [])
    evidence_count = _safe_get(result, "evidence_count", 0)
    used_llm = _safe_get(result, "used_llm", False)

    content = _join_nonempty([
        "# Claim Support Memory",
        "",
        "## Claim",
        claim,
        "",
        "## Claim Type",
        claim_type,
        "",
        "## Report",
        report_text[:3000] if report_text else "(no report)",
        "",
        "## Sources",
        _extract_sources_text(sources),
        "",
        f"## Metadata",
        f"- evidence_count: {evidence_count}",
        f"- used_llm: {used_llm}",
    ])

    base_result = _try_write(
        content=content,
        source_module="claim_support",
        source_path="",
        source_title=claim[:200],
        metadata={
            "claim": claim,
            "claim_type": claim_type,
            "evidence_count": evidence_count,
            "used_llm": used_llm,
            "sources": sources[:10] if isinstance(sources, list) else [],
        },
        auto_write=auto_write,
    )

    base_result["adapter"] = "claim_support"
    return base_result


# ═══════════════════════════════════════════════════════════════════════════
# 2. Paper Reading adapter
# ═══════════════════════════════════════════════════════════════════════════

def save_paper_reading_result(
    result: Dict[str, Any],
    auto_write: bool = True,
) -> Dict[str, Any]:
    """
    Save a ``read_paper()`` result as a memory record.
    """
    paper_path = _safe_get(result, "paper_path", "")
    metadata = _safe_get(result, "metadata", {}) or {}
    sections = _safe_get(result, "sections", {}) or {}
    reading_note = _safe_get(result, "reading_note", "")
    sources = _safe_get(result, "sources", [])
    used_llm = _safe_get(result, "used_llm", False)
    title = _safe_get(metadata, "title", "Unknown Paper")

    sections_list = list(sections.keys()) if isinstance(sections, dict) else []

    content = _join_nonempty([
        "# Paper Reading Memory",
        "",
        "## Basic Information",
        f"- Title: {title}",
        f"- Source: {paper_path}",
        f"- used_llm: {used_llm}",
        "",
        "## Reading Note",
        reading_note[:4000] if reading_note else "(no reading note)",
        "",
        "## Sections Detected",
        ", ".join(sections_list) if sections_list else "(none)",
        "",
        "## Sources",
        _extract_sources_text(sources),
    ])

    base_result = _try_write(
        content=content,
        source_module="paper_reading",
        source_path=paper_path,
        source_title=title,
        metadata={
            "paper_metadata": metadata,
            "sections_detected": sections_list,
            "used_llm": used_llm,
            "sources": sources[:10] if isinstance(sources, list) else [],
        },
        auto_write=auto_write,
    )

    base_result["adapter"] = "paper_reading"
    return base_result


# ═══════════════════════════════════════════════════════════════════════════
# 3. Progress Memory adapter
# ═══════════════════════════════════════════════════════════════════════════

def save_progress_memory_result(
    result: Dict[str, Any],
    auto_write: bool = True,
) -> Dict[str, Any]:
    """
    Save a ``generate_progress_memory()`` result as a memory record.
    """
    source_path = _safe_get(result, "source_path", "")
    metadata = _safe_get(result, "metadata", {}) or {}
    slides = _safe_get(result, "slides", []) or []
    topics = _safe_get(result, "topics", {}) or {}
    progress_memory = _safe_get(result, "progress_memory", "")
    memory_records = _safe_get(result, "memory_records", []) or []
    used_llm = _safe_get(result, "used_llm", False)
    title = _safe_get(metadata, "title", "Untitled Presentation")

    # Format topics
    topic_lines: List[str] = []
    for k, v in topics.items():
        if k == "keywords":
            continue
        topic_lines.append(f"- {k}: {', '.join(v[:5]) if isinstance(v, list) else str(v)[:200]}")

    content = _join_nonempty([
        "# Progress Memory",
        "",
        "## Source",
        source_path,
        "",
        f"## Slide Count",
        str(len(slides)),
        "",
        "## Progress Summary",
        progress_memory[:4000] if progress_memory else "(no progress memory)",
        "",
        "## Memory Records",
        "\n".join(f"- {r}" for r in memory_records[:10]) if memory_records else "(none)",
        "",
        "## Topics",
        "\n".join(topic_lines) if topic_lines else "(none)",
        "",
        f"## Metadata",
        f"- used_llm: {used_llm}",
    ])

    base_result = _try_write(
        content=content,
        source_module="ppt_progress",
        source_path=source_path,
        source_title=title,
        metadata={
            "slide_count": len(slides),
            "topics": {k: v for k, v in topics.items() if k != "keywords"},
            "memory_records": memory_records[:20] if isinstance(memory_records, list) else [],
            "used_llm": used_llm,
        },
        auto_write=auto_write,
    )

    base_result["adapter"] = "progress_memory"
    return base_result


# ═══════════════════════════════════════════════════════════════════════════
# 4. Report adapter
# ═══════════════════════════════════════════════════════════════════════════

def save_report_result(
    result: Dict[str, Any],
    auto_write: bool = True,
) -> Dict[str, Any]:
    """
    Save a report result as a memory record.

    Compatible with multiple report formats (report_writer, LLM report, etc.).
    """
    report_text = (
        _safe_get(result, "report_text")
        or _safe_get(result, "report")
        or _safe_get(result, "final_answer")
        or _safe_get(result, "answer")
        or ""
    )
    task_type = (
        _safe_get(result, "task_type")
        or _safe_get(result, "report_style")
        or "report"
    )
    sources = _safe_get(result, "sources", [])
    used_llm = _safe_get(result, "used_llm", False)
    evidence_status = _safe_get(result, "evidence_status", "unknown")

    content = _join_nonempty([
        "# Report Memory",
        "",
        "## Task Type",
        task_type,
        "",
        "## Report",
        report_text[:4000] if report_text else "(no report text)",
        "",
        "## Sources",
        _extract_sources_text(sources),
        "",
        "## Metadata",
        f"- used_llm: {used_llm}",
        f"- evidence_status: {evidence_status}",
    ])

    base_result = _try_write(
        content=content,
        source_module="report_writer",
        source_title=str(task_type),
        metadata={
            "task_type": task_type,
            "used_llm": used_llm,
            "evidence_status": evidence_status,
            "sources": sources[:10] if isinstance(sources, list) else [],
        },
        auto_write=auto_write,
    )

    base_result["adapter"] = "report_writer"
    return base_result


# ═══════════════════════════════════════════════════════════════════════════
# 5. Experiment Analysis adapter
# ═══════════════════════════════════════════════════════════════════════════

def save_experiment_analysis_result(
    result: Dict[str, Any],
    auto_write: bool = True,
) -> Dict[str, Any]:
    """
    Save an experiment / tool analysis result as a memory record.

    Compatible with CSV analyzer, JSONL analyzer, and general experiment results.
    """
    analysis = (
        _safe_get(result, "analysis")
        or _safe_get(result, "summary")
        or _safe_get(result, "report")
        or _safe_get(result, "final_answer")
        or ""
    )
    file_path = (
        _safe_get(result, "file_path")
        or _safe_get(result, "input_path")
        or _safe_get(result, "source_path")
        or ""
    )
    tool_used = (
        _safe_get(result, "tool_used")
        or _safe_get(result, "analyzer")
        or "experiment_analysis"
    )
    metrics = _safe_get(result, "metrics", None)
    evidence_status = _safe_get(result, "evidence_status", "unknown")
    sources = _safe_get(result, "sources", [])
    used_llm = _safe_get(result, "used_llm", False)

    # Format metrics if present
    metrics_text = ""
    if metrics and isinstance(metrics, dict):
        metrics_lines = [f"- {k}: {v}" for k, v in list(metrics.items())[:15]]
        metrics_text = "\n".join(metrics_lines)

    content = _join_nonempty([
        "# Experiment Analysis Memory",
        "",
        "## Tool",
        str(tool_used),
        "",
        "## File",
        str(file_path) if file_path else "(no file)",
        "",
        "## Analysis",
        analysis[:4000] if analysis else "(no analysis)",
        "",
        "## Metrics" if metrics_text else "",
        metrics_text if metrics_text else "",
        "",
        "## Sources",
        _extract_sources_text(sources),
        "",
        "## Metadata",
        f"- used_llm: {used_llm}",
        f"- evidence_status: {evidence_status}",
    ])

    base_result = _try_write(
        content=content,
        source_module="experiment_tool",
        source_path=str(file_path),
        source_title=str(tool_used),
        metadata={
            "tool_used": tool_used,
            "metrics": metrics if isinstance(metrics, dict) else {},
            "evidence_status": evidence_status,
            "sources": sources[:10] if isinstance(sources, list) else [],
        },
        auto_write=auto_write,
    )

    base_result["adapter"] = "experiment_analysis"
    return base_result


# ═══════════════════════════════════════════════════════════════════════════
# 6. Generic fallback adapter
# ═══════════════════════════════════════════════════════════════════════════

def save_generic_result(
    result: Dict[str, Any],
    source_module: str,
    source_title: str = "",
    auto_write: bool = True,
) -> Dict[str, Any]:
    """
    Save an arbitrary result dict as a memory record.

    Used when no specific adapter matches.
    """
    # Try to find meaningful content
    content_text = (
        _safe_get(result, "content")
        or _safe_get(result, "report")
        or _safe_get(result, "final_answer")
        or _safe_get(result, "answer")
        or _safe_get(result, "reading_note")
        or _safe_get(result, "progress_memory")
        or _safe_get(result, "analysis")
        or str(result)[:2000]
    )

    content = _join_nonempty([
        f"# Memory: {source_module}",
        "",
        str(content_text)[:4000],
    ])

    base_result = _try_write(
        content=content,
        source_module=source_module,
        source_title=source_title or source_module,
        metadata={"original_result_keys": list(result.keys())[:20] if isinstance(result, dict) else []},
        auto_write=auto_write,
    )

    base_result["adapter"] = "generic"
    return base_result


# ═══════════════════════════════════════════════════════════════════════════
# 7. Unified dispatcher
# ═══════════════════════════════════════════════════════════════════════════

_MODULE_DISPATCH: Dict[str, str] = {
    "claim_support": "claim_support",
    "paper_reading": "paper_reading",
    "paper": "paper_reading",
    "ppt_progress": "progress_memory",
    "progress_memory": "progress_memory",
    "ppt": "progress_memory",
    "report_writer": "report_writer",
    "report": "report_writer",
    "llm_report": "report_writer",
    "experiment_tool": "experiment_analysis",
    "experiment_analysis": "experiment_analysis",
    "experiment": "experiment_analysis",
    "tool": "experiment_analysis",
}

_ADAPTER_FUNCTIONS = {
    "claim_support": save_claim_support_result,
    "paper_reading": save_paper_reading_result,
    "progress_memory": save_progress_memory_result,
    "report_writer": save_report_result,
    "experiment_analysis": save_experiment_analysis_result,
}


def save_module_result(
    result: Dict[str, Any],
    module_name: str,
    auto_write: bool = True,
) -> Dict[str, Any]:
    """
    Save a module result to memory — dispatches to the correct adapter.

    Args:
        result: The result dict from any Phase-2 module.
        module_name: Canonical name (see _MODULE_DISPATCH for aliases).
        auto_write: If True, write through to the Memory Store.

    Returns:
        A standardised result dict with keys: ok, adapter, memory_id,
        record, write_result, error.
    """
    # Resolve alias
    canonical = _MODULE_DISPATCH.get(module_name, module_name)

    adapter_fn = _ADAPTER_FUNCTIONS.get(canonical)
    if adapter_fn is not None:
        return adapter_fn(result, auto_write=auto_write)

    # Generic fallback
    return save_generic_result(
        result,
        source_module=module_name,
        source_title=module_name,
        auto_write=auto_write,
    )
