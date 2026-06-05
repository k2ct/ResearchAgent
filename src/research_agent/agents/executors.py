"""
Specialist Agent Executors — real module dispatch for multi-agent handoffs.

Each ``_execute_*_agent()`` function receives a ``HandoffRequest`` and
returns a ``HandoffResult``.  All executors are wrapped in try/except —
they never raise.

Used by ``orchestrator._execute_handoffs()`` when real execution is
available; falls back to simulated execution otherwise.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Lazy import — only needed if this module is actually called
_HANDOFF_OK = False
HandoffRequest = None  # type: ignore
HandoffResult = None   # type: ignore

try:
    from research_agent.agents.handoff import HandoffRequest as _HR
    from research_agent.agents.handoff import HandoffResult as _HRes
    HandoffRequest = _HR
    HandoffResult = _HRes
    _HANDOFF_OK = True
except ImportError:
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rf(rec: Any, key: str, default: Any = None) -> Any:
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


# ═══════════════════════════════════════════════════════════════════════════
# Main dispatch
# ═══════════════════════════════════════════════════════════════════════════

def execute_handoff_request(
    request: Any,
    rag_docs: Optional[List[Dict[str, Any]]] = None,
    memories: Optional[List[Dict[str, Any]]] = None,
    use_llm: bool = False,
    save_memory: bool = False,
) -> Any:
    """
    Execute a single handoff request against the real specialist module.

    Args:
        request: A ``HandoffRequest`` dataclass.
        rag_docs: Pre-retrieved RAG documents (list of dicts).
        memories: Pre-retrieved memory records.
        use_llm: Whether to enable LLM-based enhancement.
        save_memory: Whether to write results back to the Memory Store.

    Returns:
        A ``HandoffResult`` with status=completed or status=failed.
    """
    if not _HANDOFF_OK:
        return _failed_result(request, "Handoff module unavailable")

    agent = getattr(request, "to_agent", "")

    dispatcher = {
        "paper_agent": _execute_paper_agent,
        "claim_agent": _execute_claim_agent,
        "progress_agent": _execute_progress_agent,
        "report_agent": _execute_report_agent,
        "experiment_agent": _execute_experiment_agent,
        "memory_agent": _execute_memory_agent,
        "code_agent": _execute_code_agent,
        "general_agent": _execute_general_agent,
        "coordinator_agent": _execute_general_agent,
    }

    executor = dispatcher.get(agent, _execute_general_agent)

    try:
        return executor(request, rag_docs, memories, use_llm, save_memory)
    except Exception as e:
        return _failed_result(request, f"{type(e).__name__}: {e}")


def _failed_result(request: Any, error: str) -> Any:
    return HandoffResult(
        handoff_id=getattr(request, "handoff_id", ""),
        from_agent=getattr(request, "from_agent", ""),
        to_agent=getattr(request, "to_agent", ""),
        status="failed",
        result_text="",
        confidence=0.0,
        sources=[],
        memory_ids=[],
        created_at=_utc_now(),
        error=error,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Paper Agent
# ═══════════════════════════════════════════════════════════════════════════

def _execute_paper_agent(
    request, rag_docs, memories, use_llm, save_memory,
):
    input_text = getattr(request, "input_text", "")
    sources: List[Dict] = []
    memory_ids: List[str] = []

    # Try RAG retrieval for paper_note
    rag_text = ""
    if rag_docs:
        paper_docs = [
            d for d in rag_docs
            if _rf(d, "metadata", {}).get("source_type") == "paper_note"
        ]
        if paper_docs:
            rag_text = "\n".join(
                f"- {_rf(d, 'metadata', {}).get('title', '?')} "
                f"(`{_rf(d, 'metadata', {}).get('path', '?')}`)"
                for d in paper_docs[:5]
            )
            sources.extend([
                {"path": _rf(d, "metadata", {}).get("path", ""),
                 "source_type": "paper_note"}
                for d in paper_docs[:5]
            ])

    # Try memory
    mem_text = ""
    if memories:
        paper_mems = [
            m for m in memories
            if _rf(m, "memory_type", "") in ("paper_note", "claim_support", "research_direction")
        ]
        if paper_mems:
            mem_text = "\n".join(
                f"- [{_rf(m, 'memory_id', '?')[:12]}] {_rf(m, 'summary', '')[:120]}"
                for m in paper_mems[:5]
            )
            memory_ids.extend(_rf(m, "memory_id", "") for m in paper_mems[:5])

    lines = [
        "## Paper Agent Output",
        "",
        f"**Task**: {getattr(request, 'task', input_text[:100])}",
        "",
    ]
    if rag_text:
        lines.append(f"**RAG Paper Documents**:\n{rag_text}\n")
    if mem_text:
        lines.append(f"**Paper Memory**:\n{mem_text}\n")
    if not rag_text and not mem_text:
        lines.append(
            "*(No paper-specific RAG documents or memory records found. "
            "Try adding paper notes to data/papers/ or ingesting papers.)*\n"
        )

    return HandoffResult(
        handoff_id=getattr(request, "handoff_id", ""),
        from_agent=getattr(request, "from_agent", ""),
        to_agent="paper_agent",
        status="completed",
        result_text="\n".join(lines),
        confidence=0.8 if (rag_text or mem_text) else 0.3,
        sources=sources,
        memory_ids=memory_ids,
        created_at=_utc_now(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Claim Agent
# ═══════════════════════════════════════════════════════════════════════════

def _execute_claim_agent(
    request, rag_docs, memories, use_llm, save_memory,
):
    input_text = getattr(request, "input_text", "")
    sources: List[Dict] = []
    memory_ids: List[str] = []

    # Try real claim support module
    claim_result = None
    try:
        from research_agent.claim.claim_support import generate_claim_support
        claim_result = generate_claim_support(
            claim=input_text[:500],
            top_k_per_query=3,
            use_llm=use_llm,
        )
    except Exception:
        pass

    if claim_result:
        report = claim_result.get("report", "")[:3000]
        sources = claim_result.get("sources", [])[:10]
        confidence = 0.8
        lines = [
            "## Claim Agent Output",
            "",
            f"**Claim Type**: {claim_result.get('claim_type', '?')}",
            f"**Evidence Count**: {claim_result.get('evidence_count', 0)}",
            "",
            report,
        ]
    else:
        # Fallback to memory-based claim support
        claim_mems = []
        if memories:
            claim_mems = [
                m for m in memories
                if _rf(m, "memory_type", "") in ("claim_support", "paper_note", "research_direction")
            ]
        if claim_mems:
            mem_text = "\n".join(
                f"- [{_rf(m, 'memory_id', '?')[:12]}] {_rf(m, 'summary', '')[:150]}"
                for m in claim_mems[:5]
            )
            memory_ids.extend(_rf(m, "memory_id", "") for m in claim_mems[:5])
            confidence = 0.5
        else:
            mem_text = "*(No claim-support memory found.)*"
            confidence = 0.3

        lines = [
            "## Claim Agent Output",
            "",
            f"**Claim**: {input_text[:200]}",
            "",
            f"**Memory-based Evidence**:\n{mem_text}",
        ]

    # Write to memory if requested
    if save_memory and claim_result:
        try:
            from research_agent.memory.adapters import save_claim_support_result
            mr = save_claim_support_result(claim_result, auto_write=True)
            if mr.get("ok"):
                m_id = mr.get("memory_id", "")
                if m_id:
                    memory_ids.append(m_id)
        except Exception:
            pass

    return HandoffResult(
        handoff_id=getattr(request, "handoff_id", ""),
        from_agent=getattr(request, "from_agent", ""),
        to_agent="claim_agent",
        status="completed",
        result_text="\n".join(lines),
        confidence=confidence,
        sources=sources,
        memory_ids=memory_ids,
        created_at=_utc_now(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Progress Agent
# ═══════════════════════════════════════════════════════════════════════════

def _execute_progress_agent(
    request, rag_docs, memories, use_llm, save_memory,
):
    input_text = getattr(request, "input_text", "")
    sources: List[Dict] = []
    memory_ids: List[str] = []

    # Try continuity suggestions from memory_aware_agent
    suggestions = []
    try:
        from research_agent.memory.memory_aware_agent import get_continuity_suggestions
        suggestions = get_continuity_suggestions(
            task_type="general",
            max_suggestions=5,
        )
    except Exception:
        pass

    # Memory filter
    progress_mems = []
    if memories:
        progress_mems = [
            m for m in memories
            if _rf(m, "memory_type", "") in ("progress_update", "meeting_note", "todo", "project_decision")
        ]

    lines = ["## Progress Agent Output", ""]

    if suggestions:
        lines.append("**Continuity Suggestions (from Memory):**")
        for s in suggestions:
            mid = _rf(s, "memory_id", "?")[:12]
            summary = _rf(s, "summary", "")[:120]
            lines.append(f"- [{mid}] {summary}")
            memory_ids.append(_rf(s, "memory_id", ""))
        lines.append("")
        confidence = 0.7
    elif progress_mems:
        lines.append("**Progress Memory:**")
        for m in progress_mems[:5]:
            lines.append(f"- [{_rf(m, 'memory_id', '?')[:12]}] {_rf(m, 'summary', '')[:120]}")
            memory_ids.append(_rf(m, "memory_id", ""))
        lines.append("")
        confidence = 0.5
    else:
        lines.append("*(No progress memory records found.)*")
        lines.append("")
        confidence = 0.3

    lines.append(f"**Next Steps**: Review recent experiment results and update progress memory.")

    return HandoffResult(
        handoff_id=getattr(request, "handoff_id", ""),
        from_agent=getattr(request, "from_agent", ""),
        to_agent="progress_agent",
        status="completed",
        result_text="\n".join(lines),
        confidence=confidence,
        sources=sources,
        memory_ids=memory_ids,
        created_at=_utc_now(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Report Agent
# ═══════════════════════════════════════════════════════════════════════════

def _execute_report_agent(
    request, rag_docs, memories, use_llm, save_memory,
):
    input_text = getattr(request, "input_text", "")
    sources: List[Dict] = []
    memory_ids: List[str] = []

    # Memory-backed report generation
    report_mems = []
    if memories:
        report_mems = [
            m for m in memories
            if _rf(m, "memory_type", "") in (
                "report_summary", "progress_update", "paper_note",
                "experiment_result", "claim_support",
            )
        ]

    lines = ["## Report Agent Output", "", f"**Query**: {input_text[:200]}", ""]

    if report_mems:
        lines.append("**Relevant Context (from Memory):**")
        for m in report_mems[:5]:
            lines.append(f"- [{_rf(m, 'memory_id', '?')[:12]}] {_rf(m, 'summary', '')[:120]}")
            memory_ids.append(_rf(m, "memory_id", ""))
        lines.append("")
        confidence = 0.6

    # Try LLM report writer
    report_text = ""
    if use_llm:
        try:
            from research_agent.report.llm_report_writer import generate_report_with_llm
            llm_result = generate_report_with_llm(
                query=input_text,
                retrieved_docs=rag_docs or [],
                tool_result_text="",
            )
            if llm_result.get("ok"):
                report_text = llm_result.get("report_text", "")[:3000]
                confidence = 0.8
        except Exception:
            pass

    if report_text:
        lines.append(f"**LLM-Assisted Report**:\n\n{report_text}")
    else:
        lines.append("**Template Report Draft**:")
        lines.append(
            "Based on available memory and RAG context, the following structure is suggested:\n"
            "1. Research Background\n2. Recent Progress\n3. Key Findings\n"
            "4. Limitations\n5. Next Steps"
        )

    # RAG sources
    if rag_docs:
        for d in rag_docs[:5]:
            path = _rf(d, "metadata", {}).get("path", "")
            if path:
                sources.append({"path": path, "source_type": _rf(d, "metadata", {}).get("source_type", "")})

    return HandoffResult(
        handoff_id=getattr(request, "handoff_id", ""),
        from_agent=getattr(request, "from_agent", ""),
        to_agent="report_agent",
        status="completed",
        result_text="\n".join(lines),
        confidence=confidence if 'confidence' in dir() else 0.5,
        sources=sources,
        memory_ids=memory_ids,
        created_at=_utc_now(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Experiment Agent
# ═══════════════════════════════════════════════════════════════════════════

def _execute_experiment_agent(
    request, rag_docs, memories, use_llm, save_memory,
):
    input_text = getattr(request, "input_text", "")
    sources: List[Dict] = []
    memory_ids: List[str] = []

    # Try CSV/JSONL tool
    tool_text = ""
    tool_used = "none"
    if ".csv" in input_text.lower() or ".jsonl" in input_text.lower():
        try:
            from research_agent.tools.tool_router import run_tool_from_query
            tool_output = run_tool_from_query(input_text)
            tool_used = tool_output.get("tool_used", "none")
            if tool_used != "none":
                tool_text = tool_output.get("formatted_text", "")[:3000]
        except Exception:
            tool_text = "(Tool execution failed — falling back to memory/rag)"

    # Experiment memory
    exp_mems = []
    if memories:
        exp_mems = [
            m for m in memories
            if _rf(m, "memory_type", "") in ("experiment_result", "progress_update", "issue")
        ]

    lines = ["## Experiment Agent Output", ""]

    if tool_used != "none" and tool_text:
        lines.append(f"**Tool Used**: {tool_used}")
        lines.append(f"\n{tool_text}\n")
        confidence = 0.9
    elif exp_mems:
        lines.append("**Experiment Memory**:")
        for m in exp_mems[:5]:
            lines.append(f"- [{_rf(m, 'memory_id', '?')[:12]}] {_rf(m, 'summary', '')[:120]}")
            memory_ids.append(_rf(m, "memory_id", ""))
        lines.append("")
        confidence = 0.6
    else:
        lines.append(
            "*(No experiment data files detected and no experiment memory found. "
            "Provide a CSV/JSONL path or ingest experiment results.)*"
        )
        confidence = 0.3

    # RAG experiment docs
    if rag_docs:
        exp_docs = [
            d for d in rag_docs
            if _rf(d, "metadata", {}).get("source_type") == "experiment_doc"
        ]
        for d in exp_docs[:5]:
            sources.append({
                "path": _rf(d, "metadata", {}).get("path", ""),
                "source_type": "experiment_doc",
            })

    return HandoffResult(
        handoff_id=getattr(request, "handoff_id", ""),
        from_agent=getattr(request, "from_agent", ""),
        to_agent="experiment_agent",
        status="completed",
        result_text="\n".join(lines),
        confidence=confidence if 'confidence' in dir() else 0.3,
        sources=sources,
        memory_ids=memory_ids,
        created_at=_utc_now(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Memory Agent
# ═══════════════════════════════════════════════════════════════════════════

def _execute_memory_agent(
    request, rag_docs, memories, use_llm, save_memory,
):
    input_text = getattr(request, "input_text", "")
    sources: List[Dict] = []
    memory_ids: List[str] = []

    lines = ["## Memory Agent Output", ""]

    # Memory retrieval
    total_count = 0
    try:
        from research_agent.memory.store import load_memories
        all_records = load_memories()
        total_count = len(all_records)
    except Exception:
        pass

    lines.append(f"**Total Memory Records**: {total_count}")

    # Query matching memories
    query_mems: List[Dict] = []
    if memories:
        scored = []
        for m in memories:
            score = 0
            content = (_rf(m, "content", "") + " " + _rf(m, "summary", "")).lower()
            for word in input_text.lower().split()[:10]:
                if len(word) >= 2 and word in content:
                    score += 1
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        query_mems = [m for _, m in scored[:5]]

    if query_mems:
        lines.append(f"\n**Relevant Memories ({len(query_mems)} found)**:")
        for m in query_mems:
            mid = _rf(m, "memory_id", "?")[:12]
            mtype = _rf(m, "memory_type", "?")
            summary = _rf(m, "summary", "")[:120]
            lines.append(f"- [{mid}] ({mtype}) {summary}")
            memory_ids.append(_rf(m, "memory_id", ""))
    else:
        lines.append("\n*(No memories matched the query.)*")

    # Consolidation dry-run preview
    lines.append("\n**Consolidation Preview (dry-run)**:")
    try:
        from research_agent.memory.consolidation import preview_consolidation_plan
        preview = preview_consolidation_plan()
        lines.append(f"- Duplicates to merge: {preview.get('duplicates_to_merge', 0)}")
        lines.append(f"- Long memories to compress: {preview.get('long_memories_to_compress', 0)}")
        lines.append(f"- Memories to expire: {preview.get('memories_to_expire', 0)}")
        lines.append(f"- Stage summary records: {preview.get('stage_summary_records', 0)}")
        lines.append("*(dry-run only — no changes applied)*")
    except Exception:
        lines.append("*(Consolidation preview unavailable)*")

    return HandoffResult(
        handoff_id=getattr(request, "handoff_id", ""),
        from_agent=getattr(request, "from_agent", ""),
        to_agent="memory_agent",
        status="completed",
        result_text="\n".join(lines),
        confidence=0.7 if query_mems else 0.4,
        sources=sources,
        memory_ids=memory_ids,
        created_at=_utc_now(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Code Agent
# ═══════════════════════════════════════════════════════════════════════════

def _execute_code_agent(
    request, rag_docs, memories, use_llm, save_memory,
):
    input_text = getattr(request, "input_text", "")
    memory_ids: List[str] = []

    # Memory lookup for code notes
    code_mems = []
    if memories:
        code_mems = [
            m for m in memories
            if _rf(m, "memory_type", "") in ("code_note", "project_decision", "issue", "todo")
        ]

    lines = ["## Code Agent Output", "", f"**Issue**: {input_text[:200]}", ""]

    if code_mems:
        lines.append("**Related Code Notes**:")
        for m in code_mems[:5]:
            lines.append(f"- [{_rf(m, 'memory_id', '?')[:12]}] {_rf(m, 'summary', '')[:120]}")
            memory_ids.append(_rf(m, "memory_id", ""))
        lines.append("")

    lines.append(
        "**Suggestions**:\n"
        "1. Check that all dependencies are installed (`pip install -r requirements.txt`).\n"
        "2. Verify Python path and virtual environment are correctly configured.\n"
        "3. Review the traceback for the specific module and line number.\n"
        "4. For detailed debugging, run the script with `python -X dev` or use a debugger."
    )

    return HandoffResult(
        handoff_id=getattr(request, "handoff_id", ""),
        from_agent=getattr(request, "from_agent", ""),
        to_agent="code_agent",
        status="completed",
        result_text="\n".join(lines),
        confidence=0.5 if code_mems else 0.3,
        sources=[],
        memory_ids=memory_ids,
        created_at=_utc_now(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# General Agent (fallback)
# ═══════════════════════════════════════════════════════════════════════════

def _execute_general_agent(
    request, rag_docs, memories, use_llm, save_memory,
):
    input_text = getattr(request, "input_text", "")

    lines = [
        "## General Agent Output",
        "",
        f"**Query**: {input_text[:300]}",
        "",
        "This is a general-purpose response. For specialised tasks, "
        "please route to a domain-specific agent (paper_agent, "
        "experiment_agent, claim_agent, etc.).",
    ]

    return HandoffResult(
        handoff_id=getattr(request, "handoff_id", ""),
        from_agent=getattr(request, "from_agent", ""),
        to_agent="general_agent",
        status="completed",
        result_text="\n".join(lines),
        confidence=0.4,
        sources=[],
        memory_ids=[],
        created_at=_utc_now(),
    )
