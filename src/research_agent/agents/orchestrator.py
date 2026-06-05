"""
Multi-Agent Orchestrator — ties agent profiles, handoff, RAG, memory,
and memory adapters into a single dispatch-and-aggregate pipeline.

Used by the LangGraph ``multi_agent_router_node`` when
``ENABLE_MULTI_AGENT=true``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set

# ── imports (graceful fallbacks) ──────────────────────────────────────
_PROFILES_OK = False
_HANDOFF_OK = False
_MEMORY_OK = False
_ADAPTERS_OK = False
_EXECUTORS_OK = False

try:
    from research_agent.agents.profiles import select_agent_for_task, get_agent_profile  # type: ignore
    _PROFILES_OK = True
except ImportError:
    pass

try:
    from research_agent.agents.handoff import (  # type: ignore
        build_handoff_plan, aggregate_handoff_results,
        create_handoff_request, HandoffPlan, HandoffResult,
        append_handoff_log,
    )
    _HANDOFF_OK = True
except ImportError:
    pass

try:
    from research_agent.memory.memory_aware_agent import (  # type: ignore
        retrieve_memories_for_query, load_memories_for_agent,
    )
    _MEMORY_OK = True
except ImportError:
    pass

try:
    from research_agent.memory.adapters import save_module_result  # type: ignore
    _ADAPTERS_OK = True
except ImportError:
    pass

try:
    from research_agent.agents.executors import execute_handoff_request  # type: ignore
    _EXECUTORS_OK = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Main orchestration function
# ═══════════════════════════════════════════════════════════════════════════

def run_multi_agent_pipeline(
    query: str,
    task_type: str,
    *,
    rag_docs: Optional[List[Dict[str, Any]]] = None,
    rag_sources: Optional[List[Dict[str, Any]]] = None,
    memory_context: str = "",
    retrieved_memories: Optional[List[Dict[str, Any]]] = None,
    auto_write_memory: bool = True,
) -> Dict[str, Any]:
    """
    Execute the full multi-agent pipeline for a user query.

    1. Select primary agent via ``select_agent_for_task()``.
    2. Build a ``HandoffPlan`` with sub-tasks.
    3. Simulate handoff execution (uses RAG docs + memory context).
    4. Aggregate results into a combined answer.
    5. Optionally write to Memory Store via adapters.

    Returns a dict suitable for merging into ``AgentState``.
    """
    result: Dict[str, Any] = {
        "primary_agent": "",
        "handoff_plan": None,
        "handoff_results": [],
        "combined_answer": "",
        "handoff_sources": [],
        "handoff_memory_ids": [],
        "handoff_count": 0,
        "handoff_summary": "",
        "memory_written": False,
        "memory_write_error": "",
    }

    # ── 1. Select primary agent ────────────────────────────────────
    if _PROFILES_OK:
        selection = select_agent_for_task(task_type=task_type, query=query)
        result["primary_agent"] = selection.get("agent_id", "")
    else:
        result["primary_agent"] = "general_agent"

    # ── 2. Build handoff plan ──────────────────────────────────────
    if not _HANDOFF_OK:
        result["handoff_summary"] = "Handoff module unavailable"
        return result

    plan = build_handoff_plan(root_query=query, task_type=task_type)
    result["handoff_plan"] = {
        "plan_id": plan.plan_id,
        "root_task": plan.root_task,
        "handoff_count": len(plan.handoffs),
        "targets": [h.to_agent for h in plan.handoffs],
    }
    result["handoff_count"] = len(plan.handoffs)

    # ── 3. Execute handoffs (simulated) ────────────────────────────
    handoff_results = _execute_handoffs(
        plan=plan,
        query=query,
        rag_docs=rag_docs or [],
        memory_context=memory_context,
        retrieved_memories=retrieved_memories or [],
    )
    result["handoff_results"] = [
        {
            "handoff_id": r.handoff_id,
            "to_agent": r.to_agent,
            "status": r.status,
            "result_text": r.result_text[:300],
            "confidence": r.confidence,
            "sources_count": len(r.sources),
        }
        for r in handoff_results
    ]

    # Log handoffs
    for r in handoff_results:
        try:
            from research_agent.agents.handoff import handoff_result_to_dict
            append_handoff_log(handoff_result_to_dict(r))
        except Exception:
            pass

    # ── 4. Aggregate ───────────────────────────────────────────────
    agg = aggregate_handoff_results(plan, handoff_results)
    result["combined_answer"] = agg.get("combined_answer", "")
    result["handoff_sources"] = agg.get("sources", [])[:20]
    result["handoff_memory_ids"] = agg.get("memory_ids", [])[:20]

    # Build summary
    completed = agg.get("completed", 0)
    failed = agg.get("failed", 0)
    total = completed + failed
    result["handoff_summary"] = (
        f"Multi-Agent: {result['primary_agent']} coordinated "
        f"{total} handoffs ({completed} completed, {failed} failed)"
    )

    # ── 4.5. Arbitration (conflict detection + coordinator summary) ──
    result["arbitration"] = None
    result["coordinator_summary"] = ""
    try:
        from research_agent.agents.arbitration import (
            arbitrate_results, build_coordinator_final_summary,
        )
        arb = arbitrate_results(handoff_results, root_query=query)
        result["arbitration"] = arb
        if arb.get("conflicts", {}).get("has_conflict"):
            summary = build_coordinator_final_summary(arb, handoff_results, root_query=query)
            result["coordinator_summary"] = summary
            # Append to combined answer
            result["combined_answer"] += (
                "\n\n## Coordinator Arbitration\n\n" + arb.get("arbitration_text", "")
            )
    except Exception:
        pass  # arbitration failure must not affect main result

    # ── 5. Write to Memory Store ───────────────────────────────────
    if auto_write_memory and _ADAPTERS_OK and completed > 0:
        try:
            mem_result = save_module_result(
                result={
                    "analysis": result["combined_answer"],
                    "sources": result["handoff_sources"],
                    "tool_used": "multi_agent_orchestrator",
                    "evidence_status": "passed",
                    "used_llm": False,
                },
                module_name="experiment_analysis",
                auto_write=True,
            )
            result["memory_written"] = mem_result.get("ok", False)
            if not result["memory_written"]:
                result["memory_write_error"] = mem_result.get("error", "")
        except Exception as e:
            result["memory_write_error"] = str(e)

    # ── 6. Optional trace ────────────────────────────────────────────
    try:
        from research_agent.agents.tracing import trace_and_evaluate
        trace_eval = trace_and_evaluate(
            orchestrator_result=result,
            query=query,
            task_type=task_type,
            save_trace=True,
        )
        result["trace"] = trace_eval.get("trace")
        result["trace_quality"] = trace_eval.get("quality")
    except Exception:
        result["trace"] = None
        result["trace_quality"] = None

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Simulated handoff execution
# ═══════════════════════════════════════════════════════════════════════════

def _execute_handoffs(
    plan: Any,
    query: str,
    rag_docs: List[Dict[str, Any]],
    memory_context: str,
    retrieved_memories: List[Dict[str, Any]],
) -> List[Any]:
    """
    Execute a handoff plan using real specialist executors when available.

    Falls back to simulated execution if ``executors.py`` is unavailable.
    Each handoff is processed independently — one failure does not stop the batch.
    """
    # ── Try real executors first ──────────────────────────────────
    if _EXECUTORS_OK:
        results: List[Any] = []
        for h in plan.handoffs:
            try:
                result = execute_handoff_request(
                    request=h,
                    rag_docs=rag_docs,
                    memories=retrieved_memories,
                    use_llm=False,
                    save_memory=False,
                )
                results.append(result)
            except Exception as e:
                from datetime import datetime, timezone
                results.append(HandoffResult(
                    handoff_id=getattr(h, "handoff_id", ""),
                    from_agent=getattr(h, "from_agent", ""),
                    to_agent=getattr(h, "to_agent", ""),
                    status="failed",
                    result_text="",
                    confidence=0.0,
                    sources=[],
                    memory_ids=[],
                    created_at=datetime.now(timezone.utc).isoformat(),
                    error=f"{type(e).__name__}: {e}",
                ))
        return results

    # ── Fallback: simulated execution ─────────────────────────────
    return _execute_handoffs_simulated(
        plan=plan,
        rag_docs=rag_docs,
        retrieved_memories=retrieved_memories,
    )


def _execute_handoffs_simulated(
    plan: Any,
    rag_docs: List[Dict[str, Any]],
    retrieved_memories: List[Dict[str, Any]],
) -> List[Any]:
    """
    Simulated handoff execution — used when executors.py is unavailable.
    """
    from research_agent.agents.handoff import HandoffResult
    from datetime import datetime, timezone

    results: List[Any] = []
    _AGENT_RAG_FILTERS: Dict[str, Optional[str]] = {
        "paper_agent": "paper_note", "experiment_agent": "experiment_doc",
        "claim_agent": "paper_note", "progress_agent": "slide_doc",
        "report_agent": None, "code_agent": None, "memory_agent": None, "general_agent": None,
    }
    _AGENT_MEMORY_FILTERS: Dict[str, List[str]] = {
        "paper_agent": ["paper_note", "claim_support", "research_direction"],
        "experiment_agent": ["experiment_result", "progress_update"],
        "claim_agent": ["claim_support", "paper_note"],
        "progress_agent": ["progress_update", "meeting_note", "todo"],
        "report_agent": ["report_summary", "progress_update"],
        "code_agent": ["code_note", "todo"],
        "memory_agent": ["research_direction", "project_decision"],
        "general_agent": [],
    }

    for h in plan.handoffs:
        agent = h.to_agent
        now = datetime.now(timezone.utc).isoformat()
        rag_filter = _AGENT_RAG_FILTERS.get(agent)
        agent_rag = [d for d in rag_docs if rag_filter is None
                     or d.get("metadata", {}).get("source_type") == rag_filter] if rag_docs else []
        mem_types = _AGENT_MEMORY_FILTERS.get(agent, [])
        agent_mem = [m for m in retrieved_memories if _rf(m, "memory_type", "") in mem_types
                     ] if retrieved_memories and mem_types else []

        lines = [f"## {_agent_label(agent)} Output [simulated]", "",
                 f"**Task**: {h.task}", f"**Query**: {h.input_text[:200]}", ""]
        if agent_rag:
            lines.append(f"**RAG documents found**: {len(agent_rag)}")
            for d in agent_rag[:3]:
                path = d.get("metadata", {}).get("path", "?")
                title = d.get("metadata", {}).get("title", "")
                lines.append(f"- {title} (`{path}`)")
            lines.append("")
        if agent_mem:
            lines.append(f"**Memory records found**: {len(agent_mem)}")
            for m in agent_mem[:3]:
                lines.append(f"- [{_rf(m, 'memory_id', '?')[:12]}...] {_rf(m, 'summary', '')[:80]}")
            lines.append("")
        if not agent_rag and not agent_mem:
            lines.append("*(No specific RAG documents or memory records found.)*\n")

        results.append(HandoffResult(
            handoff_id=h.handoff_id, from_agent=h.from_agent, to_agent=agent,
            status="completed", result_text="\n".join(lines),
            confidence=0.7 if (agent_rag or agent_mem) else 0.3,
            sources=[{"path": d.get("metadata", {}).get("path", ""),
                      "source_type": d.get("metadata", {}).get("source_type", "")}
                     for d in agent_rag[:5]],
            memory_ids=[_rf(m, "memory_id", "") for m in agent_mem[:5]],
            created_at=now,
        ))
    return results


def _agent_label(agent_id: str) -> str:
    labels = {
        "paper_agent": "Paper Reader",
        "experiment_agent": "Experiment Analyst",
        "claim_agent": "Claim Supporter",
        "progress_agent": "Progress Tracker",
        "report_agent": "Report Writer",
        "code_agent": "Code Assistant",
        "memory_agent": "Memory Manager",
        "general_agent": "General Assistant",
        "coordinator_agent": "Coordinator",
    }
    return labels.get(agent_id, agent_id)


def _rf(rec: Any, key: str, default: Any = None) -> Any:
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)


# ═══════════════════════════════════════════════════════════════════════════
# Multi-agent toggle
# ═══════════════════════════════════════════════════════════════════════════

def is_multi_agent_enabled() -> bool:
    """Check the ENABLE_MULTI_AGENT env var."""
    return os.getenv("ENABLE_MULTI_AGENT", "false").strip().lower() == "true"
