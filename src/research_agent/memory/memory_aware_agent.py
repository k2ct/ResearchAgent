"""
Memory-aware Agent — augments RAG and LLM with Memory System results.

Core responsibilities
--------------------
1. **Retrieve memories** relevant to a query from the Memory Store.
2. **Merge** memory context with RAG documents.
3. **Format** memory-augmented prompts for LLM-based answering.
4. **Multi-level filtering**: short-term / mid-term / long-term,
   shared / private / global, status filtering.
5. **Expired / duplicate handling**: optionally calls consolidation
   before retrieval.

All functions are backward-compatible: when the Memory Store is unavailable
they return empty results rather than raising.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe_get(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, dict):
        return data.get(key, default)
    return default


def _is_active(record: Dict[str, Any]) -> bool:
    """True if the record status permits retrieval."""
    status = _safe_get(record, "status", "active")
    return status not in ("merged", "archived")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Memory loader (graceful fallback)
# ═══════════════════════════════════════════════════════════════════════════

def load_memories_for_agent() -> List[Dict[str, Any]]:
    """
    Load all memory records via the Memory Retriever's store-backed path.

    Returns an empty list when the store is unavailable.
    """
    try:
        from research_agent.memory.retriever import load_memories_for_retrieval
        return load_memories_for_retrieval()
    except ImportError:
        pass

    # Direct fallback
    try:
        from research_agent.memory.store import load_memories
        return load_memories()
    except ImportError:
        return []


# ═══════════════════════════════════════════════════════════════════════════
# 2. Retrieve memories for a query
# ═══════════════════════════════════════════════════════════════════════════

# Map task_type → suggested memory retrieval filters
_TASK_TO_MEMORY_FILTERS: Dict[str, Dict[str, Any]] = {
    "paper_question": {
        "memory_types": ["paper_note", "claim_support", "research_direction"],
        "source_modules": ["paper_reading", "claim_support"],
        "levels": ["long_term", "mid_term"],
    },
    "experiment_analysis": {
        "memory_types": ["experiment_result", "progress_update"],
        "source_modules": ["experiment_tool", "ppt_progress"],
        "levels": ["mid_term", "long_term"],
    },
    "dataset_recommendation": {
        "memory_types": ["experiment_result", "research_direction", "general_note"],
        "source_modules": ["experiment_tool"],
        "levels": ["long_term", "mid_term"],
    },
    "report_generation": {
        "memory_types": ["report_summary", "progress_update", "paper_note"],
        "source_modules": ["report_writer", "ppt_progress", "paper_reading"],
        "levels": ["mid_term", "long_term"],
    },
    "code_help": {
        "memory_types": ["code_note", "general_note"],
        "source_modules": ["code_assistant"],
        "levels": ["mid_term", "short_term"],
    },
    "general": {
        "memory_types": ["research_direction", "user_preference", "project_decision",
                         "todo", "general_note"],
        "source_modules": [],
        "levels": ["long_term", "mid_term"],
    },
}

# Agent-type aliases (friendly names → canonical)
_AGENT_ALIASES: Dict[str, str] = {
    "paper": "paper_agent",
    "claim": "claim_agent",
    "experiment": "experiment_agent",
    "progress": "progress_agent",
    "report": "report_agent",
    "code": "code_agent",
    "coordinator": "coordinator",
    "memory": "memory_agent",
    "general": "general_agent",
}


def _normalise_agent(agent: str) -> str:
    return _AGENT_ALIASES.get(agent, agent)


def retrieve_memories_for_query(
    query: str,
    task_type: str = "general",
    *,
    owner_agent: Optional[str] = None,
    max_results: int = 5,
    include_short_term: bool = True,
    include_expired: bool = False,
    extra_tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve memory records relevant to *query* and *task_type*.

    Strategy:
    1. Load all memories from the store.
    2. Apply task-type-specific filters (memory_type, source_module, level).
    3. Keyword search against the query.
    4. Tag-based broadening.
    5. Sort by importance (desc) and recency.
    6. Truncate to *max_results*.

    Args:
        query: The user's question.
        task_type: One of paper_question / experiment_analysis /
            dataset_recommendation / report_generation / code_help / general.
        owner_agent: If set, prefers memories owned by this agent.
        max_results: Maximum number of memory records to return.
        include_short_term: Whether to include short-term memories.
        include_expired: Whether to include expired memories.
        extra_tags: Additional tags to match against.

    Returns:
        List of memory record dicts, sorted by relevance.
    """
    records = load_memories_for_agent()
    if not records:
        return []

    # Get task-appropriate filters
    filters = _TASK_TO_MEMORY_FILTERS.get(task_type, _TASK_TO_MEMORY_FILTERS["general"])
    allowed_types = set(filters.get("memory_types", []))
    allowed_modules = set(filters.get("source_modules", []))
    allowed_levels = set(filters.get("levels", ["long_term", "mid_term"]))

    if include_short_term:
        allowed_levels.add("short_term")

    now = datetime.now(timezone.utc)
    query_lower = query.lower()

    scored: List[Tuple[float, Dict[str, Any]]] = []

    for r in records:
        # Skip non-active records (merged, archived) unless expired wanted
        status = _safe_get(r, "status", "active")
        if status == "merged":
            continue
        if status == "archived":
            continue
        if status == "expired" and not include_expired:
            continue

        # Filter by type
        mt = _safe_get(r, "memory_type", "")
        if allowed_types and mt not in allowed_types:
            # Relaxed: if source_module matches, allow through
            sm = _safe_get(r, "source_module", "")
            if not allowed_modules or sm not in allowed_modules:
                continue

        # Filter by level
        level = _safe_get(r, "memory_level", "short_term")
        if level not in allowed_levels:
            continue

        # Score the record against the query
        score = _score_record_for_query(r, query_lower, owner_agent)

        if score > 0:
            scored.append((score, r))

    # Sort by score descending, then by importance descending, then by recency
    scored.sort(
        key=lambda x: (
            x[0],
            _safe_get(x[1], "importance", 3),
            _safe_get(x[1], "created_at", "0"),
        ),
        reverse=True,
    )

    return [r for _, r in scored[:max_results]]


def _score_record_for_query(
    record: Dict[str, Any],
    query_lower: str,
    owner_agent: Optional[str],
) -> float:
    """Score a single memory record against a query. Higher = more relevant."""
    score = 0.0

    content = (_safe_get(record, "content", "") or "").lower()
    summary = (_safe_get(record, "summary", "") or "").lower()
    title = (_safe_get(record, "source_title", "") or "").lower()
    tags = [t.lower() for t in (_safe_get(record, "tags", []) or [])]
    haystack = f"{content} {summary} {title}"

    # 1. Keyword match in content/summary/title
    keywords = _extract_keywords(query_lower)
    for kw in keywords:
        if kw in haystack:
            score += 2.0
        # Bonus for tag match
        if any(kw in t for t in tags):
            score += 3.0
        # Big bonus for title match
        if kw in title:
            score += 2.0

    # 2. Exact phrase match
    if query_lower in haystack:
        score += 5.0

    # 3. Owner match (slight boost)
    if owner_agent:
        record_owner = _safe_get(record, "owner_agent", "").lower()
        if record_owner == _normalise_agent(owner_agent):
            score += 1.0

    # 4. Importance boost
    importance = _safe_get(record, "importance", 3)
    if isinstance(importance, (int, float)):
        score += importance * 0.5

    # 5. Recency boost (newer = slightly better)
    created = _safe_get(record, "created_at", "")
    if created:
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            days_ago = (datetime.now(timezone.utc) - created_dt).days
            if days_ago <= 1:
                score += 3.0
            elif days_ago <= 7:
                score += 1.5
            elif days_ago <= 30:
                score += 0.5
        except (ValueError, TypeError):
            pass

    return score


def _extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from a query."""
    import re
    # Chinese + English words
    tokens = re.findall(r'[a-zA-Z0-9_\-]{2,}|[一-鿿]{2,}', text)
    # Filter stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "of", "in", "on",
        "at", "to", "for", "with", "and", "or", "not", "it", "this", "that",
        "我们", "这个", "那个", "什么", "怎么", "如何", "为什么", "可以",
        "哪些", "哪个", "怎么样", "是什么", "有没有",
    }
    return [t for t in tokens if t.lower() not in stopwords]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Merge memory with RAG context
# ═══════════════════════════════════════════════════════════════════════════

def format_memory_context(
    memories: List[Dict[str, Any]],
    max_memories: int = 5,
    max_chars_per_memory: int = 400,
) -> str:
    """
    Format memory records into a string suitable for inclusion in an LLM prompt.

    Args:
        memories: List of memory records from ``retrieve_memories_for_query()``.
        max_memories: Maximum number of memory records to include.
        max_chars_per_memory: Truncation length for each memory's content.

    Returns:
        A formatted string like::

            ## Research Memory (long-term)
            [mem_001] (paper_note) Summary of the paper...
            Source: data/papers/paper.md | Tags: hallucination, bias

            ## Research Memory (mid-term)
            ...
    """
    if not memories:
        return "No relevant research memories found."

    # Group by level
    by_level: Dict[str, List[Dict[str, Any]]] = {
        "long_term": [],
        "mid_term": [],
        "short_term": [],
    }
    for m in memories[:max_memories]:
        level = _safe_get(m, "memory_level", "mid_term")
        if level in by_level:
            by_level[level].append(m)

    sections: List[str] = []
    level_labels = {
        "long_term": "## Research Memory (long-term)",
        "mid_term": "## Research Memory (mid-term)",
        "short_term": "## Research Memory (short-term)",
    }

    for level in ("long_term", "mid_term", "short_term"):
        items = by_level[level]
        if not items:
            continue

        sections.append(level_labels[level])
        sections.append("")

        for m in items:
            mid = _safe_get(m, "memory_id", "?")[:12]
            mtype = _safe_get(m, "memory_type", "?")
            summary = (_safe_get(m, "summary", "") or "")[:max_chars_per_memory]
            content = (_safe_get(m, "content", "") or "")[:max_chars_per_memory]
            source = _safe_get(m, "source_title", "") or _safe_get(m, "source_path", "")
            tags = ", ".join((_safe_get(m, "tags", []) or [])[:5])
            scope = _safe_get(m, "memory_scope", "private")

            # Use summary if available, otherwise content preview
            display_text = summary if summary else content

            sections.append(f"- **[{mid}]** ({mtype}, {scope}) {display_text}")
            if source:
                sections.append(f"  Source: {source}")
            if tags:
                sections.append(f"  Tags: {tags}")
            sections.append("")

    if not sections:
        return "No relevant research memories found."

    return "\n".join(sections)


def merge_rag_and_memory_context(
    rag_context: str,
    memory_context: str,
    query: str = "",
) -> str:
    """
    Combine RAG retrieval results with memory context into a single
    prompt-ready string.

    The merged context can be passed directly to an LLM or used as-is
    for template-based answering.
    """
    parts: List[str] = []

    if memory_context and memory_context != "No relevant research memories found.":
        parts.append("# Memory Context (Research Progress & Prior Knowledge)")
        parts.append("")
        parts.append(memory_context)
        parts.append("")

    if rag_context and rag_context != "未检索到相关资料。":
        parts.append("# RAG Context (Knowledge Base Documents)")
        parts.append("")
        parts.append(rag_context)

    if not parts:
        return "No relevant context available from RAG or Memory."

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Memory-augmented answer builder
# ═══════════════════════════════════════════════════════════════════════════

def build_memory_augmented_answer(
    query: str,
    task_type: str,
    rag_context: str = "",
    rag_sources: Optional[List[Dict[str, Any]]] = None,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """
    Build an answer that integrates RAG + Memory.

    This is the main entry point for memory-aware question answering.
    It:
    1. Retrieves relevant memories.
    2. Formats memory + RAG context.
    3. Optionally calls LLM for synthesis.

    Args:
        query: The user's question.
        task_type: Task type hint (paper_question, experiment_analysis, etc.).
        rag_context: Pre-retrieved RAG context string.
        rag_sources: Pre-retrieved RAG sources.
        use_llm: If True, attempt LLM-based synthesis.

    Returns:
        {
            "query": str,
            "task_type": str,
            "memories": [...],
            "memory_context": str,
            "rag_context": str,
            "merged_context": str,
            "answer": str,
            "sources": [...],
            "memory_count": int,
            "used_llm": bool,
        }
    """
    memories = retrieve_memories_for_query(
        query=query,
        task_type=task_type,
        max_results=5,
    )

    memory_context = format_memory_context(memories)
    merged_context = merge_rag_and_memory_context(rag_context, memory_context, query)

    # Build answer
    if use_llm:
        answer = _try_llm_answer(query, merged_context, task_type)
        used_llm = True
    else:
        answer = _build_template_answer(query, merged_context, memories, rag_sources or [])
        used_llm = False

    # Merge sources: RAG + memory
    all_sources: List[Dict[str, Any]] = list(rag_sources or [])
    seen_paths = {s.get("path", "") for s in all_sources}
    for m in memories:
        sp = _safe_get(m, "source_path", "")
        if sp and sp not in seen_paths:
            seen_paths.add(sp)
            all_sources.append({
                "path": sp,
                "source_type": _safe_get(m, "memory_type", "memory"),
                "title": _safe_get(m, "source_title", ""),
                "memory_id": _safe_get(m, "memory_id", ""),
                "from_memory": True,
            })

    return {
        "query": query,
        "task_type": task_type,
        "memories": memories,
        "memory_context": memory_context,
        "rag_context": rag_context,
        "merged_context": merged_context,
        "answer": answer,
        "sources": all_sources,
        "memory_count": len(memories),
        "used_llm": used_llm,
    }


def _build_template_answer(
    query: str,
    merged_context: str,
    memories: List[Dict[str, Any]],
    rag_sources: List[Dict[str, Any]],
) -> str:
    """Build a template-based answer from merged context."""
    parts: List[str] = [
        f"用户问题：{query}",
        "",
    ]

    if memories:
        parts.append("## 研究记忆（前期成果 & 偏好）")
        parts.append("")
        for i, m in enumerate(memories[:3], start=1):
            summary = _safe_get(m, "summary", "") or _safe_get(m, "content", "")[:200]
            mtype = _safe_get(m, "memory_type", "?")
            parts.append(f"{i}. [{mtype}] {summary}")
        parts.append("")

    if rag_sources:
        parts.append("## 知识库资料")
        parts.append("")
        for i, s in enumerate(rag_sources[:3], start=1):
            path = _safe_get(s, "path", "?")
            title = _safe_get(s, "title", "")
            parts.append(f"{i}. {title} (`{path}`)" if title else f"{i}. `{path}`")
        parts.append("")

    if not memories and not rag_sources:
        parts.append("暂未在本地科研资料库和记忆中检索到足够相关的资料。")
    else:
        parts.append(
            "说明：以上回答基于本地科研资料库（RAG）和研究记忆（Memory）的综合检索结果。"
            "如需更详细的论证支持，请参考 Claim Support 报告或 Paper Reading 笔记。"
        )

    return "\n".join(parts)


def _try_llm_answer(
    query: str,
    merged_context: str,
    task_type: str,
) -> str:
    """Try LLM-based synthesis. Falls back to template on failure."""
    try:
        from research_agent.llm.client import (
            is_llm_enhancement_enabled,
            invoke_llm_with_fallback,
        )
        from langchain_core.messages import SystemMessage, HumanMessage

        if not is_llm_enhancement_enabled():
            return _build_template_answer(query, merged_context, [], [])

        system = SystemMessage(content=(
            "你是科研助手。你必须严格基于提供的 RAG Context 和 Memory Context "
            "回答用户问题。\n"
            "规则：\n"
            "1. 只能使用提供的信息。\n"
            "2. 不要编造实验结果或论文结论。\n"
            "3. 如果资料不足，请明确说明。\n"
            "4. 在回答中引用 Memory 来源（标注 memory_id）。\n"
            "5. 在回答中引用 RAG Sources（标注 path）。"
        ))

        human = HumanMessage(content=(
            f"## 用户问题\n{query}\n\n"
            f"## 任务类型\n{task_type}\n\n"
            f"{merged_context[:8000]}\n\n"
            "请综合以上 RAG Context 和 Memory Context 回答用户问题。"
        ))

        result = invoke_llm_with_fallback(
            messages=[system, human],
            fallback_text=_build_template_answer(query, merged_context, [], []),
            feature_name="memory_aware_answer",
        )

        return result["text"]

    except Exception:
        return _build_template_answer(query, merged_context, [], [])


# ═══════════════════════════════════════════════════════════════════════════
# 5. Task-type-aware memory retrieval
# ═══════════════════════════════════════════════════════════════════════════

def get_continuity_suggestions(
    task_type: str,
    max_suggestions: int = 3,
) -> List[Dict[str, Any]]:
    """
    Retrieve continuity suggestions — memories that indicate prior work,
    preferences, or direction relevant to the current task.

    This is used for "今天应该怎么安排科研任务" type queries.
    """
    records = load_memories_for_agent()
    if not records:
        return []

    # Prioritise: research_direction, todo, progress_update, project_decision
    priority_types = {
        "research_direction": 5,
        "project_decision": 4,
        "user_preference": 4,
        "todo": 3,
        "progress_update": 3,
        "report_summary": 2,
    }

    scored: List[Tuple[int, Dict[str, Any]]] = []
    now = datetime.now(timezone.utc)

    for r in records:
        if not _is_active(r):
            continue

        mt = _safe_get(r, "memory_type", "")
        priority = priority_types.get(mt, 0)
        if priority == 0:
            continue

        # Boost recent items
        created = _safe_get(r, "created_at", "")
        if created:
            try:
                days = (now - datetime.fromisoformat(created.replace("Z", "+00:00"))).days
                if days <= 3:
                    priority += 2
                elif days <= 14:
                    priority += 1
            except (ValueError, TypeError):
                pass

        scored.append((priority, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:max_suggestions]]


# ═══════════════════════════════════════════════════════════════════════════
# 6. Convenience: answer with memory
# ═══════════════════════════════════════════════════════════════════════════

def answer_with_memory(
    query: str,
    task_type: str = "general",
    rag_context: str = "",
    rag_sources: Optional[List[Dict[str, Any]]] = None,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """
    Convenience wrapper — answer a query with full RAG + Memory integration.

    This is the recommended one-call entry point for memory-aware answering.
    """
    return build_memory_augmented_answer(
        query=query,
        task_type=task_type,
        rag_context=rag_context,
        rag_sources=rag_sources,
        use_llm=use_llm,
    )
