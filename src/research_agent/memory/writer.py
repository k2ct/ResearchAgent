"""
Memory Writer — rule-based decision engine for research agent memory.

Decides **memory_level**, **memory_scope**, **owner_agent**, **memory_type**,
**tags**, and **shared_with** for any piece of content produced by a
research module (claim support, paper reading, PPT progress, experiment
analysis, report writer, etc.).

Design:
- v1: pure rule-based heuristics — no LLM dependency.
- Explicit user instructions ("remember this", "save to long-term memory")
  always take precedence.
- Mid-term topics that are repeatedly mentioned (>=3 times) without a
  completion boundary are automatically promoted to long-term.
- Calls into ``schema.py`` / ``store.py`` when available; graceful
  fallback stubs otherwise.
"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Fallback imports — schema / store may not be merged yet
# ═══════════════════════════════════════════════════════════════════════════

_SCHEMA_AVAILABLE = False
_STORE_AVAILABLE = False
_PRIVACY_AVAILABLE = False

try:
    from research_agent.memory.schema import create_memory_record  # type: ignore[import-untyped]
    _SCHEMA_AVAILABLE = True
except ImportError:
    pass

try:
    from research_agent.memory.store import append_memory, query_memories  # type: ignore[import-untyped]
    _STORE_AVAILABLE = True
except ImportError:
    pass

try:
    from research_agent.memory.privacy_scope import validate_scope_on_write  # type: ignore[import-untyped]
    _PRIVACY_AVAILABLE = True
except ImportError:
    pass


def normalize_memory_id(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure every record has a ``memory_id`` field.

    If the record was created by an older fallback path that used
    ``record_id``, migrate it to ``memory_id`` and stash the old key
    in metadata for traceability.
    """
    if "memory_id" not in record and "record_id" in record:
        record["memory_id"] = record["record_id"]
        record.setdefault("metadata", {})["legacy_record_id"] = record.pop("record_id")
    return record


def _fallback_create_record(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal record builder when schema.py is unavailable."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "memory_id": f"mem_{int(datetime.now().timestamp() * 1000)}",
        "created_at": now,
        "updated_at": now,
        "last_accessed_at": None,
        "memory_level": fields.get("memory_level", "mid_term"),
        "memory_scope": fields.get("memory_scope", "private"),
        "memory_type": fields.get("memory_type", "general_note"),
        "owner_agent": fields.get("owner_agent", "memory_agent"),
        "shared_with": fields.get("shared_with", []),
        "content": fields.get("content", ""),
        "summary": fields.get("summary", fields.get("content", "")[:200]),
        "tags": fields.get("tags", []),
        "source_module": fields.get("source_module", ""),
        "source_path": fields.get("source_path", ""),
        "source_id": fields.get("source_id", ""),
        "source_title": fields.get("source_title", ""),
        "importance": fields.get("importance", 3),
        "metadata": fields.get("metadata", {}),
        "status": "active",
        "visibility": fields.get("visibility", "private"),
    }


def _fallback_append(record: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback when store.py is unavailable — returns record as-is."""
    memory_id = record.get("memory_id", record.get("record_id", ""))
    return {"ok": True, "memory_id": memory_id,
            "written": False, "reason": "store.py not available — record returned but not persisted"}


def _fallback_query(**kwargs: Any) -> List[Dict[str, Any]]:
    """Fallback query — returns empty list."""
    return []


# ═══════════════════════════════════════════════════════════════════════════
# 1. Explicit memory instruction detection
# ═══════════════════════════════════════════════════════════════════════════

# (patterns, level)
_EXPLICIT_INSTRUCTIONS: List[Tuple[List[str], str]] = [
    # ── long_term ──
    (["记住", "长期记忆", "保存到长期记忆", "以后都要记得", "这是我的长期方向",
      "长期方向", "永久保存", "请记住", "牢记", "永远记住",
      "remember this", "save to long-term memory", "long-term memory",
      "permanent memory", "never forget"], "long_term"),
    # ── mid_term ──
    (["这个任务之后还要继续", "暂时记住", "本阶段记住", "近期任务",
      "阶段性记忆", "这阶段", "当前阶段需要", "项目期间记住",
      "ongoing task", "keep this for this project", "mid-term",
      "medium-term", "keep for now"], "mid_term"),
    # ── short_term ──
    (["临时记一下", "本次对话", "临时上下文", "短期记忆", "暂时记",
      "本轮", "这次先记", "仅本次",
      "short-term", "temporary", "temp note", "this session"], "short_term"),
]


def detect_explicit_memory_instruction(text: str) -> Dict[str, Any]:
    """
    Detect whether *text* contains an explicit memory-save instruction.

    Returns::

        {
            "has_instruction": bool,
            "suggested_level": "long_term" | "mid_term" | "short_term" | None,
            "reason": str,
        }
    """
    text_lower = text.lower()
    for patterns, level in _EXPLICIT_INSTRUCTIONS:
        for pat in patterns:
            if pat.lower() in text_lower:
                return {
                    "has_instruction": True,
                    "suggested_level": level,
                    "reason": f"Matched explicit instruction pattern: '{pat}' → {level}",
                }
    return {"has_instruction": False, "suggested_level": None, "reason": "No explicit memory instruction found"}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Memory level inference
# ═══════════════════════════════════════════════════════════════════════════

# Patterns strongly indicative of long-term research value
_LONG_TERM_PATTERNS: List[str] = [
    "研究方向", "research direction", "research agenda", "long-term goal",
    "核心结论", "最终结论", "core finding", "key insight",
    "架构决策", "architecture decision", "design principle",
    "用户偏好", "user preference", "我的习惯",
    "论文核心", "paper core", "thesis",
    "项目定位", "project positioning",
]

_MID_TERM_PATTERNS: List[str] = [
    "当前实验", "current experiment", "ongoing experiment",
    "本阶段", "this phase", "当前阶段",
    "未完成", "unfinished", "todo", "下一步", "next step",
    "近期", "组会", "group meeting",
    "开发中", "in progress", "under development",
    "正在推进", "pushing forward",
]

_SHORT_TERM_PATTERNS: List[str] = [
    "临时", "temporary", "temp",
    "一次性", "one-off", "one time",
    "debug", "调试",
    "本轮对话", "this conversation",
    "端口占用", "port conflict",
    "报错", "error message", "stack trace",
]

# Source modules that strongly indicate long-term
_LONG_TERM_SOURCES = {"paper_reading"}
_MID_TERM_SOURCES = {"ppt_progress", "experiment_tool"}
_SHORT_TERM_SOURCES = set()


def infer_memory_level(
    content: str,
    source_module: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Infer the appropriate memory level for *content*.

    Priority:
    A. Explicit user instruction (always wins).
    B. Source-module heuristics.
    C. Content keyword matching.
    D. Default → mid_term.

    Returns::

        {"memory_level": str, "confidence": float, "reason": str}
    """
    # A. Explicit instruction
    explicit = detect_explicit_memory_instruction(content)
    if explicit["has_instruction"] and explicit["suggested_level"]:
        return {
            "memory_level": explicit["suggested_level"],
            "confidence": 1.0,
            "reason": explicit["reason"],
        }

    content_lower = content.lower()
    scores: Dict[str, float] = {"long_term": 0.0, "mid_term": 0.0, "short_term": 0.0}

    # B. Source-module signal
    if source_module in _LONG_TERM_SOURCES:
        scores["long_term"] += 2.0
    if source_module in _MID_TERM_SOURCES:
        scores["mid_term"] += 1.0
    if source_module in _SHORT_TERM_SOURCES:
        scores["short_term"] += 1.0

    # Content contains research direction / next steps → long-term
    meta = metadata or {}
    if meta.get("has_next_steps") or meta.get("is_research_direction"):
        scores["long_term"] += 2.0

    # C. Keyword scoring
    for pat in _LONG_TERM_PATTERNS:
        if pat.lower() in content_lower:
            scores["long_term"] += 1.5
    for pat in _MID_TERM_PATTERNS:
        if pat.lower() in content_lower:
            scores["mid_term"] += 1.0
    for pat in _SHORT_TERM_PATTERNS:
        if pat.lower() in content_lower:
            scores["short_term"] += 1.5

    # Special: paper reading with structured notes → long-term
    if source_module == "paper_reading" and ("## " in content or "# " in content):
        scores["long_term"] += 2.0
    # PPT progress with research direction or next steps → elevated
    if source_module == "ppt_progress":
        if any(kw in content_lower for kw in ("research question", "研究方向", "长期", "next step", "下一步")):
            scores["long_term"] += 1.5

    # Determine level
    best_level = "mid_term"
    best_score = scores["mid_term"]
    for level in ("long_term", "mid_term", "short_term"):
        if scores[level] > best_score:
            best_level = level
            best_score = scores[level]

    # Confidence: normalize
    total = sum(scores.values())
    confidence = min(0.95, best_score / max(total, 1.0) + 0.3) if total > 0 else 0.5

    reasons_parts = []
    if source_module:
        reasons_parts.append(f"source_module={source_module}")
    reasons_parts.append(f"scores L={scores['long_term']:.1f} M={scores['mid_term']:.1f} S={scores['short_term']:.1f}")

    return {
        "memory_level": best_level,
        "confidence": round(confidence, 2),
        "reason": "; ".join(reasons_parts),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Owner agent inference
# ═══════════════════════════════════════════════════════════════════════════

_SOURCE_TO_OWNER: Dict[str, str] = {
    "paper_reading": "paper_agent",
    "paper_note": "paper_agent",
    "claim_support": "claim_agent",
    "ppt_progress": "progress_agent",
    "progress_update": "progress_agent",
    "experiment_analysis": "experiment_agent",
    "experiment_result": "experiment_agent",
    "experiment_tool": "experiment_agent",
    "report_writer": "report_agent",
    "report_summary": "report_agent",
    "code_assistant": "code_agent",
    "code_note": "code_agent",
    "project_decision": "coordinator",
    "user_preference": "general_agent",
}

# Keywords in content that hint at owner even without source_module
_CONTENT_OWNER_HINTS: List[Tuple[List[str], str]] = [
    (["论文", "paper", "abstract", "introduction", "related work", "conclusion",
      "methodology", "experiment section"], "paper_agent"),
    (["claim", "论点", "evidence", "support", "refute", "evidence check"], "claim_agent"),
    (["slide", "PPT", "组会", "presentation", "progress report"], "progress_agent"),
    (["experiment", "实验", "benchmark", "evaluation", "metric", "result"], "experiment_agent"),
    (["report", "汇报", "summary", "summarize", "总结"], "report_agent"),
    (["code", "代码", "bug", "error", "import", "function", "class"], "code_agent"),
    (["architecture", "架构", "decision", "决策", "project direction"], "coordinator"),
]


def infer_owner_agent(source_module: str, memory_type: str = "", content: str = "") -> str:
    """
    Infer which agent owns this memory.

    Rules:
    1. Direct ``source_module`` → ``owner_agent`` mapping.
    2. Content keyword hints as fallback.
    3. Default → ``"memory_agent"``.
    """
    # Direct mapping
    if source_module in _SOURCE_TO_OWNER:
        return _SOURCE_TO_OWNER[source_module]
    if memory_type in _SOURCE_TO_OWNER:
        return _SOURCE_TO_OWNER[memory_type]

    # Content hints
    content_lower = content.lower()
    for patterns, agent in _CONTENT_OWNER_HINTS:
        for pat in patterns:
            if pat.lower() in content_lower:
                return agent

    return "memory_agent"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Memory type inference
# ═══════════════════════════════════════════════════════════════════════════

_SOURCE_TO_MEMORY_TYPE: Dict[str, str] = {
    "claim_support": "claim_support",
    "paper_reading": "paper_note",
    "paper_note": "paper_note",
    "ppt_progress": "progress_update",
    "progress_update": "progress_update",
    "experiment_analysis": "experiment_result",
    "experiment_result": "experiment_result",
    "experiment_tool": "experiment_result",
    "report_writer": "report_summary",
    "report_summary": "report_summary",
}

# Content-based hints for memory_type
_TYPE_HINTS: List[Tuple[List[str], str]] = [
    (["bug", "error", "traceback", "exception", "报错", "修复", "fix"], "code_note"),
    (["todo", "下一步", "next step", "future work", "待办"], "todo"),
    (["preference", "偏好", "习惯", "我喜欢", "我常用", "settings"], "user_preference"),
    (["architecture", "架构", "design decision", "设计决策"], "project_decision"),
    (["研究方向", "research direction", "长期方向"], "research_direction"),
]


def infer_memory_type(
    source_module: str,
    content: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Infer the memory type (category tag) from source module and content.

    Returns a string like ``"claim_support"``, ``"paper_note"``,
    ``"progress_update"``, ``"experiment_result"``, ``"report_summary"``,
    ``"code_note"``, ``"todo"``, ``"user_preference"``,
    ``"research_direction"``, ``"project_decision"``, or ``"general_note"``.
    """
    # Direct source mapping
    if source_module in _SOURCE_TO_MEMORY_TYPE:
        return _SOURCE_TO_MEMORY_TYPE[source_module]

    # Content hints
    content_lower = content.lower()
    for patterns, mtype in _TYPE_HINTS:
        for pat in patterns:
            if pat.lower() in content_lower:
                return mtype

    return "general_note"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Memory scope inference  (private / shared / global)
# ═══════════════════════════════════════════════════════════════════════════

# Types that should always be shared (plus to whom)
_SHARED_SCOPE_RULES: Dict[str, List[str]] = {
    "research_direction": ["paper_agent", "claim_agent", "progress_agent", "report_agent", "coordinator"],
    "project_decision": ["paper_agent", "claim_agent", "experiment_agent", "progress_agent", "report_agent", "coordinator"],
    "claim_support": ["paper_agent", "report_agent", "progress_agent"],
    "progress_update": ["experiment_agent", "report_agent", "coordinator"],
    "experiment_result": ["claim_agent", "report_agent", "progress_agent"],
    "paper_note": ["claim_agent", "report_agent"],
    "report_summary": ["progress_agent", "coordinator"],
    "todo": [],
    "code_note": [],
    "user_preference": ["coordinator"],
    "general_note": [],
}


def infer_memory_scope(
    content: str,
    memory_level: str,
    owner_agent: str,
    source_module: str = "",
    memory_type: str = "",
) -> Dict[str, Any]:
    """
    Decide memory scope: ``"private"``, ``"shared"``, or ``"global"``.

    Default is ``"private"``.  Certain memory types are automatically
    promoted to ``"shared"`` with a suggested ``shared_with`` list.

    Returns::

        {"memory_scope": str, "shared_with": [str, ...], "reason": str}
    """
    # Global: long_term research_direction or project_decision
    if memory_level == "long_term" and memory_type in ("research_direction", "project_decision"):
        return {
            "memory_scope": "global",
            "shared_with": _SHARED_SCOPE_RULES.get(memory_type, []),
            "reason": f"long_term {memory_type} → global scope",
        }

    # Shared: types with explicit sharing rules
    if memory_type in _SHARED_SCOPE_RULES:
        shared_with = _SHARED_SCOPE_RULES[memory_type]
        if shared_with:
            # Remove self from shared_with
            shared_with = [a for a in shared_with if a != owner_agent]
            return {
                "memory_scope": "shared",
                "shared_with": shared_with,
                "reason": f"memory_type={memory_type} has sharing rules → shared with {shared_with}",
            }

    # Long-term content that mentions cross-module relevance → shared
    if memory_level == "long_term":
        content_lower = content.lower()
        cross_module_keywords = ["pipeline", "workflow", "integration", "end-to-end",
                                  "all agents", "shared", "common", "shared knowledge"]
        if any(kw in content_lower for kw in cross_module_keywords):
            return {
                "memory_scope": "shared",
                "shared_with": ["coordinator", "paper_agent", "claim_agent", "experiment_agent", "progress_agent", "report_agent"],
                "reason": "long_term cross-module content → shared",
            }

    # Default: private
    return {
        "memory_scope": "private",
        "shared_with": [],
        "reason": "default → private scope",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Tag inference
# ═══════════════════════════════════════════════════════════════════════════

_TAG_KEYWORDS: List[str] = [
    # English
    "LVLM", "VLM", "MLLM", "LLM",
    "hallucination", "bias", "stereotype", "co-occurrence",
    "RAG", "LangGraph", "MinerU", "PDF", "PPT",
    "experiment", "dataset", "COCO", "OpenImages", "MIAP",
    "VIGNETTE", "MoLE", "fairness", "evaluation", "benchmark",
    "safety", "guardrail", "caption", "generation", "attribute",
    "bounding box", "object detection", "visual reasoning",
    "cross-modal", "alignment",
    # Chinese
    "幻觉", "偏见", "刻板印象", "共现", "公平性",
    "论文", "实验", "组会", "数据集", "评估",
    "多模态", "视觉语言", "生成", "检测", "推理",
    "对齐", "安全", "护栏",
]

# Canonicalize: lowercase, strip punctuation
_TAG_LOOKUP = {t.lower(): t for t in _TAG_KEYWORDS}


def infer_tags(content: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Extract relevant tags from content and metadata.

    Matches a controlled vocabulary of ~40 research-domain terms
    (English + Chinese).  Returns at most 15 tags.
    """
    text = content.lower()
    # Also search metadata values
    if metadata:
        for v in metadata.values():
            if isinstance(v, str):
                text += " " + v.lower()

    found: List[str] = []
    seen: set = set()

    # Exact match for multi-word tags first
    for tag in sorted(_TAG_KEYWORDS, key=lambda t: -len(t)):
        tag_lower = tag.lower()
        if tag_lower in text and tag_lower not in seen:
            seen.add(tag_lower)
            found.append(_TAG_LOOKUP.get(tag_lower, tag))

    return found[:15]


# ═══════════════════════════════════════════════════════════════════════════
# 7. Mid-term → long-term promotion
# ═══════════════════════════════════════════════════════════════════════════

_COMPLETION_MARKERS = [
    "completed", "finished", "done", "已完成", "完成", "结束",
    "finished", "closed", "resolved", "已解决", "merged",
]


def should_promote_mid_to_long(
    content: str,
    existing_memories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Check whether a mid_term memory should be promoted to long_term.

    Rule: if the same topic / tags appear in *existing_memories* >= 3
    times (mid_term or long_term), and none of those have a completion
    marker (completed/finished/done/已完成), suggest promotion.

    Returns::

        {"promote": bool, "reason": str, "matched_count": int}
    """
    tags = set(infer_tags(content))
    content_lower = content.lower()

    matched = 0
    for mem in existing_memories:
        mem_level = mem.get("memory_level", "")
        mem_tags = set(mem.get("tags", []))
        mem_content = (mem.get("content", "") + " " + mem.get("summary", "")).lower()

        # Check tag overlap
        tag_overlap = len(tags & mem_tags)
        # Check content keyword overlap (simple shared word ratio)
        content_words = set(content_lower.split())
        mem_words = set(mem_content.split())
        word_overlap = len(content_words & mem_words) / max(len(content_words), 1)

        if tag_overlap >= 1 or word_overlap > 0.3:
            matched += 1

    # Check if any existing matching memory has a completion marker
    has_completion = False
    for mem in existing_memories:
        mem_text = (mem.get("content", "") + " " + mem.get("summary", "")).lower()
        if any(marker.lower() in mem_text for marker in _COMPLETION_MARKERS):
            has_completion = True
            break

    if matched >= 3 and not has_completion:
        return {
            "promote": True,
            "reason": f"Topic appears {matched} times in existing memories with no completion marker",
            "matched_count": matched,
        }

    return {
        "promote": False,
        "reason": f"Matched {matched} existing memories (need >=3 without completion marker)",
        "matched_count": matched,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. Summary extraction
# ═══════════════════════════════════════════════════════════════════════════

def _rf(record: Any, key: str, default: Any = None) -> Any:
    """Safely get a field from a dict or MemoryRecord dataclass."""
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _extract_summary(content: str, max_chars: int = 200) -> str:
    """Extract a short summary from content."""
    # Take first meaningful paragraph
    lines = content.strip().splitlines()
    clean_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    summary = " ".join(clean_lines)[:max_chars]
    if len(content) > max_chars:
        summary += "..."
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# 9. Main record builder
# ═══════════════════════════════════════════════════════════════════════════

def build_memory_record_from_source(
    content: str,
    source_module: str,
    source_path: str = "",
    source_title: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    explicit_level: Optional[str] = None,
    explicit_scope: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a complete memory record from source content.

    Orchestrates all inference functions and returns a record dict
    ready for storage.

    Parameters
    ----------
    content : str
        The text content to be remembered.
    source_module : str
        Which module produced this (paper_reading, claim_support, etc.).
    source_path : str
        File path or identifier of the source.
    source_title : str
        Human-readable title.
    metadata : dict or None
        Additional metadata from the source module.
    explicit_level : str or None
        If set, overrides the inferred memory_level.
    explicit_scope : str or None
        If set, overrides the inferred memory_scope.

    Returns
    -------
    dict
        Memory record.
    """
    meta = deepcopy(metadata) or {}

    # Inference pipeline
    level_info = infer_memory_level(content, source_module, meta)
    memory_level = explicit_level or level_info["memory_level"]

    memory_type = infer_memory_type(source_module, content, meta)
    owner_agent = infer_owner_agent(source_module, memory_type, content)
    scope_info = infer_memory_scope(content, memory_level, owner_agent, source_module, memory_type)
    memory_scope = explicit_scope or scope_info["memory_scope"]
    shared_with = scope_info.get("shared_with", [])
    tags = infer_tags(content, meta)
    summary = _extract_summary(content)

    # Confidence-based importance
    importance = level_info.get("confidence", 0.5)

    fields = {
        "content": content,
        "summary": summary,
        "memory_level": memory_level,
        "memory_scope": memory_scope,
        "memory_type": memory_type,
        "owner_agent": owner_agent,
        "shared_with": shared_with,
        "tags": tags,
        "source_module": source_module,
        "source_path": source_path,
        "source_title": source_title,
        "importance": round(importance, 2),
        "metadata": meta,
    }

    if _SCHEMA_AVAILABLE:
        try:
            return create_memory_record(**fields)
        except Exception:
            pass

    return _fallback_create_record(fields)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Write orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def write_memory_from_source(
    content: str,
    source_module: str,
    source_path: str = "",
    source_title: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    explicit_level: Optional[str] = None,
    explicit_scope: Optional[str] = None,
    existing_memories: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build a memory record and write it to the Memory Store.

    Also checks mid-term promotion if applicable.

    Returns::

        {
            "ok": bool,
            "record": dict,
            "write_result": dict,
            "decision": {
                "level_reason": str,
                "scope_reason": str,
                "owner_agent": str,
                "promoted_from_mid": bool,
            },
        }
    """
    # Infer level (with mid→long promotion check)
    level_info = infer_memory_level(content, source_module, metadata)

    memory_level = explicit_level or level_info["memory_level"]
    promoted = False
    promote_reason = ""

    # Check mid→long promotion
    if memory_level == "mid_term" and existing_memories:
        promo = should_promote_mid_to_long(content, existing_memories)
        if promo["promote"]:
            memory_level = "long_term"
            promoted = True
            promote_reason = promo["reason"]

    # Build record
    record = build_memory_record_from_source(
        content=content,
        source_module=source_module,
        source_path=source_path,
        source_title=source_title,
        metadata=metadata,
        explicit_level=memory_level,
        explicit_scope=explicit_scope,
    )

    # Ensure memory_id is the canonical primary key
    if isinstance(record, dict):
        record = normalize_memory_id(record)

    # Validate scope and auto-correct shared_with
    if _PRIVACY_AVAILABLE:
        try:
            validation = validate_scope_on_write(record)
            if isinstance(validation, dict):
                record = validation.get("record", record)
        except Exception:
            pass  # validation is advisory; never block a write

    # Write to store
    if _STORE_AVAILABLE:
        write_result = append_memory(record)
    else:
        write_result = _fallback_append(record)

    return {
        "ok": write_result.get("ok", False),
        "record": record,
        "write_result": write_result,
        "decision": {
            "level_reason": level_info["reason"] + (f"; promoted: {promote_reason}" if promoted else ""),
            "scope_reason": infer_memory_scope(content, memory_level, _rf(record, "owner_agent", ""), source_module, _rf(record, "memory_type", "")).get("reason", ""),
            "owner_agent": _rf(record, "owner_agent", ""),
            "promoted_from_mid": promoted,
        },
    }
