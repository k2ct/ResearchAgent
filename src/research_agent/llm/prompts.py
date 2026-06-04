"""
LLM prompt builders for the ResearchAgent Enhancement Layer.

Each builder returns a list of LangChain messages (SystemMessage + HumanMessage)
suitable for ``invoke_llm_with_fallback()``.

All system prompts enforce:
- Use ONLY provided evidence / content
- Do NOT fabricate paper conclusions, experimental results, or file paths
- When information is insufficient, state it clearly
"""

from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Shared system prompt prefix
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM_RULES = (
    "你必须严格基于提供的资料和证据生成内容。\n"
    "重要规则：\n"
    "1. 只能使用下面 Evidence / Content 中明确出现的信息。\n"
    "2. 不要编造论文结论、实验数值、数据集属性或不存在的文件路径。\n"
    "3. 不要编造实验结果。\n"
    "4. 如果资料不足，请明确写「当前资料不足以支持该结论」。\n"
    "5. 保留清晰的小标题和层次结构。\n"
    "6. 不要输出虚假的引用编号。\n"
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Claim Support
# ═══════════════════════════════════════════════════════════════════════════

def build_claim_support_llm_messages(
    claim: str,
    grouped_evidence: Dict[str, List[Dict]],
    rule_based_report: str,
) -> List:
    """
    Build messages for LLM-enhanced claim support report.

    Args:
        claim: The original scientific claim.
        grouped_evidence: Evidence grouped by purpose (theory/experiment/...).
        rule_based_report: The rule-based report for reference.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # Summarise evidence for the prompt
    evidence_parts: List[str] = []
    for purpose, items in grouped_evidence.items():
        if not items:
            continue
        evidence_parts.append(f"\n### {purpose}")
        for i, item in enumerate(items[:3], start=1):
            path = item.get("path", "?")
            content_preview = item.get("content", "")[:500]
            evidence_parts.append(
                f"[{purpose}-{i}] source={path}\n{content_preview}"
            )

    evidence_text = "\n".join(evidence_parts) if evidence_parts else "(无证据)"

    system = SystemMessage(content=(
        f"{_SYSTEM_RULES}\n"
        "你是科研论证助手。你的任务是基于提供的证据，为学术论点生成结构化的论证支持报告。\n"
        "输出格式：\n"
        "# Claim Support Report\n"
        "## Claim\n"
        "## Evidence-backed Support\n"
        "## Related Work\n"
        "## Empirical Evidence\n"
        "## Limitations\n"
        "## Suggested Academic Wording\n"
        "## Sources Used\n"
    ))

    user = HumanMessage(content=(
        f"## 论点 (Claim)\n{claim}\n\n"
        f"## 检索到的证据 (Evidence)\n{evidence_text}\n\n"
        f"## 规则版报告 (Reference)\n{rule_based_report[:2000]}\n\n"
        "请基于证据生成结构化的 Claim Support Report。"
        "Evidence 不足的类别请明确标注。"
    ))

    return [system, user]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Paper Reading
# ═══════════════════════════════════════════════════════════════════════════

def build_paper_reading_llm_messages(
    paper_metadata: Dict[str, Any],
    sections: Dict[str, str],
    rule_based_note: str,
) -> List:
    """
    Build messages for LLM-enhanced paper reading note.

    Args:
        paper_metadata: Extracted metadata (title, year, venue, ...).
        sections: Detected paper sections (abstract, method, ...).
        rule_based_note: The rule-based reading note for reference.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # Build section content for the prompt
    section_text_parts: List[str] = []
    for key, text in sections.items():
        if text.strip():
            truncated = text[:2000] if len(text) > 2000 else text
            section_text_parts.append(f"### {key}\n{truncated}")

    section_text = "\n\n".join(section_text_parts) if section_text_parts else "(无内容)"

    title = paper_metadata.get("title", "Unknown")
    year = paper_metadata.get("year", "?")
    venue = paper_metadata.get("venue", "unknown")

    system = SystemMessage(content=(
        f"{_SYSTEM_RULES}\n"
        "你是论文阅读助手。你的任务是基于提供的论文文本，生成结构化的论文阅读笔记。\n"
        "输出格式：\n"
        "# Paper Reading Note\n"
        "## Basic Information\n"
        "## Research Background\n"
        "## Research Problem\n"
        "## Method\n"
        "## Experiments\n"
        "## Key Findings\n"
        "## Contributions\n"
        "## Limitations\n"
        "## Relevance to My Research\n"
        "## PPT Outline\n"
        "## Follow-up Questions\n"
    ))

    user = HumanMessage(content=(
        f"## 论文基本信息\n"
        f"- Title: {title}\n"
        f"- Year: {year}\n"
        f"- Venue: {venue}\n\n"
        f"## 论文章节内容 (Paper Sections)\n"
        f"{section_text}\n\n"
        f"## 规则版阅读笔记 (Reference)\n"
        f"{rule_based_note[:2000]}\n\n"
        "请基于论文章节内容生成结构化的 Paper Reading Note。"
        "资料不足的部分请明确标注「当前文档未明确提供」。"
    ))

    return [system, user]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Progress Memory
# ═══════════════════════════════════════════════════════════════════════════

def build_progress_memory_llm_messages(
    slide_metadata: Dict[str, Any],
    slides: List[Dict[str, Any]],
    topics: Dict[str, Any],
    rule_based_memory: str,
) -> List:
    """
    Build messages for LLM-enhanced research progress memory.

    Args:
        slide_metadata: Metadata from the slide document.
        slides: Parsed slides (title, bullets, content).
        topics: Inferred topics (research_questions, experiments, ...).
        rule_based_memory: The rule-based progress memory for reference.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    # Build slide summary
    slide_parts: List[str] = []
    for s in slides[:10]:  # limit to 10 slides
        num = s.get("slide_number", "?")
        title = s.get("title", "(untitled)")
        bullets = s.get("bullets", [])
        bullet_text = "\n".join(f"  - {b}" for b in bullets[:6])
        slide_parts.append(f"Slide {num}: {title}\n{bullet_text}")

    slide_text = "\n\n".join(slide_parts) if slide_parts else "(无 slides)"

    # Topics summary
    topic_text = "\n".join(
        f"- {k}: {', '.join(v[:5]) if v else '(none)'}"
        for k, v in topics.items()
        if k != "keywords"
    )

    title = slide_metadata.get("title", "Untitled")

    system = SystemMessage(content=(
        f"{_SYSTEM_RULES}\n"
        "你是科研进展整理助手。你的任务是基于提供的 PPT/Slide 内容，"
        "生成结构化的科研进展记忆文档。\n"
        "不要添加 PPT 中不存在的内容。\n"
        "输出格式：\n"
        "# Research Progress Memory\n"
        "## Presentation Summary\n"
        "## Research Questions\n"
        "## Completed Work\n"
        "## Experiments and Results\n"
        "## Findings\n"
        "## Issues\n"
        "## Next Steps\n"
        "## Long-term Memory Records\n"
    ))

    user = HumanMessage(content=(
        f"## 演示文稿标题\n{title}\n\n"
        f"## Slide 内容\n{slide_text}\n\n"
        f"## 启发式推断的主题\n{topic_text}\n\n"
        f"## 规则版 Progress Memory (Reference)\n{rule_based_memory[:2000]}\n\n"
        "请基于 Slide 内容生成结构化的 Research Progress Memory。"
        "PPT 中未提供的部分请明确标注。"
    ))

    return [system, user]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Report Polish
# ═══════════════════════════════════════════════════════════════════════════

def build_report_polish_llm_messages(
    report_text: str,
    sources: List[Dict[str, Any]],
    style: str = "group_meeting",
) -> List:
    """
    Build messages for polishing an existing report (from Report Writer).

    Args:
        report_text: The existing report text (rule-based or previous LLM output).
        sources: Source documents used in the report.
        style: Presentation style (group_meeting / ppt_slide / summary).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    sources_text = "\n".join(
        f"- {s.get('path', '?')} ({s.get('source_type', '?')})"
        for s in sources[:10]
    ) if sources else "(无 sources)"

    style_hints = {
        "group_meeting": "组会汇报讲稿风格：口语化但不失专业，结构清晰，适合口头讲解。",
        "ppt_slide": "PPT 页面文案风格：每页包含标题和 3-5 个要点，语言简洁精炼。",
        "summary": "摘要风格：300-500 字，强调任务目标、核心依据和结论边界。",
    }
    style_hint = style_hints.get(style, style_hints["group_meeting"])

    system = SystemMessage(content=(
        f"{_SYSTEM_RULES}\n"
        "你是科研汇报润色助手。你的任务是优化已有报告的表达，"
        "使其更自然、更适合科研组会或 PPT 展示。\n"
        "核心约束：保留所有事实信息，不添加原文中没有的内容。\n"
        "可以优化的方面：段落流畅度、小标题层次、语言精炼度。\n"
    ))

    user = HumanMessage(content=(
        f"## 报告风格要求\n{style_hint}\n\n"
        f"## 原始报告\n{report_text[:4000]}\n\n"
        f"## Sources\n{sources_text}\n\n"
        "请润色以上报告，保留所有事实，仅优化表达。"
    ))

    return [system, user]
