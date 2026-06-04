"""
LLM enhancers for ResearchAgent research modules.

Each enhancer:
1. Accepts a rule-based output + supporting data.
2. If LLM is available, calls it to produce a more natural output.
3. If LLM is unavailable, returns the rule-based output unchanged.
4. Always returns a standardised dict with ``used_llm`` and ``text``.

Usage from other modules::

    from research_agent.llm.enhancers import enhance_claim_support_report

    result = enhance_claim_support_report(claim, grouped_evidence, rule_based_report)
    final_report = result["text"]  # always works, LLM or not
"""

from typing import Any, Dict, List

from .client import invoke_llm_with_fallback
from .prompts import (
    build_claim_support_llm_messages,
    build_paper_reading_llm_messages,
    build_progress_memory_llm_messages,
    build_report_polish_llm_messages,
)


# ── 1. Claim Support Enhancement ─────────────────────────────────────


def enhance_claim_support_report(
    claim: str,
    grouped_evidence: Dict[str, List[Dict]],
    rule_based_report: str,
) -> Dict[str, Any]:
    """
    Enhance a Claim Support Report with LLM.

    Args:
        claim: The original scientific claim.
        grouped_evidence: Evidence grouped by purpose.
        rule_based_report: The pre-built rule-based report.

    Returns::

        {"text": str, "used_llm": bool, "error": str}
    """
    messages = build_claim_support_llm_messages(
        claim, grouped_evidence, rule_based_report
    )
    result = invoke_llm_with_fallback(
        messages=messages,
        fallback_text=rule_based_report,
        feature_name="claim_support",
    )
    return {
        "text": result["text"],
        "used_llm": result["used_llm"],
        "error": result["error"],
    }


# ── 2. Paper Reading Enhancement ─────────────────────────────────────


def enhance_paper_reading_note(
    paper_metadata: Dict[str, Any],
    sections: Dict[str, str],
    rule_based_note: str,
) -> Dict[str, Any]:
    """
    Enhance a Paper Reading Note with LLM.

    Args:
        paper_metadata: Extracted paper metadata (title, year, etc.).
        sections: Detected paper sections.
        rule_based_note: The pre-built rule-based reading note.

    Returns::

        {"text": str, "used_llm": bool, "error": str}
    """
    messages = build_paper_reading_llm_messages(
        paper_metadata, sections, rule_based_note
    )
    result = invoke_llm_with_fallback(
        messages=messages,
        fallback_text=rule_based_note,
        feature_name="paper_reading",
    )
    return {
        "text": result["text"],
        "used_llm": result["used_llm"],
        "error": result["error"],
    }


# ── 3. Progress Memory Enhancement ───────────────────────────────────


def enhance_progress_memory(
    slide_metadata: Dict[str, Any],
    slides: List[Dict[str, Any]],
    topics: Dict[str, Any],
    rule_based_memory: str,
) -> Dict[str, Any]:
    """
    Enhance a Research Progress Memory with LLM.

    Args:
        slide_metadata: Slide document metadata.
        slides: Parsed slides.
        topics: Inferred topics.
        rule_based_memory: The pre-built rule-based progress memory.

    Returns::

        {"text": str, "used_llm": bool, "error": str}
    """
    messages = build_progress_memory_llm_messages(
        slide_metadata, slides, topics, rule_based_memory
    )
    result = invoke_llm_with_fallback(
        messages=messages,
        fallback_text=rule_based_memory,
        feature_name="progress_memory",
    )
    return {
        "text": result["text"],
        "used_llm": result["used_llm"],
        "error": result["error"],
    }


# ── 4. Report Polish Enhancement ─────────────────────────────────────


def enhance_report_text(
    report_text: str,
    sources: List[Dict[str, Any]],
    style: str = "group_meeting",
) -> Dict[str, Any]:
    """
    Polish an existing report with LLM.

    Args:
        report_text: The report to polish.
        sources: Source documents used.
        style: Presentation style (group_meeting / ppt_slide / summary).

    Returns::

        {"text": str, "used_llm": bool, "error": str}
    """
    messages = build_report_polish_llm_messages(report_text, sources, style)
    result = invoke_llm_with_fallback(
        messages=messages,
        fallback_text=report_text,
        feature_name="report_polish",
    )
    return {
        "text": result["text"],
        "used_llm": result["used_llm"],
        "error": result["error"],
    }
