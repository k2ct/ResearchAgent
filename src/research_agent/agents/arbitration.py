"""
Agent Conflict Resolution & Coordinator Arbitration.

Detects conflicts among multi-agent ``HandoffResult`` outputs and
produces a coordinator-level final arbitration with recommended
wording and next actions.

Pure functions — no side effects, no LangGraph dependency.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# 1. Stance classification
# ═══════════════════════════════════════════════════════════════════════════

_SUPPORT_KEYWORDS: List[str] = [
    "支持", "证明", "表明", "consistent with", "supports", "supported",
    "证实", "验证", "confirmed", "confirm", "validated", "evidence supports",
    "实验结果支持", "理论支持", "provides evidence for",
    "is consistent", "agrees with", "corroborates",
    "finds that", "finding", "shows that", "demonstrates that",
    "in favor of", "supporting",
]

_OPPOSE_KEYWORDS: List[str] = [
    "反驳", "不支持", "contradict", "against",
    "否定", "推翻", "refutes", "disproves", "counters",
    "证据反驳", "does not support", "inconsistent with",
    "challenges", "opposes", "refuted", "contradicts",
    "counter-evidence", "opposing",
]

_UNCERTAIN_KEYWORDS: List[str] = [
    "证据不足", "不明确", "需要进一步实验", "insufficient evidence",
    "尚不明确", "有待验证", "需要更多", "进一步研究",
    "unclear", "inconclusive", "more research needed",
    "preliminary", "mixed results", "cannot determine",
    "当前资料不足", "无法确定", "暂未",
]

_FAILED_KEYWORDS: List[str] = [
    "error", "failed", "无法完成", "失败",
    "timeout", "timed out", "unavailable",
    "no result", "execution error",
]


def classify_result_stance(result_text: str) -> str:
    """
    Classify the stance of a result text.

    Returns one of: ``"support"``, ``"oppose"``, ``"uncertain"``,
    ``"neutral"``, or ``"failed"``.

    Uses keyword scoring — first category with a match wins.
    """
    if not result_text or not result_text.strip():
        return "neutral"

    text_lower = result_text.lower()

    # Check failed first (terminal)
    for kw in _FAILED_KEYWORDS:
        if kw.lower() in text_lower:
            return "failed"

    # Score each stance category
    scores: Dict[str, int] = {"support": 0, "oppose": 0, "uncertain": 0}

    for kw in _SUPPORT_KEYWORDS:
        if kw.lower() in text_lower:
            scores["support"] += 1

    for kw in _OPPOSE_KEYWORDS:
        if kw.lower() in text_lower:
            scores["oppose"] += 1

    for kw in _UNCERTAIN_KEYWORDS:
        if kw.lower() in text_lower:
            scores["uncertain"] += 1

    # Highest score wins (with minimum threshold)
    best = max(scores, key=lambda k: scores[k])
    if scores[best] >= 1:
        return best

    # Default: check if text has any content at all
    if len(result_text.strip()) > 20:
        return "neutral"
    return "neutral"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Conflict detection
# ═══════════════════════════════════════════════════════════════════════════

_CONFIDENCE_GAP_THRESHOLD = 0.4
_SOURCE_OVERLAP_THRESHOLD = 0.3  # < 30% overlap → potential conflict


def detect_agent_conflicts(results: List[Any]) -> Dict[str, Any]:
    """
    Detect conflicts across multiple HandoffResult objects.

    Returns::

        {
            "has_conflict": bool,
            "conflict_types": [str, ...],
            "conflict_summary": str,
            "affected_agents": [str, ...],
        }
    """
    if len(results) < 2:
        return {
            "has_conflict": False,
            "conflict_types": [],
            "conflict_summary": "Only one result — no conflicts possible.",
            "affected_agents": [],
        }

    conflict_types: List[str] = []
    affected: Set[str] = set()
    details: List[str] = []

    # Extract data
    stances: Dict[str, str] = {}
    confidences: Dict[str, float] = {}
    sources_sets: Dict[str, Set[str]] = {}
    memory_sets: Dict[str, Set[str]] = {}

    for r in results:
        agent = _agent(r)
        try:
            stances[agent] = classify_result_stance(_text(r))
        except Exception:
            stances[agent] = "neutral"
        try:
            confidences[agent] = float(_rf(r, "confidence", 0.5))
        except (TypeError, ValueError):
            confidences[agent] = 0.5
        sources_sets[agent] = {
            s.get("path", "") if isinstance(s, dict) else str(s)
            for s in (_rf(r, "sources", []) or [])
        }
        memory_sets[agent] = set(_rf(r, "memory_ids", []) or [])

    agents = list(stances.keys())

    # ── Stance conflict ──────────────────────────────────────────
    unique_stances = set(stances.values())
    if "oppose" in unique_stances or ("support" in unique_stances and "oppose" in unique_stances):
        conflict_types.append("stance_contradiction")
        for a, s in stances.items():
            if s in ("oppose", "support"):
                affected.add(a)
        details.append(f"Conflicting stances: {stances}")

    # ── Support vs uncertain conflict ────────────────────────────
    if "support" in unique_stances and "uncertain" in unique_stances:
        conflict_types.append("support_vs_uncertain")
        for a, s in stances.items():
            if s in ("support", "uncertain"):
                affected.add(a)
        details.append(f"Mixed confidence: some agents support, others uncertain — {stances}")

    # ── Failed results ───────────────────────────────────────────
    failed_agents = [a for a, s in stances.items() if s == "failed"]
    if failed_agents:
        conflict_types.append("agent_failed")
        affected.update(failed_agents)
        details.append(f"Failed agents: {failed_agents}")

    # ── Confidence gap ───────────────────────────────────────────
    if len(confidences) >= 2:
        conf_vals = list(confidences.values())
        if max(conf_vals) - min(conf_vals) > _CONFIDENCE_GAP_THRESHOLD:
            conflict_types.append("confidence_gap")
            affected.update(confidences.keys())
            details.append(
                f"Confidence gap: min={min(conf_vals):.2f}, max={max(conf_vals):.2f}"
            )

    # ── Source overlap ───────────────────────────────────────────
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a1, a2 = agents[i], agents[j]
            s1, s2 = sources_sets.get(a1, set()), sources_sets.get(a2, set())
            if s1 and s2:
                union = s1 | s2
                if union:
                    overlap = len(s1 & s2) / len(union)
                    if overlap < _SOURCE_OVERLAP_THRESHOLD:
                        conflict_types.append("source_divergence")
                        affected.update([a1, a2])
                        details.append(
                            f"Low source overlap between {a1} and {a2}: {overlap:.1%}"
                        )

    # ── Build summary ────────────────────────────────────────────
    unique_types = list(dict.fromkeys(conflict_types))  # dedup, preserve order
    has_conflict = len(unique_types) > 0

    summary = "; ".join(details) if details else "No conflicts detected."

    return {
        "has_conflict": has_conflict,
        "conflict_types": unique_types,
        "conflict_summary": summary,
        "affected_agents": sorted(affected),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. Arbitration
# ═══════════════════════════════════════════════════════════════════════════

def arbitrate_results(
    results: List[Any],
    root_query: str = "",
) -> Dict[str, Any]:
    """
    Produce a coordinator-level final arbitration from handoff results.

    Returns::

        {
            "final_position": "support" | "oppose" | "uncertain" | "mixed",
            "arbitration_text": str,
            "confidence": float,
            "recommended_action": str,
            "conflicts": dict,   # output of detect_agent_conflicts()
        }
    """
    conflicts = detect_agent_conflicts(results)

    # Classify each result
    stances: Dict[str, str] = {}
    confidences: Dict[str, float] = {}
    for r in results:
        agent = _agent(r)
        stances[agent] = classify_result_stance(_text(r))
        try:
            confidences[agent] = float(_rf(r, "confidence", 0.5))
        except (TypeError, ValueError):
            confidences[agent] = 0.5

    support_count = sum(1 for s in stances.values() if s == "support")
    oppose_count = sum(1 for s in stances.values() if s == "oppose")
    uncertain_count = sum(1 for s in stances.values() if s == "uncertain")
    failed_count = sum(1 for s in stances.values() if s == "failed")
    total_meaningful = support_count + oppose_count + uncertain_count

    avg_conf = sum(confidences.values()) / max(len(confidences), 1)

    # ── Determine final position ──────────────────────────────────
    lines: List[str] = []
    final_position = "mixed"
    recommended_action = ""

    if failed_count > 0:
        lines.append(
            f"**Warning**: {failed_count} agent(s) failed to produce results "
            f"({', '.join(a for a, s in stances.items() if s == 'failed')})."
        )

    if total_meaningful == 0:
        final_position = "uncertain"
        lines.append("No agent produced a definitive stance.")
        recommended_action = "Re-run with more specific queries or additional evidence sources."
    elif support_count >= 2 and oppose_count == 0 and uncertain_count == 0:
        final_position = "support"
        lines.append(f"**Consensus**: {support_count} agents support the claim.")
        recommended_action = "Proceed with high confidence. Consider writing up findings."
    elif oppose_count >= 2:
        final_position = "oppose"
        lines.append(f"**Consensus against**: {oppose_count} agents oppose or contradict.")
        recommended_action = "Re-evaluate the claim. Seek additional counter-evidence."
    elif support_count >= 1 and uncertain_count >= 1:
        final_position = "mixed"
        lines.append(
            f"**Mixed signal**: {support_count} agent(s) support, "
            f"{uncertain_count} agent(s) uncertain. "
            f"Use conservative wording — acknowledge the supporting evidence "
            f"but flag the uncertainty."
        )
        recommended_action = (
            "Conservative recommendation: present findings with caveats. "
            "Prioritise additional experiments to resolve uncertainty."
        )
    elif support_count >= 1 and oppose_count >= 1:
        final_position = "mixed"
        lines.append(
            f"**Contradiction**: {support_count} support vs {oppose_count} oppose. "
            f"Evidence is conflicting — do not draw a strong conclusion."
        )
        recommended_action = "Investigate the conflicting evidence before making claims."
    elif uncertain_count >= 2:
        final_position = "uncertain"
        lines.append("Multiple agents report insufficient evidence.")
        recommended_action = "Gather more data or run targeted experiments."
    else:
        final_position = "mixed"
        lines.append(f"Aggregated stance is mixed (support={support_count}, oppose={oppose_count}, uncertain={uncertain_count}).")
        recommended_action = "Review individual agent outputs for specific action items."

    # Adjust confidence
    if uncertain_count > 0:
        avg_conf *= 0.7
    if failed_count > 0:
        avg_conf *= 0.8
    if oppose_count > 0 and support_count > 0:
        avg_conf *= 0.6
    final_confidence = round(min(avg_conf, 1.0), 2)

    # Stance summary
    stance_summary = ", ".join(
        f"{a}: {s}" for a, s in sorted(stances.items())
    )

    arbitration_text = "\n\n".join([
        f"**Stance summary**: {stance_summary}",
        f"**Final position**: {final_position}",
        f"**Confidence**: {final_confidence}",
        "",
        "\n\n".join(lines),
    ])

    return {
        "final_position": final_position,
        "arbitration_text": arbitration_text,
        "confidence": final_confidence,
        "recommended_action": recommended_action,
        "conflicts": conflicts,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Coordinator final summary (Markdown)
# ═══════════════════════════════════════════════════════════════════════════

def build_coordinator_final_summary(
    arbitration: Dict[str, Any],
    results: List[Any],
    root_query: str = "",
) -> str:
    """
    Generate a structured Markdown coordinator final arbitration document.
    """
    conflicts = arbitration.get("conflicts", {})
    lines: List[str] = []
    lines.append("# Coordinator Final Arbitration\n")

    # ── Final Position ────────────────────────────────────────────
    lines.append("## 1. Final Position\n")
    lines.append(f"**Position**: {arbitration.get('final_position', 'unknown')}")
    lines.append(f"**Confidence**: {arbitration.get('confidence', 0.0)}")
    lines.append(f"**Query**: {root_query}" if root_query else "")
    lines.append("")

    # ── Evidence Agreement ────────────────────────────────────────
    lines.append("## 2. Evidence Agreement\n")
    for r in results:
        agent = _agent(r)
        stance = classify_result_stance(_text(r))
        conf = _rf(r, "confidence", 0.5)
        sources_n = len(_rf(r, "sources", []) or [])
        lines.append(f"- **{agent}**: stance={stance}, confidence={conf:.2f}, sources={sources_n}")
    lines.append("")

    # ── Conflicts ─────────────────────────────────────────────────
    lines.append("## 3. Conflicts\n")
    if conflicts.get("has_conflict"):
        lines.append(f"**Conflict types**: {', '.join(conflicts.get('conflict_types', []))}")
        lines.append(f"**Affected agents**: {', '.join(conflicts.get('affected_agents', []))}")
        lines.append(f"**Summary**: {conflicts.get('conflict_summary', '')}")
    else:
        lines.append("No conflicts detected — all agents agree.")
    lines.append("")

    # ── Recommended Wording ───────────────────────────────────────
    lines.append("## 4. Recommended Wording\n")
    fp = arbitration.get("final_position", "mixed")
    if fp == "support":
        wording = (
            "Based on multi-agent analysis, the evidence supports the claim. "
            "Key findings are consistent across paper review, experimental data, "
            "and progress tracking."
        )
    elif fp == "oppose":
        wording = (
            "Multi-agent analysis does not support the claim at this time. "
            "Multiple agents found contradictory or insufficient evidence."
        )
    elif fp == "uncertain":
        wording = (
            "The available evidence is insufficient to draw a firm conclusion. "
            "Agents report mixed or inconclusive results. Further investigation "
            "is recommended before making claims."
        )
    else:  # mixed
        wording = (
            "The evidence presents a mixed picture. Some agents provide supporting "
            "evidence, while others report uncertainty or limitations. A conservative "
            "interpretation is warranted: acknowledge the positive signals but clearly "
            "state the open questions and limitations."
        )
    lines.append(wording)
    lines.append("")

    # ── Next Actions ──────────────────────────────────────────────
    lines.append("## 5. Next Actions\n")
    lines.append(f"- **Recommended**: {arbitration.get('recommended_action', 'Review findings.')}")
    lines.append("- Review individual agent outputs for detailed evidence and sources.")
    lines.append("- If uncertain, run targeted experiments or expand literature search.")
    if conflicts.get("has_conflict"):
        lines.append("- Resolve conflicts before making claims in reports or presentations.")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _agent(r: Any) -> str:
    """Extract agent name from a HandoffResult (dataclass or dict)."""
    if isinstance(r, dict):
        return r.get("to_agent", r.get("from_agent", "unknown"))
    return getattr(r, "to_agent", getattr(r, "from_agent", "unknown"))


def _text(r: Any) -> str:
    """Extract result_text from a HandoffResult."""
    if isinstance(r, dict):
        return r.get("result_text", "")
    return getattr(r, "result_text", "")


def _rf(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
