"""
Claim Support Retrieval v1.

Given a scientific claim, retrieves structured evidence from the local RAG
knowledge base, grouped by purpose: theory, experiment, related work,
limitations, and notes.

Independent module — does not modify LangGraph workflow, nodes, or app.
"""

import re
from typing import Dict, List, Optional, Set

from research_agent.rag.retriever import (
    retrieve_documents,
    extract_sources_from_docs,
    format_retrieved_docs,
    document_to_dict,
)


# ── 1. Claim Intent Classification ────────────────────────────────

# Keyword patterns for each claim type
_CLAIM_TYPE_PATTERNS = {
    "theoretical_claim": [
        # Chinese
        "假设", "理论", "机制", "归因", "因果", "诱发", "导致", "促进",
        "表征", "编码", "先验", "语义", "共现", "统计", "概率",
        # English
        "hypothesis", "theory", "mechanism", "causal", "representation",
        "encoding", "co-occurrence", "prior", "semantic", "statistical",
    ],
    "empirical_claim": [
        "实验", "结果", "指标", "精度", "召回", "提升", "下降", "F1",
        "benchmark", "数据集", "验证", "观测", "测量", "量化",
        "experiment", "result", "benchmark", "metric", "precision",
        "recall", "accuracy", "observation", "measurement", "dataset",
    ],
    "method_claim": [
        "方法", "框架", "架构", "流程", "pipeline", "模块", "组件",
        "设计", "策略", "采样", "聚合", "融合", "训练", "微调",
        "method", "framework", "architecture", "pipeline", "module",
        "design", "strategy", "training", "fine-tuning", "fusion",
    ],
    "related_work_claim": [
        "前人", "已有", "文献", "相关工作", "之前", "传统", "经典",
        "对比", "差异", "不同", "创新", "贡献", "填补", "首次",
        "related work", "previous", "existing", "literature", "novel",
        "contribution", "comparison", "difference", "state-of-the-art",
    ],
}

_CLAIM_TYPE_TO_TASK_TYPES = {
    "theoretical_claim": ["paper_question", "general"],
    "empirical_claim": ["experiment_analysis", "dataset_recommendation"],
    "method_claim": ["paper_question", "general", "experiment_analysis"],
    "related_work_claim": ["paper_question", "general"],
    "mixed_claim": ["paper_question", "experiment_analysis", "general"],
}


def classify_claim_intent(claim: str) -> Dict:
    """
    Classify a scientific claim into one of five types
    based on keyword matching.

    Returns:
        {"claim_type": str, "keywords": list, "suggested_task_types": list}
    """
    claim_lower = claim.lower()
    scores = {}

    for claim_type, patterns in _CLAIM_TYPE_PATTERNS.items():
        score = 0
        matched_keywords = []
        for pattern in patterns:
            if pattern in claim_lower:
                score += 1
                matched_keywords.append(pattern)
        if score > 0:
            scores[claim_type] = (score, matched_keywords)

    if not scores:
        return {
            "claim_type": "mixed_claim",
            "keywords": [],
            "suggested_task_types": _CLAIM_TYPE_TO_TASK_TYPES["mixed_claim"],
        }

    # If multiple types matched, check if one is clearly dominant
    sorted_types = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
    top_type, (top_score, top_keywords) = sorted_types[0]

    # If top score is >= 2x second, it's clearly one type
    if len(sorted_types) >= 2:
        second_score = sorted_types[1][1][0]
        if top_score < second_score * 2 and top_score <= 3:
            # Close scores — mixed
            all_keywords = []
            for _, (_, kws) in sorted_types:
                all_keywords.extend(kws)
            return {
                "claim_type": "mixed_claim",
                "keywords": list(set(all_keywords)),
                "suggested_task_types": _CLAIM_TYPE_TO_TASK_TYPES["mixed_claim"],
            }

    return {
        "claim_type": top_type,
        "keywords": top_keywords,
        "suggested_task_types": _CLAIM_TYPE_TO_TASK_TYPES.get(
            top_type, ["paper_question", "general"]
        ),
    }


# ── 2. Query Decomposition ────────────────────────────────────────

# Purpose → (task_type, query template)
_PURPOSE_TEMPLATES = [
    {
        "purpose": "theory",
        "task_type": "paper_question",
        "label": "理论依据",
        "prefix": "理论解释",
    },
    {
        "purpose": "experiment",
        "task_type": "experiment_analysis",
        "label": "实验依据",
        "prefix": "实验证据",
    },
    {
        "purpose": "related_work",
        "task_type": "paper_question",
        "label": "相关论文",
        "prefix": "相关研究",
    },
    {
        "purpose": "limitation",
        "task_type": "general",
        "label": "潜在反例或限制",
        "prefix": "局限性",
    },
    {
        "purpose": "notes",
        "task_type": "general",
        "label": "可引用表述",
        "prefix": "相关笔记",
    },
]


def _extract_core_terms(claim: str) -> List[str]:
    """
    Extract core noun phrases and English identifiers from the claim.

    These become the anchor terms carried into each retrieval query.
    """
    # English / code identifiers
    en_pattern = r'[a-zA-Z0-9][a-zA-Z0-9_\-]*[a-zA-Z0-9]|[A-Z]{2,}'
    en_matches = re.findall(en_pattern, claim)
    # Filter stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "of",
                 "in", "on", "at", "to", "for", "with", "and", "or", "not",
                 "can", "may", "should", "could", "would", "it", "this",
                 "that", "we", "no", "has", "have", "its"}
    core_terms = [m for m in en_matches if m.lower() not in stopwords and len(m) >= 2]

    # Chinese noun phrases (3+ chars)
    cn_pattern = r'[一-鿿]{3,8}'
    cn_matches = re.findall(cn_pattern, claim)
    core_terms.extend(cn_matches)

    return core_terms


def build_claim_queries(claim: str) -> List[Dict]:
    """
    Decompose a scientific claim into multiple retrieval queries.

    Each query targets a different evidence purpose (theory, experiment,
    related work, limitation, notes).

    Returns:
        [{"query": str, "purpose": str, "task_type": str, "label": str}, ...]
    """
    core_terms = _extract_core_terms(claim)
    core_text = " ".join(core_terms) if core_terms else claim

    queries = [
        {
            "query": claim,
            "purpose": "theory",
            "task_type": "paper_question",
            "label": "理论依据",
        },
    ]

    for tmpl in _PURPOSE_TEMPLATES:
        if tmpl["purpose"] == "theory":
            continue  # already added original claim

        query_text = f"{tmpl['prefix']} {core_text} {claim[:60]}"
        queries.append({
            "query": query_text.strip(),
            "purpose": tmpl["purpose"],
            "task_type": tmpl["task_type"],
            "label": tmpl["label"],
        })

    return queries


# ── 3. Evidence Retrieval ─────────────────────────────────────────


def retrieve_claim_evidence(
    claim: str,
    top_k_per_query: int = 3,
) -> List[Dict]:
    """
    Retrieve evidence for a claim by running multiple targeted queries.

    Uses the existing retriever (vector or hybrid, depending on
    RAG_RETRIEVAL_MODE env var). Deduplicates results across queries.

    Returns a list of evidence dicts, each with:
        content, metadata, purpose, query, path, source_type, section_title
    """
    queries = build_claim_queries(claim)
    seen_keys: Set[str] = set()
    evidence_list: List[Dict] = []

    for q in queries:
        try:
            docs = retrieve_documents(
                query=q["query"],
                task_type=q["task_type"],
                top_k=top_k_per_query,
                retrieval_mode=None,  # use env var
            )
        except Exception:
            continue

        for doc in docs:
            meta = doc.metadata
            path = meta.get("path", "")
            content_prefix = doc.page_content[:120]
            dedup_key = f"{path}::{content_prefix}"

            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            evidence_list.append({
                "content": doc.page_content,
                "metadata": dict(meta),
                "purpose": q["purpose"],
                "query": q["query"],
                "path": path,
                "source_type": meta.get("source_type", "unknown"),
                "section_title": meta.get("section_title", ""),
            })

    return evidence_list


# ── 4. Group Evidence by Purpose ──────────────────────────────────


def group_evidence_by_purpose(
    evidence: List[Dict],
) -> Dict[str, List[Dict]]:
    """
    Group evidence items by their purpose field.

    Returns:
        {"theory": [...], "experiment": [...], "related_work": [...],
         "limitation": [...], "notes": [...]}
    """
    groups: Dict[str, List[Dict]] = {
        "theory": [],
        "experiment": [],
        "related_work": [],
        "limitation": [],
        "notes": [],
    }

    for item in evidence:
        purpose = item.get("purpose", "notes")
        if purpose in groups:
            groups[purpose].append(item)
        else:
            groups["notes"].append(item)

    return groups


# ── 5. Report Builder ─────────────────────────────────────────────


def _format_evidence_item(item: Dict, idx: int, max_chars: int = 300) -> str:
    """Format a single evidence item for the report."""
    path = item.get("path", "?")
    section_title = item.get("section_title", "")
    source_type = item.get("source_type", "?")
    content_preview = item["content"][:max_chars].replace("\n", " ")

    lines = [
        f"**证据 {idx}**",
        f"- 来源: `{path}`",
        f"- 类型: `{source_type}`",
    ]
    if section_title:
        lines.append(f"- 章节: {section_title}")
    lines.append(f"- 内容: {content_preview}...")
    lines.append("")

    return "\n".join(lines)


def _generate_wording_suggestions(
    claim: str,
    grouped: Dict[str, List[Dict]],
) -> List[str]:
    """
    Generate 2-3 academic wording suggestions based on retrieved evidence.

    Heuristic rules — v1 does not call LLM.
    """
    suggestions = []

    # Suggestion 1: claim + theory evidence
    theory_items = grouped.get("theory", [])
    if theory_items and len(theory_items) >= 1:
        source = theory_items[0].get("path", "knowledge base")
        source_short = source.split("/")[-1].replace(".md", "") if "/" in source else source
        suggestions.append(
            f"已有研究表明，{claim}（参见 {source_short}）。"
        )

    # Suggestion 2: claim + experiment support
    exp_items = grouped.get("experiment", [])
    if exp_items:
        suggestions.append(
            f"实验结果进一步支持了这一观点：{claim}。"
            f"相关分析表明该论点具有可验证的实证基础。"
        )
    elif theory_items and len(theory_items) >= 2:
        suggestions.append(
            f"从理论层面看，{claim}。"
            f"现有文献为该论点提供了概念支撑。"
        )

    # Suggestion 3: cautious / limitation-aware wording
    lim_items = grouped.get("limitation", [])
    if lim_items:
        suggestions.append(
            f"需要指出的是，{claim}这一结论仍存在一定限制，"
            f"具体条件和边界效应有待进一步探索。"
        )
    else:
        suggestions.append(
            f"当前本地知识库未检索到直接反例，{claim}的结论"
            f"在当前证据范围内是合理的，但仍建议通过更广泛的文献调研加以验证。"
        )

    # Ensure at least 2, at most 3
    if len(suggestions) == 1:
        suggestions.append(
            f"综上，{claim}。该观点可在论文 Introduction 或 Discussion 中"
            f"作为核心论点之一呈现。"
        )

    return suggestions[:3]


def build_claim_support_report(
    claim: str,
    grouped_evidence: Dict[str, List[Dict]],
    use_llm: bool = False,
) -> str:
    """
    Build a structured Markdown report from grouped evidence.

    v1 uses template-based generation. Set use_llm=True for LLM enhancement
    in a future version.
    """
    lines = ["# Claim Support Report", "", f"## 1. Claim", "", claim, ""]

    # ── 2. Theoretical Support ──
    lines.append("## 2. Theoretical Support")
    lines.append("")
    theory = grouped_evidence.get("theory", [])
    if theory:
        for i, item in enumerate(theory[:3], start=1):
            lines.append(_format_evidence_item(item, i))
    else:
        lines.append("当前知识库未检索到直接的理论依据。")
        lines.append("")

    # ── 3. Empirical / Experimental Support ──
    lines.append("## 3. Empirical / Experimental Support")
    lines.append("")
    experiment = grouped_evidence.get("experiment", [])
    if experiment:
        for i, item in enumerate(experiment[:3], start=1):
            lines.append(_format_evidence_item(item, i))
    else:
        lines.append("当前知识库未检索到直接的实验依据。")
        lines.append("")

    # ── 4. Related Work ──
    lines.append("## 4. Related Work")
    lines.append("")
    related = grouped_evidence.get("related_work", [])
    if related:
        for i, item in enumerate(related[:3], start=1):
            lines.append(_format_evidence_item(item, i))
    else:
        lines.append("当前知识库未检索到直接的相关论文或笔记。")
        lines.append("")

    # ── 5. Potential Limitations or Counter-evidence ──
    lines.append("## 5. Potential Limitations or Counter-evidence")
    lines.append("")
    limitation = grouped_evidence.get("limitation", [])
    if limitation:
        for i, item in enumerate(limitation[:3], start=1):
            lines.append(_format_evidence_item(item, i))
    else:
        lines.append(
            "当前知识库未检索到明显反例。建议通过更广泛的文献调研"
            "（如 Google Scholar、Semantic Scholar）验证该论点的边界条件和适用范围。"
        )
        lines.append("")

    # ── 6. Suggested Academic Wording ──
    lines.append("## 6. Suggested Academic Wording")
    lines.append("")
    suggestions = _generate_wording_suggestions(claim, grouped_evidence)
    for i, s in enumerate(suggestions, start=1):
        lines.append(f"{i}. {s}")
        lines.append("")

    return "\n".join(lines)


# ── 6. Main Entry Point ───────────────────────────────────────────


def generate_claim_support(
    claim: str,
    top_k_per_query: int = 3,
    use_llm: bool = False,
) -> Dict:
    """
    Main entry point for Claim Support Retrieval.

    Args:
        claim: The scientific claim to find support for.
        top_k_per_query: Number of results per sub-query.
        use_llm: Whether to use LLM for report enhancement (v2).

    Returns:
        {
            "claim": str,
            "claim_type": str,
            "queries": list,
            "evidence_count": int,
            "grouped_evidence": dict,
            "report": str,
            "sources": list,
        }
    """
    # 1. Classify claim
    intent = classify_claim_intent(claim)

    # 2. Build queries
    queries = build_claim_queries(claim)

    # 3. Retrieve evidence
    evidence = retrieve_claim_evidence(claim, top_k_per_query=top_k_per_query)

    # 4. Group by purpose
    grouped = group_evidence_by_purpose(evidence)

    # 5. Build report
    report = build_claim_support_report(claim, grouped, use_llm=use_llm)

    # 6. LLM enhancement (if requested and available)
    used_llm = False
    llm_error = ""
    if use_llm:
        try:
            from research_agent.llm.enhancers import enhance_claim_support_report
            enhanced = enhance_claim_support_report(claim, grouped, report)
            if enhanced["used_llm"]:
                report = enhanced["text"]
                used_llm = True
            if enhanced.get("error"):
                llm_error = enhanced["error"]
        except Exception as e:
            llm_error = f"{type(e).__name__}: {e}"

    # 7. Extract deduplicated sources
    sources = []
    seen_paths = set()
    for item in evidence:
        path = item.get("path", "")
        if path not in seen_paths:
            seen_paths.add(path)
            sources.append({
                "path": path,
                "source_type": item.get("source_type", "unknown"),
                "section_title": item.get("section_title", ""),
            })

    return {
        "claim": claim,
        "claim_type": intent["claim_type"],
        "queries": queries,
        "evidence_count": len(evidence),
        "grouped_evidence": grouped,
        "report": report,
        "sources": sources,
        "used_llm": used_llm,
        "llm_error": llm_error,
    }
