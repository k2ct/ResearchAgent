"""
Heuristic Reranker v1 for Hybrid RAG.

Re-ranks fused retrieval candidates using lightweight heuristic features:
- Metadata field matching (title, path, dataset, run_tag, section_title)
- Content token overlap
- Exact phrase matching
- Section title boosting

No ML model, no cross-encoder, no extra dependencies.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document

# Reuse the tokenizer from hybrid_retriever
from .hybrid_retriever import tokenize_text


# ── 1. Query Term Extraction ──────────────────────────────────────


def extract_query_terms(query: str) -> Dict:
    """
    Extract structured query information for reranking.

    Returns a dict with:
    - tokens: list of lowercase tokens
    - token_set: set of tokens for fast lookup
    - phrases: list of noun phrases (>= 3 chars, e.g. run_tag, dataset names)
    - original: the original query string
    """
    tokens = tokenize_text(query)
    token_set = set(tokens)

    # Extract noun phrases (run_tags, dataset names, paper titles, etc.)
    phrases = _extract_key_phrases(query)

    return {
        "tokens": tokens,
        "token_set": token_set,
        "phrases": phrases,
        "original": query,
    }


def _extract_key_phrases(text: str) -> List[str]:
    """
    Extract key phrases from text — alphanumeric sequences with underscores/hyphens,
    plus CJK character sequences of length >= 2.

    These often correspond to run_tags (coco_val_n300_g1),
    dataset names (OpenImages-MIAP), and paper titles.
    """
    phrases: List[str] = []

    # English/key phrases: alphanumeric with _/-
    pattern = r'[a-zA-Z0-9][a-zA-Z0-9_\-]*[a-zA-Z0-9]|[a-zA-Z0-9]'
    matches = re.findall(pattern, text)
    for m in matches:
        if len(m) >= 3:
            phrases.append(m.lower())

    # CJK phrases: continuous CJK characters, length >= 2
    cjk_pattern = r'[一-鿿㐀-䶿]{2,}'
    cjk_matches = re.findall(cjk_pattern, text)
    phrases.extend(cjk_matches)

    return phrases


# ── 2. Metadata Match Score ───────────────────────────────────────


# Metadata fields checked, with bonus weights
_METADATA_FIELDS = {
    "title": 4.0,
    "run_tag": 5.0,       # highest bonus: precise experiment identifier
    "dataset": 4.0,
    "section_title": 3.0,
    "section_path": 2.0,
    "source_type": 1.0,
    "path": 2.0,
    "topic": 1.5,
    "tags": 1.0,
}


def compute_metadata_match_score(query: str, doc: Document) -> float:
    """
    Score a document based on how well query terms match its metadata fields.

    Higher weights for run_tag, dataset, title because these are
    strong relevance signals.
    """
    metadata = doc.metadata
    query_lower = query.lower()
    query_terms = extract_query_terms(query)
    query_tokens = query_terms["token_set"]
    query_phrases = query_terms["phrases"]

    score = 0.0

    for field, weight in _METADATA_FIELDS.items():
        value = metadata.get(field, "")
        if not value:
            continue

        value_str = str(value).lower()

        # Exact phrase match in field → full weight
        for phrase in query_phrases:
            if phrase in value_str:
                score += weight
                break  # one phrase match per field is enough

        # Token-level match → partial weight
        if not any(p in value_str for p in query_phrases):
            field_tokens = set(tokenize_text(value_str))
            overlap = query_tokens & field_tokens
            if overlap:
                score += weight * 0.4 * min(len(overlap), 3) / 3.0

    return score


# ── 3. Content Match Score ────────────────────────────────────────


def compute_content_match_score(query: str, doc: Document) -> float:
    """
    Score a document based on query term overlap with page_content.

    - Token overlap gives baseline score
    - Exact phrase match in content gives bonus
    - Match in the first 500 characters (likely most relevant part) gives extra bonus
    """
    query_terms = extract_query_terms(query)
    query_tokens = query_terms["token_set"]
    query_phrases = query_terms["phrases"]

    content = doc.page_content
    content_lower = content.lower()

    # Check first 500 chars separately (lead bias)
    lead_content = content_lower[:500]
    # Rest
    rest_content = content_lower[500:]

    score = 0.0

    # --- Token overlap ---
    content_tokens = set(tokenize_text(content_lower))
    lead_tokens = set(tokenize_text(lead_content))

    # Tokens appearing in the lead get 2x weight
    lead_overlap = query_tokens & lead_tokens
    if lead_overlap:
        score += 2.0 * min(len(lead_overlap), 5) / 5.0

    rest_overlap = query_tokens & content_tokens - lead_tokens
    if rest_overlap:
        score += 1.0 * min(len(rest_overlap), 10) / 10.0

    # --- Exact phrase match ---
    phrase_matches = 0
    for phrase in query_phrases:
        if phrase in content_lower:
            phrase_matches += 1
            if phrase in lead_content:
                phrase_matches += 1  # double bonus for lead match

    score += min(phrase_matches, 5) * 0.5

    return min(score, 5.0)  # cap to prevent content from dominating


# ── 4. Section Title Score ────────────────────────────────────────


def compute_section_title_score(query: str, doc: Document) -> float:
    """
    Bonus for query matching the section_title in chunk metadata.

    Section titles from markdown-aware chunking are highly specific
    relevance signals.
    """
    section_title = doc.metadata.get("section_title", "")
    if not section_title or section_title == "Document":
        return 0.0

    query_terms = extract_query_terms(query)
    query_tokens = query_terms["token_set"]
    query_phrases = query_terms["phrases"]

    st_lower = section_title.lower()
    st_tokens = set(tokenize_text(st_lower))

    score = 0.0

    # Phrase match in section title
    for phrase in query_phrases:
        if phrase in st_lower:
            score += 3.0

    # Token overlap
    overlap = query_tokens & st_tokens
    if overlap:
        score += 2.0 * min(len(overlap), 3) / 3.0

    return min(score, 5.0)


# ── 5. Combined Rerank Score ──────────────────────────────────────


def compute_rerank_score(
    query: str,
    doc: Document,
    base_score: float = 0.0,
) -> float:
    """
    Compute a final relevance score for a document.

    Combines:
    - base_score (from upstream fusion): 60% weight
    - metadata_match_score: 25% weight
    - content_match_score: 10% weight
    - section_title_score: 5% weight

    The heuristic signals refine the fusion score rather than replacing it,
    preserving the vector+keyword ranking while boosting documents where
    query terms appear in metadata and section titles.
    """
    metadata_score = compute_metadata_match_score(query, doc)
    content_score = compute_content_match_score(query, doc)
    section_score = compute_section_title_score(query, doc)

    # Normalize heuristic scores to [0, 1] range (rough ceiling)
    meta_norm = min(metadata_score / 15.0, 1.0)
    content_norm = min(content_score / 6.0, 1.0)
    section_norm = min(section_score / 5.0, 1.0)

    final = (
        0.60 * base_score
        + 0.25 * meta_norm
        + 0.10 * content_norm
        + 0.05 * section_norm
    )

    return final


# ── 6. Batch Rerank ───────────────────────────────────────────────


def rerank_candidates(
    query: str,
    candidates: List[Tuple[Document, float]],
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Re-rank a list of (Document, score) candidates.

    Args:
        query: The original user query.
        candidates: List of (Document, base_score) from fusion.
        top_k: Number of results to return.

    Returns:
        Re-ranked List[Tuple[Document, float]] sorted by new score descending.
    """
    if not candidates:
        return []

    reranked: List[Tuple[Document, float]] = []
    for doc, base_score in candidates:
        new_score = compute_rerank_score(query, doc, base_score)
        reranked.append((doc, new_score))

    # Sort descending by new score
    reranked.sort(key=lambda x: x[1], reverse=True)

    return reranked[:top_k]
