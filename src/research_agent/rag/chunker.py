"""
Markdown-aware / Section-aware Chunking v1

Core strategy:
- Prioritize splitting by Markdown headings (H1–H6)
- Recognize ## Page N markers from PDF extraction
- Recognize ## Slide N markers from PPT extraction
- Merge small neighbouring sections
- Split oversized sections by paragraphs (with overlap)
- Add rich chunk-level metadata (section_title, chunk_index, chunk_strategy, …)

This module is designed to be a drop-in replacement for the simple
``RecursiveCharacterTextSplitter`` call inside ``indexer.py``.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# ## Page 1, ## Page 2, ...
PAGE_HEADING_PATTERN = re.compile(r"^#{1,3}\s+Page\s+(\d+)", re.IGNORECASE | re.MULTILINE)

# ## Slide 1, ## Slide 2, ...
SLIDE_HEADING_PATTERN = re.compile(r"^#{1,3}\s+Slide\s+(\d+)", re.IGNORECASE | re.MULTILINE)

# Split paragraphs on blank lines
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")

# Sentence boundaries (Chinese + English)
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?])\s*")

# Sentinel value: "use profile / env config"
_AUTO = object()

# Hardcoded ultimate fallback defaults
_DEFAULT_MAX_CHARS = 1200
_DEFAULT_MIN_CHARS = 250
_DEFAULT_OVERLAP_CHARS = 150


# ---------------------------------------------------------------------------
# 0. Configuration helpers
# ---------------------------------------------------------------------------

def get_chunking_config() -> dict:
    """
    Read chunking parameters from environment variables.

    Checks (in order):
    1. ``RAG_CHUNK_MAX_CHARS``  (default 1200)
    2. ``RAG_CHUNK_MIN_CHARS``  (default 250)
    3. ``RAG_CHUNK_OVERLAP_CHARS`` (default 150)

    Returns a dict with keys *max_chars*, *min_chars*, *overlap_chars*.
    """
    def _int_env(key: str, fallback: int) -> int:
        val = os.getenv(key, "").strip()
        if val.isdigit():
            return int(val)
        return fallback

    return {
        "max_chars": _int_env("RAG_CHUNK_MAX_CHARS", _DEFAULT_MAX_CHARS),
        "min_chars": _int_env("RAG_CHUNK_MIN_CHARS", _DEFAULT_MIN_CHARS),
        "overlap_chars": _int_env("RAG_CHUNK_OVERLAP_CHARS", _DEFAULT_OVERLAP_CHARS),
    }


def get_doc_type_chunking_profile(metadata: dict) -> dict:
    """
    Return recommended chunking parameters for a document based on its
    ``source_type`` / ``doc_type`` metadata.

    Profiles
    --------
    - **slide_doc / pptx**: tighter chunks, zero overlap, preserve slide
      boundaries.
    - **paper_note / pdf**: larger chunks, higher overlap for academic prose.
    - **experiment_doc**: moderate chunks, moderate overlap.
    - **note_doc / misc_doc / default**: standard settings.

    Returns a dict that may contain *max_chars*, *min_chars*,
    *overlap_chars*, and *preserve_slide*.  Missing keys mean "use the
    global default".
    """
    source_type = metadata.get("source_type", "")
    doc_type = metadata.get("doc_type", "")

    # --- slide / pptx ---
    if source_type == "slide_doc" or doc_type == "pptx":
        return {
            "max_chars": 1000,
            "min_chars": 100,
            "overlap_chars": 0,
            "preserve_slide": True,
        }

    # --- paper / PDF ---
    if source_type == "paper_note" or doc_type == "pdf":
        return {
            "max_chars": 1500,
            "min_chars": 300,
            "overlap_chars": 180,
        }

    # --- experiment ---
    if source_type == "experiment_doc":
        return {
            "max_chars": 1000,
            "min_chars": 200,
            "overlap_chars": 120,
        }

    # --- note / misc / default ---
    if source_type in ("note_doc", "misc_doc"):
        return {
            "max_chars": 1200,
            "min_chars": 250,
            "overlap_chars": 150,
        }

    # dataset_doc and anything else: return empty → fall back to env / global
    return {}


def _resolve_chunking_params(
    metadata: dict,
    max_chars: Optional[int],
    min_chars: Optional[int],
    overlap_chars: Optional[int],
) -> Tuple[int, int, int]:
    """
    Resolve the final (max_chars, min_chars, overlap_chars) for a document.

    Resolution order (first wins):
    1. Explicitly passed value (not None)
    2. Per-doc-type profile via :func:`get_doc_type_chunking_profile`
    3. Environment config via :func:`get_chunking_config`
    4. Hardcoded defaults (1200 / 250 / 150)
    """
    profile = get_doc_type_chunking_profile(metadata)
    env = get_chunking_config()

    def _pick(key: str, explicit: Optional[int]) -> int:
        if explicit is not None:
            return explicit
        if key in profile:
            return profile[key]
        return env.get(key, _DEFAULT_MAX_CHARS)

    return (
        _pick("max_chars", max_chars),
        _pick("min_chars", min_chars),
        _pick("overlap_chars", overlap_chars),
    )


# ---------------------------------------------------------------------------
# 1. YAML front matter stripping
# ---------------------------------------------------------------------------

def strip_yaml_front_matter(text: str) -> str:
    """
    Remove YAML front matter from the beginning of a markdown string.

    Front matter is delimited by ``---`` on the first line and a matching
    ``---`` on a later line.  If no front matter is found the text is
    returned unchanged.

    This avoids duplicate metadata leaking into chunk page_content.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text

    # Find the closing ---
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            # Rejoin everything after the closing delimiter
            remaining = lines[i + 1:]
            return "\n".join(remaining)

    # No closing delimiter found — treat as plain text
    return text


# ---------------------------------------------------------------------------
# 2. Markdown section detection
# ---------------------------------------------------------------------------

def _build_section_path(heading_stack: List[Tuple[int, str]]) -> str:
    """Join heading stack into a path like ``Title > Section > Subsection``."""
    return " > ".join(title for _, title in heading_stack)


def detect_markdown_sections(text: str) -> List[dict]:
    """
    Scan *text* for Markdown headings and return a list of section dicts.

    Each dict contains:

    - **title**      – the heading text of this section
    - **level**      – heading level (1-6)
    - **start**      – character offset where the section begins
    - **end**        – character offset where the section ends (exclusive)
    - **text**       – the full section body (including the heading line)
    - **section_path** – hierarchical path, e.g. ``"Title > Section"``
    - **is_page**    – True if the heading matches ``## Page N``
    - **is_slide**   – True if the heading matches ``## Slide N``

    If the document has **no** heading at all, the entire text is returned
    as a single section with ``title="Document"`` and ``level=0``.
    """
    # Strip YAML front matter first so it doesn't become a spurious preamble
    clean_text = strip_yaml_front_matter(text)

    # Find all heading matches
    matches = list(HEADING_PATTERN.finditer(clean_text))

    if not matches:
        # No headings — entire document is one section
        return [{
            "title": "Document",
            "level": 0,
            "start": 0,
            "end": len(clean_text),
            "text": clean_text,
            "section_path": "Document",
            "is_page": False,
            "is_slide": False,
        }]

    sections: List[dict] = []

    # Preamble: text before the first heading
    first_match = matches[0]
    if first_match.start() > 0:
        preamble_text = clean_text[:first_match.start()].strip()
        if preamble_text:
            sections.append({
                "title": "Preamble",
                "level": 0,
                "start": 0,
                "end": first_match.start(),
                "text": preamble_text,
                "section_path": "Preamble",
                "is_page": False,
                "is_slide": False,
            })

    # Track heading hierarchy for section_path
    # stack entries: (level, title)
    heading_stack: List[Tuple[int, str]] = []

    for idx, match in enumerate(matches):
        hashes = match.group(1)
        title = match.group(2).strip()
        level = len(hashes)
        section_start = match.start()

        # Determine section end (start of next heading, or end of text)
        if idx + 1 < len(matches):
            section_end = matches[idx + 1].start()
        else:
            section_end = len(clean_text)

        section_text = clean_text[section_start:section_end]

        # Update heading stack for section_path
        # Pop headings that are >= current level (same or deeper)
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        section_path = _build_section_path(heading_stack)

        # Detect Page / Slide markers in the heading text
        is_page = bool(PAGE_HEADING_PATTERN.search(match.group(0)))
        is_slide = bool(SLIDE_HEADING_PATTERN.search(match.group(0)))

        sections.append({
            "title": title,
            "level": level,
            "start": section_start,
            "end": section_end,
            "text": section_text,
            "section_path": section_path,
            "is_page": is_page,
            "is_slide": is_slide,
        })

    return sections


# ---------------------------------------------------------------------------
# 3. Paragraph-based splitting for oversized sections
# ---------------------------------------------------------------------------

def _split_single_paragraph(text: str, max_chars: int) -> List[str]:
    """
    Split a single paragraph that is too long into smaller pieces,
    trying to break at sentence boundaries.
    """
    if len(text) <= max_chars:
        return [text]

    # Try sentence boundaries first
    sentences = SENTENCE_BOUNDARY.split(text)
    # Filter out empty strings
    sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) <= max_chars:
            current += sent
        else:
            if current:
                chunks.append(current)
            # If a single sentence is still too long, hard-split it
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i:i + max_chars])
                current = ""
            else:
                current = sent

    if current:
        chunks.append(current)

    return chunks or [text]


def split_long_text_by_paragraphs(
    text: str,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[str]:
    """
    Split *text* into chunks at paragraph boundaries (blank lines).

    If a paragraph is still longer than *max_chars*, further split it at
    sentence boundaries.  If a sentence is STILL too long, fall back to a
    hard character split.

    *overlap_chars* characters of context are carried over between adjacent
    chunks to reduce information loss at boundaries.
    """
    # Normalise whitespace while preserving paragraph breaks
    paragraphs = PARAGRAPH_SPLIT_PATTERN.split(text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return [text] if text.strip() else []

    chunks: List[str] = []
    current_buf: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # If the paragraph alone exceeds max_chars, flush current buffer
        # first, then split this paragraph further
        if para_len > max_chars:
            # Flush current buffer
            if current_buf:
                chunks.append("\n\n".join(current_buf))
                current_buf = []
                current_len = 0

            # Split this long paragraph
            sub_chunks = _split_single_paragraph(para, max_chars)
            for i, sc in enumerate(sub_chunks):
                if i > 0 and overlap_chars > 0:
                    # Prepend overlap from the previous sub-chunk
                    prev = sub_chunks[i - 1]
                    overlap_text = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
                    sc = overlap_text + "\n\n" + sc
                chunks.append(sc)
            continue

        # Can we add this paragraph to the current buffer?
        separator_len = 2 if current_buf else 0  # "\n\n"
        if current_len + separator_len + para_len <= max_chars:
            current_buf.append(para)
            current_len += separator_len + para_len
        else:
            # Buffer is full — flush it
            if current_buf:
                chunks.append("\n\n".join(current_buf))
            current_buf = [para]
            current_len = para_len

    # Flush remaining buffer
    if current_buf:
        chunks.append("\n\n".join(current_buf))

    # Add overlap between consecutive chunks
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]
            overlap_text = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            overlapped.append(overlap_text + "\n\n" + curr)
        return overlapped

    return chunks


# ---------------------------------------------------------------------------
# 4. Merge small sections
# ---------------------------------------------------------------------------

def _should_avoid_merge(doc_metadata: dict, sections: List[dict], idx: int) -> bool:
    """
    Decide whether merging section *idx* with its neighbour should be avoided.

    For slide_doc / pptx documents we avoid merging across slide boundaries
    because each slide is a semantically independent unit.
    """
    source_type = doc_metadata.get("source_type", "")
    doc_type = doc_metadata.get("doc_type", "")

    if source_type == "slide_doc" or doc_type == "pptx":
        return True

    section = sections[idx]
    if section.get("is_slide"):
        return True

    return False


def merge_small_sections(
    sections: List[dict],
    min_chars: int = 300,
    max_chars: int = 1200,
    doc_metadata: Optional[dict] = None,
) -> List[dict]:
    """
    Merge adjacent sections whose text is shorter than *min_chars*.

    Merging continues as long as the combined length stays within *max_chars*.
    For slide/PPT documents, cross-slide merging is avoided.

    Parameters
    ----------
    sections : list[dict]
        Output of :func:`detect_markdown_sections`.
    min_chars : int
        Sections shorter than this are candidates for merging.
    max_chars : int
        Merged section should not exceed this limit (soft).
    doc_metadata : dict or None
        Original document metadata; used to decide merge policy.

    Returns
    -------
    list[dict]
        Sections after merging.  The dict shape is the same as the input
        but *title* may be updated to ``"A + B"`` to reflect a merge.
    """
    if not sections:
        return sections

    doc_meta = doc_metadata or {}
    merged: List[dict] = []
    i = 0

    while i < len(sections):
        current = dict(sections[i])  # shallow copy

        if len(current["text"]) >= min_chars:
            merged.append(current)
            i += 1
            continue

        # Current section is short — try to merge forward
        # Check if merging should be avoided
        if _should_avoid_merge(doc_meta, sections, i):
            merged.append(current)
            i += 1
            continue

        combined_text = current["text"]
        combined_end = current["end"]
        combined_titles = [current["title"]]
        j = i + 1

        while j < len(sections):
            if _should_avoid_merge(doc_meta, sections, j):
                break

            next_sec = sections[j]
            candidate_len = len(combined_text) + len(next_sec["text"])

            if candidate_len <= max_chars * 1.2:  # allow 20% slack
                combined_text += "\n\n" + next_sec["text"]
                combined_end = next_sec["end"]
                combined_titles.append(next_sec["title"])
                j += 1
            else:
                break

        if j > i + 1:
            # Successfully merged at least one extra section
            current["text"] = combined_text
            current["end"] = combined_end
            current["title"] = " + ".join(combined_titles)
            current["section_path"] = " + ".join(combined_titles)
            i = j
        else:
            i += 1

        merged.append(current)

    return merged


# ---------------------------------------------------------------------------
# 5. Main chunking function (single document)
# ---------------------------------------------------------------------------

def chunk_document_markdown_aware(
    doc: Document,
    max_chars: Optional[int] = None,
    min_chars: Optional[int] = None,
    overlap_chars: Optional[int] = None,
) -> List[Document]:
    """
    Split a single LangChain ``Document`` into chunks using a
    markdown-aware / section-aware strategy.

    Strategy (in order):

    1. Strip YAML front matter from ``page_content``.
    2. Detect sections via Markdown headings (``#``, ``##``, …).
    3. Recognise ``## Page N`` / ``## Slide N`` markers.
    4. Merge sections shorter than *min_chars*.
    5. Split sections longer than *max_chars* by paragraphs.
    6. For slide/PPT documents, prefer slide-level chunks.

    Every returned chunk carries the original document metadata PLUS:

    - ``chunk_index``
    - ``chunk_count``
    - ``chunk_strategy``
    - ``section_title``
    - ``section_path``
    - ``chunk_chars``

    Parameters
    ----------
    doc : Document
        The LangChain document to chunk.
    max_chars : int or None
        Target maximum characters per chunk.
        ``None`` → auto-detect from doc-type profile or env config.
    min_chars : int or None
        Sections shorter than this may be merged.
        ``None`` → auto-detect from doc-type profile or env config.
    overlap_chars : int or None
        Character overlap between paragraph-split chunks.
        ``None`` → auto-detect from doc-type profile or env config.

    Returns
    -------
    List[Document]
        Chunk documents with enriched metadata.
    """
    original_metadata = deepcopy(doc.metadata)
    source_type = original_metadata.get("source_type", "")
    doc_type = original_metadata.get("doc_type", "")
    text = doc.page_content

    # Resolve parameters: explicit > profile > env > hardcoded default
    _max_chars, _min_chars, _overlap_chars = _resolve_chunking_params(
        original_metadata, max_chars, min_chars, overlap_chars,
    )

    # ------------------------------------------------------------------
    # Step 1: Detect sections
    # ------------------------------------------------------------------
    sections = detect_markdown_sections(text)

    # ------------------------------------------------------------------
    # Step 2: Determine if the doc is short enough to keep as-is
    # ------------------------------------------------------------------
    if len(text) <= _max_chars and len(sections) <= 1:
        # Short document — single chunk
        chunk_meta = dict(original_metadata)
        chunk_meta.update({
            "chunk_index": 0,
            "chunk_count": 1,
            "chunk_strategy": "full_document",
            "section_title": sections[0]["title"] if sections else "Document",
            "section_path": sections[0]["section_path"] if sections else "Document",
            "chunk_chars": len(text),
        })
        return [Document(page_content=text, metadata=chunk_meta)]

    # ------------------------------------------------------------------
    # Step 3: Merge small sections (unless slide/page boundaries)
    # ------------------------------------------------------------------
    sections = merge_small_sections(sections, min_chars=_min_chars,
                                    max_chars=_max_chars,
                                    doc_metadata=original_metadata)

    # ------------------------------------------------------------------
    # Step 4: Build chunks from sections
    # ------------------------------------------------------------------
    is_slide_doc = (source_type == "slide_doc" or doc_type == "pptx")
    chunks: List[Document] = []

    for sec in sections:
        sec_text = sec["text"]
        sec_len = len(sec_text)

        # Determine chunk strategy
        if sec.get("is_slide"):
            strategy = "slide"
        elif sec.get("is_page"):
            strategy = "page"
        elif sec_len <= _max_chars:
            strategy = "section"
        else:
            strategy = "section"

        # If the section fits within _max_chars, use it as-is
        if sec_len <= _max_chars:
            chunk_meta = dict(original_metadata)
            chunk_meta["section_title"] = sec["title"]
            chunk_meta["section_path"] = sec["section_path"]
            chunk_meta["chunk_strategy"] = strategy
            chunk_meta["chunk_chars"] = sec_len
            chunks.append(Document(page_content=sec_text, metadata=chunk_meta))
        else:
            # Oversized section — split by paragraphs
            sub_texts = split_long_text_by_paragraphs(
                sec_text, max_chars=_max_chars, overlap_chars=_overlap_chars
            )
            for sub_text in sub_texts:
                chunk_meta = dict(original_metadata)
                chunk_meta["section_title"] = sec["title"]
                chunk_meta["section_path"] = sec["section_path"]
                chunk_meta["chunk_strategy"] = "paragraph_split"
                chunk_meta["chunk_chars"] = len(sub_text)
                chunks.append(Document(page_content=sub_text, metadata=chunk_meta))

    # ------------------------------------------------------------------
    # Step 5: Fallback — if chunking produced nothing, return full doc
    # ------------------------------------------------------------------
    if not chunks:
        chunk_meta = dict(original_metadata)
        chunk_meta.update({
            "chunk_index": 0,
            "chunk_count": 1,
            "chunk_strategy": "full_document",
            "section_title": "Document",
            "section_path": "Document",
            "chunk_chars": len(text),
        })
        return [Document(page_content=text, metadata=chunk_meta)]

    # ------------------------------------------------------------------
    # Step 6: Number chunks
    # ------------------------------------------------------------------
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_count"] = total

    return chunks


# ---------------------------------------------------------------------------
# 6. Batch chunking (multiple documents)
# ---------------------------------------------------------------------------

def chunk_documents_markdown_aware(
    docs: List[Document],
    max_chars: Optional[int] = None,
    min_chars: Optional[int] = None,
    overlap_chars: Optional[int] = None,
) -> List[Document]:
    """
    Split a list of LangChain ``Document`` objects using
    :func:`chunk_document_markdown_aware` on each.

    Parameters
    ----------
    docs : List[Document]
        Source documents (from loaders).
    max_chars : int or None
        Target maximum characters per chunk.
        ``None`` → auto-detect per document from its profile / env config.
    min_chars : int or None
        Sections shorter than this may be merged.
        ``None`` → auto-detect per document.
    overlap_chars : int or None
        Character overlap between paragraph-split chunks.
        ``None`` → auto-detect per document.

    Returns
    -------
    List[Document]
        All chunk documents, with enriched metadata.
    """
    all_chunks: List[Document] = []
    for doc in docs:
        chunks = chunk_document_markdown_aware(
            doc,
            max_chars=max_chars,
            min_chars=min_chars,
            overlap_chars=overlap_chars,
        )
        all_chunks.extend(chunks)
    return all_chunks
