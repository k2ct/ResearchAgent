"""
Paper Reading Pipeline v1.

Reads a paper document (markdown, PDF-parsed markdown, or ingested markdown)
and produces a structured reading note with:

1. Title / Basic Information
2. Research Background
3. Research Problem
4. Method Overview
5. Experimental Setup
6. Key Findings
7. Contributions
8. Limitations
9. Relevance to My Research
10. PPT Outline
11. Suggested Follow-up Questions

v1 scope:
- Rule-based section detection (Chinese + English headings)
- Heuristic extraction when sections are not explicitly marked
- No LLM calls (use_llm flag reserved for future upgrade)
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# 1. Load paper markdown
# ---------------------------------------------------------------------------

def _parse_front_matter(text: str) -> Dict[str, Any]:
    """
    Parse YAML front matter from markdown text.

    Lightweight local implementation — does not depend on loaders.py.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    end_idx = None
    for i in range(1, min(len(lines), 30)):  # front matter is always near the top
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return {}

    yaml_text = "\n".join(lines[1:end_idx])
    try:
        raw = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        return {}

    if not isinstance(raw, dict):
        return {}

    return {str(k): v for k, v in raw.items() if v is not None}


def _parse_legacy_basic_info(text: str) -> Dict[str, Any]:
    """
    Parse the legacy ``## 基本信息`` section for metadata.

    Format::

        ## 基本信息
        - key: value
        - key: value
    """
    metadata: Dict[str, Any] = {}
    lines = text.splitlines()
    in_basic_info = False

    for line in lines:
        stripped = line.strip()

        if stripped == "## 基本信息":
            in_basic_info = True
            continue

        if in_basic_info and stripped.startswith("## ") and stripped != "## 基本信息":
            break

        if not in_basic_info:
            continue

        if stripped.startswith("- ") and ":" in stripped:
            item = stripped[2:]
            key, value = item.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                if value.isdigit():
                    metadata[key] = int(value)
                else:
                    metadata[key] = value

    return metadata


def load_paper_markdown(path: Path) -> Dict[str, Any]:
    """
    Load a paper markdown file.

    Supports:
    - YAML front matter (ingested docs)
    - Legacy ``## 基本信息`` section (data/papers/)

    Returns::

        {
            "path": str,
            "metadata": {...},
            "content": str (full markdown body)
        }
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="gbk", errors="replace")

    # Try YAML front matter first, then legacy basic info
    metadata = _parse_front_matter(text)
    if not metadata:
        metadata = _parse_legacy_basic_info(text)

    # Ensure path is recorded
    if "path" not in metadata:
        metadata["path"] = str(path).replace("\\", "/")

    return {
        "path": str(path).replace("\\", "/"),
        "metadata": metadata,
        "content": text,
    }


# ---------------------------------------------------------------------------
# 2. Section detection
# ---------------------------------------------------------------------------

# Mapping: canonical section key -> list of heading patterns (case-insensitive)
SECTION_PATTERNS: Dict[str, List[str]] = {
    "abstract": [
        "abstract", "摘要", "abstract / 摘要",
    ],
    "introduction": [
        "introduction", "引言", "研究背景", "background",
        "introduction / 引言",
        r"^\d+\s*[\.\s]*introduction",
        r"^\d+\s*[\.\s]*简介",
    ],
    "related_work": [
        "related work", "相关工作", "related work / 相关工作",
        "literature review", "文献综述",
    ],
    "method": [
        "method", "methodology", "方法", "方法概括", "method / 方法",
        "approach", "方法学", "proposed method", "our method",
        r"^\d+\s*[\.\s]*method",
        r"^\d+\s*[\.\s]*方法",
    ],
    "experiments": [
        "experiment", "experiments", "实验",
        "experimental setup", "实验设置", "实验设计",
        "使用数据集", "评估任务", "数据集 / 评估任务",
        "evaluation", "evaluation setup",
        r"^\d+\s*[\.\s]*experiment",
        r"^\d+\s*[\.\s]*实验",
    ],
    "results": [
        "result", "results", "结果", "findings", "主要发现",
        "核心发现", "analysis", "结果分析",
        r"^\d+\s*[\.\s]*result",
        r"^\d+\s*[\.\s]*结果",
    ],
    "discussion": [
        "discussion", "讨论", "discussion / 讨论",
    ],
    "limitations": [
        "limitation", "limitations", "局限", "局限性",
        "limitation / 局限", "future work", "后续工作",
        "不足", "限制",
    ],
    "conclusion": [
        "conclusion", "结论", "conclusion / 结论",
        "summary", "总结",
        r"^\d+\s*[\.\s]*conclusion",
        r"^\d+\s*[\.\s]*结论",
    ],
}


def _normalize_heading(heading: str) -> str:
    """Remove leading markers (##, numbers, etc.) and whitespace for matching."""
    # Strip markdown heading markers
    cleaned = re.sub(r'^#+\s*', '', heading)
    # Strip numbered list markers like "1.", "1 ", "1 Introduction"
    cleaned = re.sub(r'^\d+[\.\s)\-]*\s*', '', cleaned)
    return cleaned.strip().lower()


def _split_by_headings(content: str) -> List[Tuple[str, str, str]]:
    """
    Split markdown content into (heading, heading_clean, body) segments.

    Returns a list of tuples where heading is the raw heading line text,
    heading_clean is the normalized version for matching, and body is
    the text under that heading.
    """
    lines = content.splitlines()
    segments: List[Tuple[str, str, str]] = []
    current_heading = ""
    current_body: List[str] = []

    for line in lines:
        # Match markdown headings (#, ##, ###, etc.)
        heading_match = re.match(r'^(#{1,4})\s+(.+)$', line)
        if heading_match:
            # Save previous segment
            if current_body:
                segments.append((
                    current_heading,
                    _normalize_heading(current_heading),
                    "\n".join(current_body).strip(),
                ))

            current_heading = heading_match.group(2).strip()
            current_body = []
        else:
            current_body.append(line)

    # Don't forget the last segment
    if current_body:
        segments.append((
            current_heading,
            _normalize_heading(current_heading),
            "\n".join(current_body).strip(),
        ))

    return segments


def _match_section(heading_clean: str) -> Optional[str]:
    """Try to match a heading against known section patterns."""
    for section_key, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in heading_clean:
                return section_key
            # Try regex patterns
            try:
                if re.match(pattern, heading_clean):
                    return section_key
            except re.error:
                pass
    return None


def detect_paper_sections(content: str) -> Dict[str, str]:
    """
    Detect paper sections from markdown headings.

    Returns a dict mapping canonical section names to their full text::

        {
            "abstract": "...",
            "introduction": "...",
            "related_work": "...",
            "method": "...",
            "experiments": "...",
            "results": "...",
            "discussion": "...",
            "limitations": "...",
            "conclusion": "...",
            "other": "..."
        }
    """
    result: Dict[str, List[str]] = {}
    other_parts: List[str] = []

    segments = _split_by_headings(content)

    for heading_raw, heading_clean, body in segments:
        matched = _match_section(heading_clean)
        if matched:
            if matched not in result:
                result[matched] = []
            result[matched].append(body)
        else:
            # If the heading didn't match, but it has content, add to other
            if body.strip():
                other_parts.append(f"## {heading_raw}\n\n{body}")

    # Merge sections that might have multiple matches
    sections: Dict[str, str] = {}
    for key in SECTION_PATTERNS:
        parts = result.get(key, [])
        sections[key] = "\n\n".join(parts) if parts else ""

    sections["other"] = "\n\n".join(other_parts)

    # Heuristic fallback: if sections are mostly empty, use text patterns
    filled_count = sum(1 for v in sections.values() if v.strip())
    if filled_count <= 1:
        sections = _heuristic_section_detection(content)

    return sections


def _heuristic_section_detection(content: str) -> Dict[str, str]:
    """
    Heuristic section detection when markdown headings are insufficient.

    Uses content-based pattern matching to guess section boundaries.
    """
    result: Dict[str, str] = {}

    # Abstract / Introduction: first 2000 chars of actual content
    # (skip YAML front matter)
    body_start = 0
    lines = content.splitlines()
    if lines and lines[0].strip() == "---":
        # Skip front matter
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                body_start = i + 1
                break

    body = "\n".join(lines[body_start:])
    body_stripped = body.strip()

    first_chunk = body_stripped[:2000] if len(body_stripped) > 2000 else body_stripped
    result["abstract"] = first_chunk

    # Try to identify method/experiment/result by keywords
    lower_all = body_stripped.lower()

    # Method: look for method-related keywords
    method_keywords = [
        "method", "approach", "framework", "pipeline", "architecture",
        "方法", "框架", "流程", "架构", "模型结构",
    ]
    method_text = _extract_paragraphs_around_keywords(body_stripped, method_keywords, window=2)
    result["method"] = method_text if method_text else ""

    # Experiments: look for experiment/data/evaluation keywords
    exp_keywords = [
        "experiment", "dataset", "evaluation", "benchmark", "implementation",
        "实验", "数据集", "评估", "测试", "实现细节",
        "training", "训练",
    ]
    exp_text = _extract_paragraphs_around_keywords(body_stripped, exp_keywords, window=2)
    result["experiments"] = exp_text if exp_text else ""

    # Results: look for result/performance keywords
    result_keywords = [
        "result", "performance", "accuracy", "finding", "shows that",
        "achieve", "outperform",
        "结果", "表现", "准确率", "发现", "表明",
    ]
    res_text = _extract_paragraphs_around_keywords(body_stripped, result_keywords, window=2)
    result["results"] = res_text if res_text else ""

    # Limitations
    limit_keywords = [
        "limitation", "limitations", "future work", "future direction",
        "局限", "不足", "限制", "未来工作", "后续",
    ]
    limit_text = _extract_paragraphs_around_keywords(body_stripped, limit_keywords, window=1)
    result["limitations"] = limit_text if limit_text else ""

    # Conclusion: last 1000 chars
    tail = body_stripped[-1500:] if len(body_stripped) > 1500 else body_stripped
    result["conclusion"] = tail

    # Fill empty keys
    result.setdefault("introduction", "")
    result.setdefault("related_work", "")
    result.setdefault("discussion", "")
    result.setdefault("other", "")

    return result


def _extract_paragraphs_around_keywords(
    text: str,
    keywords: List[str],
    window: int = 2,
) -> str:
    """
    Extract paragraphs that contain any of the given keywords,
    plus a window of surrounding paragraphs.
    """
    paragraphs = text.split("\n\n")
    matched_indices: set = set()

    for i, para in enumerate(paragraphs):
        para_lower = para.lower()
        if any(kw.lower() in para_lower for kw in keywords):
            for offset in range(-window, window + 1):
                idx = i + offset
                if 0 <= idx < len(paragraphs):
                    matched_indices.add(idx)

    if not matched_indices:
        return ""

    return "\n\n".join(
        paragraphs[i] for i in sorted(matched_indices)
    )


# ---------------------------------------------------------------------------
# 3. Paper metadata extraction
# ---------------------------------------------------------------------------

def extract_paper_metadata(
    path: Path,
    metadata: Dict[str, Any],
    content: str,
) -> Dict[str, Any]:
    """
    Extract structured paper metadata.

    Returns::

        {
            "title": str,
            "year": int | None,
            "venue": str,
            "authors": str,
            "source_path": str,
            "doc_type": str,
            "source_type": str,
        }
    """
    # Title
    title = metadata.get("title", "")
    if not title:
        # Try first # heading
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                title = stripped[2:].strip()
                break
    if not title:
        title = path.stem

    # Year
    year = metadata.get("year")
    if not year:
        # Try regex from filename or content
        year_match = re.search(r'\b(20\d{2})\b', content[:3000])
        if year_match:
            year = int(year_match.group(1))
        else:
            year_match = re.search(r'\b(20\d{2})\b', path.name)
            if year_match:
                year = int(year_match.group(1))

    # Venue
    venue = metadata.get("venue", "")
    if not venue:
        venue = metadata.get("journal", "")
    if not venue:
        venue = metadata.get("conference", "")
    if not venue:
        venue = "unknown"

    # Authors
    authors = metadata.get("authors", "")
    if not authors:
        authors = "unknown"

    # Source
    source_path = str(path).replace("\\", "/")

    # Doc type and source type
    doc_type = metadata.get("doc_type", path.suffix.lstrip("."))
    source_type = metadata.get("source_type", "paper_note")

    return {
        "title": title,
        "year": year,
        "venue": venue,
        "authors": authors,
        "source_path": source_path,
        "doc_type": doc_type,
        "source_type": source_type,
    }


# ---------------------------------------------------------------------------
# 4. Build structured paper note (rule-based)
# ---------------------------------------------------------------------------

def build_structured_paper_note(
    paper_data: Dict[str, Any],
    use_llm: bool = False,
) -> str:
    """
    Generate a structured paper reading note in markdown.

    v1: Uses rule-based summarisation — extracts and reformats section content.
    When ``use_llm=True`` (future), delegates to an LLM for more nuanced notes.

    Returns a multi-section markdown string.
    """
    sections = paper_data.get("sections", {})
    meta = paper_data.get("metadata", {})

    title = meta.get("title", "Unknown Paper")
    year = meta.get("year", "?")
    venue = meta.get("venue", "unknown")
    source_path = paper_data.get("path", "")

    # Helper: get section text or placeholder
    def _s(key: str) -> str:
        text = sections.get(key, "")
        if text.strip():
            # Truncate very long sections for readability
            if len(text) > 3000:
                return text[:3000] + "\n\n...(truncated)"
            return text
        return "_当前文档未明确提供。_"

    def _s_short(key: str, max_chars: int = 1500) -> str:
        text = sections.get(key, "")
        if text.strip():
            if len(text) > max_chars:
                return text[:max_chars] + "\n\n...(truncated)"
            return text
        return "_当前文档未明确提供。_"

    note = f"""# Paper Reading Note

## 1. Basic Information

- **Title:** {title}
- **Year:** {year}
- **Venue:** {venue}
- **Source:** {source_path}
- **Doc Type:** {meta.get('doc_type', 'unknown')}

---

## 2. Research Background

{_s_short("introduction")}

{_s_short("related_work")}

---

## 3. Research Problem

{_s_short("abstract")}

---

## 4. Method Overview

{_s("method")}

---

## 5. Experimental Setup

{_s("experiments")}

---

## 6. Key Findings

{_s_short("results", 2000)}

---

## 7. Contributions

{_extract_contributions(sections, meta)}

---

## 8. Limitations

{_s_short("limitations")}

---

## 9. Relevance to My Research

{_infer_relevance(sections, meta)}

---

## 10. PPT Outline

{_generate_ppt_outline(meta, sections)}

---

## 11. Suggested Follow-up Questions

{_generate_followup_questions(meta, sections)}

---

*Generated by ResearchAgent Paper Reading Pipeline v1 (rule-based).*
*Sources: {source_path}*
"""

    return note


def _extract_contributions(sections: Dict[str, str], meta: Dict[str, Any]) -> str:
    """Extract or infer contributions from paper sections."""
    # Check discussion/conclusion for contribution-like language
    for key in ["conclusion", "discussion", "abstract"]:
        text = sections.get(key, "")
        if text:
            # Look for contribution-related sentences
            contrib_lines = []
            for line in text.splitlines():
                line_lower = line.lower()
                if any(kw in line_lower for kw in [
                    "contribution", "contribute", "propose", "introduce",
                    "novel", "new", "first", "state-of-the-art",
                    "贡献", "提出", "首次", "创新", "改进",
                    "outperform", "achieve", "advance",
                ]):
                    contrib_lines.append(line.strip())
            if contrib_lines:
                return "\n".join(contrib_lines[:15])

    return "_当前文档未明确列出 contributions。_"


def _infer_relevance(sections: Dict[str, str], meta: Dict[str, Any]) -> str:
    """Infer relevance to the user's research (multi-modal bias/hallucination)."""
    # Check for explicit relationship section (legacy papers)
    for key in ["other"]:
        text = sections.get(key, "")
        if "ResearchAgent" in text or "research agent" in text.lower():
            # Extract the relevant paragraph
            for para in text.split("\n\n"):
                if "ResearchAgent" in para or "research agent" in para.lower():
                    return para.strip()[:1000]

    # Heuristic relevance based on topic/title
    title = str(meta.get("title", ""))
    topic = str(meta.get("topic", ""))

    combined = (title + " " + topic).lower()

    relevance_parts = []
    if any(kw in combined for kw in ["bias", "偏见"]):
        relevance_parts.append(
            "- This paper relates to **multi-modal bias evaluation**, "
            "which is a core research direction of this project. "
            "It may provide methods, metrics, or datasets relevant to "
            "building a stereotype library or bias auditing pipeline."
        )
    if any(kw in combined for kw in ["hallucination", "幻觉"]):
        relevance_parts.append(
            "- This paper relates to **hallucination detection/evaluation**, "
            "which connects to the hallucination screening pipeline "
            "(coco_val_n300_g1, hrs_v1)."
        )
    if any(kw in combined for kw in ["guardrail", "护栏", "safety"]):
        relevance_parts.append(
            "- This paper discusses **guardrail/safety mechanisms** in VLMs, "
            "which may inform how we design bias evaluation that is robust "
            "to model refusal behaviours."
        )
    if any(kw in combined for kw in ["stereotype", "刻板", "social"]):
        relevance_parts.append(
            "- This paper addresses **social stereotypes**, directly relevant "
            "to building a stereotype library and studying co-occurrence "
            "biases in multi-modal models."
        )

    if relevance_parts:
        return "\n\n".join(relevance_parts)

    return (
        "- This paper may provide general background or methodology "
        "relevant to multi-modal model evaluation. "
        "Further review is needed to determine specific connections "
        "to the project's bias/hallucination research directions."
    )


def _generate_ppt_outline(meta: Dict[str, Any], sections: Dict[str, str]) -> str:
    """Generate a PPT slide outline from paper sections."""
    title = meta.get("title", "Paper")
    venue = meta.get("venue", "")
    year = meta.get("year", "")

    slides = [
        f"**Slide 1 — Title Page**",
        f"  - {title}",
        f"  - {venue} ({year})" if venue or year else "",
        "",
        f"**Slide 2 — Research Background**",
        f"  - Key context and motivation",
        f"  - Gap in existing work",
        "",
        f"**Slide 3 — Research Problem**",
        f"  - Core research question",
        f"  - Why it matters",
        "",
        f"**Slide 4 — Method**",
        f"  - High-level approach (1-3 bullet points)",
        f"  - Key innovation",
        "",
        f"**Slide 5 — Experimental Setup**",
        f"  - Datasets used",
        f"  - Baselines / comparisons",
        "",
        f"**Slide 6 — Key Results**",
        f"  - Main findings (2-4 bullet points)",
        f"  - Most important numbers/trends",
        "",
        f"**Slide 7 — Contributions & Limitations**",
        f"  - Contributions",
        f"  - Known limitations",
        "",
        f"**Slide 8 — Relevance & Discussion**",
        f"  - How this relates to our project",
        f"  - Open questions / discussion points",
    ]

    if sections.get("abstract"):
        # Add a note about what we know
        abstract_len = len(sections["abstract"])
        slides.append("")
        slides.append(f"_Based on {abstract_len}-char abstract in source document._")

    return "\n".join(slides)


def _generate_followup_questions(meta: Dict[str, Any], sections: Dict[str, str]) -> str:
    """Generate suggested follow-up questions based on the paper."""
    questions = []

    title = str(meta.get("title", ""))
    topic = str(meta.get("topic", "")).lower()

    questions.append(f"1. What is the core technical contribution of this paper?")
    questions.append(f"2. What datasets and metrics does this paper use for evaluation?")

    if "bias" in topic or "bias" in title.lower() or "偏见" in title:
        questions.append(
            f"3. How does this paper's bias evaluation methodology compare "
            f"to other approaches in the field?"
        )

    if "hallucination" in topic or "hallucination" in title.lower():
        questions.append(
            f"3. What hallucination metrics does this paper propose or use?"
        )

    questions.append(
        f"4. What are the main limitations acknowledged by the authors?"
    )
    questions.append(
        f"5. Can the method/dataset from this paper be applied to our "
        f"COCO hallucination screening pipeline or stereotype library?"
    )
    questions.append(
        f"6. What follow-up experiments would strengthen the findings?"
    )

    return "\n".join(questions)


# ---------------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------------

def read_paper(path: Path, use_llm: bool = False, save_memory: bool = False) -> Dict[str, Any]:
    """
    Read a paper and generate a structured reading note.

    Args:
        path: Path to the paper markdown file.
        use_llm: Reserved for future LLM-assisted reading notes.
        save_memory: If True, write the result to the Memory Store via
            memory.adapters.save_paper_reading_result().

    Returns::

        {
            "paper_path": str,
            "metadata": {...},
            "sections": {...},
            "reading_note": str,
            "sources": [...],
            "status": "success" | "error",
            "error": str,
            "used_llm": bool,
            "llm_error": str,
            "memory_saved": bool,
            "memory_result": dict | None,
        }
    """
    result = {
        "paper_path": str(path),
        "metadata": {},
        "sections": {},
        "reading_note": "",
        "sources": [],
        "status": "error",
        "error": "",
        "used_llm": False,
        "llm_error": "",
    }

    try:
        # Step 1: Load
        paper_data = load_paper_markdown(path)
        content = paper_data["content"]
        raw_metadata = paper_data["metadata"]

        # Step 2: Detect sections
        sections = detect_paper_sections(content)

        # Step 3: Extract structured metadata
        meta = extract_paper_metadata(path, raw_metadata, content)

        # Step 4: Build rule-based reading note
        reading_note = build_structured_paper_note(
            {
                "path": paper_data["path"],
                "metadata": meta,
                "sections": sections,
                "content": content,
            },
            use_llm=False,  # always use rule-based first
        )

        # Step 5: LLM enhancement (if requested and available)
        if use_llm:
            try:
                from research_agent.llm.enhancers import enhance_paper_reading_note
                enhanced = enhance_paper_reading_note(meta, sections, reading_note)
                if enhanced["used_llm"]:
                    reading_note = enhanced["text"]
                    result["used_llm"] = True
                if enhanced.get("error"):
                    result["llm_error"] = enhanced["error"]
            except Exception as e:
                result["llm_error"] = f"{type(e).__name__}: {e}"

        result["metadata"] = meta
        result["sections"] = {
            k: (v[:200] + "...") if len(v) > 200 else v
            for k, v in sections.items() if v
        }
        result["reading_note"] = reading_note
        result["sources"] = [str(path).replace("\\", "/")]
        result["status"] = "success"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    # Step 6: Optional memory write-back
    if save_memory and result["status"] == "success":
        try:
            from research_agent.memory.adapters import save_paper_reading_result
            memory_result = save_paper_reading_result(result, auto_write=True)
        except Exception as e:
            memory_result = {"ok": False, "error": str(e)}

        result["memory_saved"] = bool(memory_result.get("ok"))
        result["memory_result"] = memory_result
    else:
        result["memory_saved"] = False
        result["memory_result"] = None

    return result


# ---------------------------------------------------------------------------
# 6. Batch processing
# ---------------------------------------------------------------------------

def batch_read_papers(
    input_dir: Path,
    output_dir: Path,
    use_llm: bool = False,
) -> List[Dict[str, Any]]:
    """
    Batch-read papers from a directory and save reading notes.

    Args:
        input_dir: Directory containing .md paper files.
        output_dir: Directory to write ``<stem>_paper_note.md`` files.
        use_llm: Reserved for future LLM-assisted mode.

    Returns:
        List of result dicts, one per paper processed.
    """
    if not input_dir.exists():
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for path in sorted(input_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        if path.stat().st_size == 0:
            continue

        result = read_paper(path, use_llm=use_llm)

        if result["status"] == "success":
            output_path = output_dir / f"{path.stem}_paper_note.md"
            output_path.write_text(result["reading_note"], encoding="utf-8")
            result["output_path"] = str(output_path)

        results.append(result)

    return results
