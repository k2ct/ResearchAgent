"""
PPT / Research Progress Memory.

Reads slide_doc markdown (from PPTX ingestion or hand-written slides) and
produces a structured *Research Progress Memory* document covering:

1. Presentation metadata
2. Slide-by-slide summary (title + bullets per slide)
3. Inferred research questions
4. Completed work
5. Experiments and results
6. Issues / limitations
7. Next steps
8. Long-term memory records (tagged bullets for future retrieval)

Design: heuristic keyword matching — no LLM dependency.
Heuristics are intentionally conservative; sections with no evidence
are marked *"Current PPT did not explicitly provide this."*
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load slide markdown
# ═══════════════════════════════════════════════════════════════════════════

def load_slide_markdown(path: Path) -> Dict[str, Any]:
    """
    Read a slide_doc markdown file (with optional YAML front matter).

    Returns::

        {
            "path": str,
            "metadata": {...},
            "content": str,
        }
    """
    text = path.read_text(encoding="utf-8")

    metadata: Dict[str, Any] = {}
    content = text

    # Try YAML front matter
    if text.startswith("---"):
        lines = text.splitlines()
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break
        if end_idx is not None:
            try:
                import yaml
                metadata = yaml.safe_load("\n".join(lines[1:end_idx])) or {}
            except Exception:
                pass
            content = "\n".join(lines[end_idx + 1:])

    return {
        "path": str(path),
        "metadata": metadata,
        "content": content,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. Parse slides
# ═══════════════════════════════════════════════════════════════════════════

_SLIDE_HEADING_RE = re.compile(r"^#{1,3}\s+Slide\s+(\d+)", re.IGNORECASE | re.MULTILINE)
_MD_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_BULLET_RE = re.compile(r"^[\-\*\+]\s+(.+)$", re.MULTILINE)


def parse_slides_from_markdown(content: str) -> List[Dict[str, Any]]:
    """
    Extract slides from markdown content.

    Prefers ``## Slide N`` markers (from PPTX ingestion).
    Falls back to Markdown heading sections if no Slide markers found.

    Each slide dict::

        {
            "slide_number": int,
            "title": str,
            "content": str,
            "bullets": [str, ...],
        }
    """
    # Try Slide markers first
    slide_matches = list(_SLIDE_HEADING_RE.finditer(content))
    if slide_matches:
        slides: List[Dict[str, Any]] = []
        for idx, m in enumerate(slide_matches):
            num = int(m.group(1))
            start = m.end()
            end = slide_matches[idx + 1].start() if idx + 1 < len(slide_matches) else len(content)
            slide_text = content[start:end].strip()
            title = _extract_slide_title(slide_text)
            bullets = _extract_bullets(slide_text)
            slides.append({
                "slide_number": num,
                "title": title,
                "content": slide_text,
                "bullets": bullets,
            })
        return slides

    # Fallback: split by markdown headings
    heading_matches = list(_MD_HEADING_RE.finditer(content))
    if not heading_matches:
        # Entire document as one slide
        bullets = _extract_bullets(content)
        return [{"slide_number": 1, "title": _extract_slide_title(content), "content": content, "bullets": bullets}]

    slides = []
    for idx, m in enumerate(heading_matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = heading_matches[idx + 1].start() if idx + 1 < len(heading_matches) else len(content)
        slide_text = content[start:end].strip()
        bullets = _extract_bullets(slide_text)
        slides.append({
            "slide_number": idx + 1,
            "title": title,
            "level": level,
            "content": slide_text,
            "bullets": bullets,
        })
    return slides


def _extract_slide_title(text: str) -> str:
    """Extract the first meaningful line as slide title."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        # Remove markdown formatting
        title = re.sub(r"^#+\s*", "", stripped)
        title = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", title)
        title = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", title)
        if len(title) > 2:
            return title.strip()
    return "(untitled)"


def _extract_bullets(text: str) -> List[str]:
    """Extract bullet points from slide text."""
    bullets: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        m = _BULLET_RE.match(stripped)
        if m:
            bullet_text = m.group(1).strip()
            if bullet_text and len(bullet_text) > 2:
                bullets.append(bullet_text)
    return bullets


# ═══════════════════════════════════════════════════════════════════════════
# 3. Heuristic topic inference
# ═══════════════════════════════════════════════════════════════════════════

# (keyword_patterns, category_label)
# Patterns are matched case-insensitively against slide title + content.
# Order matters for overlapping terms — first match wins per category per slide.
_TOPIC_PATTERNS: List[Tuple[List[str], str]] = [
    # ── research questions ──
    (["研究问题", "研究目标", "研究动机", "问题陈述", "要解决",
      "research question", "research objective", "research goal",
      "motivation", "investigate", "relationship between",
      "this work studies", "we aim to", "our goal is",
      "objective", "goal", "problem statement"], "research_questions"),
    # ── completed work ──
    (["已完成", "已完成工作", "实现了", "构建了", "开发了", "完成",
      "completed", "implemented", "built", "constructed",
      "collected", "created", "finished", "developed"], "completed_work"),
    # ── experiments ──
    (["实验", "实验设计", "指标", "结果", "测试", "验证", "评估",
      "experiment", "evaluation", "benchmark", "metric",
      "result", "ablation", "dataset", "test"], "experiments"),
    # ── findings ──
    (["发现", "观察到", "结果表明", "我们发现", "可以看出", "可以看到", "结论",
      "finding", "observation", "shows", "indicates",
      "reveals", "suggests", "demonstrates"], "findings"),
    # ── issues / limitations ──
    (["问题", "局限", "不足", "错误", "挑战", "难点", "困难", "未解决",
      "limitation", "issue", "failure", "error", "challenge",
      "failed", "problem"], "issues"),
    # ── next steps ──
    (["下一步", "后续", "计划", "将来", "接下来", "后续工作", "展望",
      "将会", "我们将", "next step", "future work", "todo",
      "plan", "will", "further"], "next_steps"),
]


def _extract_keywords(slides: List[Dict[str, Any]]) -> List[str]:
    """Extract unique keywords from slide titles and bullets."""
    seen: set = set()
    keywords: List[str] = []
    # Common research keywords to look for
    keyword_indicators = [
        "bias", "fairness", "hallucination", "evaluation", "benchmark",
        "multimodal", "vision-language", "VLM", "LVLM", "language model",
        "dataset", "annotation", "metric", "accuracy", "recall", "precision",
        "stereotype", "gender", "attribute", "bounding box", "caption",
        "generation", "detection", "classification", "retrieval",
        "偏见", "公平性", "幻觉", "评估", "多模态", "视觉语言",
        "数据集", "标注", "指标", "准确率", "召回率", "精确率",
        "刻板印象", "性别", "属性", "描述", "生成", "检测",
    ]
    for slide in slides:
        text = (slide.get("title", "") + " " + slide.get("content", "")).lower()
        for kw in keyword_indicators:
            if kw.lower() in text and kw.lower() not in seen:
                seen.add(kw.lower())
                keywords.append(kw)
    return keywords[:20]


def infer_progress_topics(slides: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Heuristic extraction of research progress topics from slides.

    Scans slide titles, bullets, and body text for keyword patterns.

    Returns::

        {
            "research_questions": [str, ...],
            "completed_work": [str, ...],
            "experiments": [str, ...],
            "findings": [str, ...],
            "issues": [str, ...],
            "next_steps": [str, ...],
            "keywords": [str, ...],
        }
    """
    # Build searchable text
    all_text_parts: List[str] = []
    for slide in slides:
        all_text_parts.append(slide.get("title", ""))
        all_text_parts.append(slide.get("content", ""))
        for b in slide.get("bullets", []):
            all_text_parts.append(b)
    all_text = "\n".join(all_text_parts)
    all_text_lower = all_text.lower()

    topics: Dict[str, List[str]] = defaultdict(list)

    # Slide-level matching: check each slide against patterns
    for slide in slides:
        slide_title = slide.get("title", "")
        slide_content = slide.get("content", "")
        bullets = slide.get("bullets", [])
        slide_text = (slide_title + "\n" + slide_content).lower()

        for patterns, category in _TOPIC_PATTERNS:
            matched = False
            for pat in patterns:
                if pat.lower() in slide_text:
                    matched = True
                    break
            if matched:
                # Extract the slide content as evidence
                evidence = slide_title if slide_title else slide_content[:120].strip()
                if evidence and evidence not in topics[category]:
                    topics[category].append(evidence)

    # Also check against bullet level
    for slide in slides:
        for bullet in slide.get("bullets", []):
            bullet_lower = bullet.lower()
            for patterns, category in _TOPIC_PATTERNS:
                for pat in patterns:
                    if pat.lower() in bullet_lower and bullet not in topics[category]:
                        topics[category].append(bullet)
                        break

    # Keywords
    keywords = _extract_keywords(slides)

    return {
        "research_questions": topics.get("research_questions", []),
        "completed_work": topics.get("completed_work", []),
        "experiments": topics.get("experiments", []),
        "findings": topics.get("findings", []),
        "issues": topics.get("issues", []),
        "next_steps": topics.get("next_steps", []),
        "keywords": keywords,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Build progress memory document
# ═══════════════════════════════════════════════════════════════════════════

def _or_na(items: List[str], singular: str = "item") -> str:
    """Format a list or a 'not provided' message."""
    if not items:
        return f"*(Current PPT did not explicitly provide this {singular}.)*"
    return "\n".join(f"- {item}" for item in items)


def build_progress_memory_doc(slide_data: Dict[str, Any]) -> str:
    """
    Generate a structured Markdown progress-memory document.

    Parameters
    ----------
    slide_data : dict
        Output of :func:`generate_progress_memory` (or a dict with keys
        *metadata*, *slides*, *topics*).

    Returns
    -------
    str
        Markdown document.
    """
    metadata = slide_data.get("metadata", {})
    slides: List[Dict] = slide_data.get("slides", [])
    topics: Dict = slide_data.get("topics", {})
    today = date.today().isoformat()

    title = metadata.get("title", metadata.get("original_path", "Untitled Presentation"))
    source = metadata.get("original_path", metadata.get("path", "unknown"))

    lines: List[str] = []
    lines.append("# Research Progress Memory\n")
    lines.append(f"*Generated: {today}*\n")

    # ── 1. Presentation Metadata ────────────────────────────────────
    lines.append("## 1. Presentation Metadata\n")
    lines.append(f"- **Title**: {title}")
    lines.append(f"- **Source**: {source}")
    lines.append(f"- **Slide Count**: {len(slides)}")
    if metadata.get("created_from"):
        lines.append(f"- **Created From**: {metadata['created_from']}")
    if metadata.get("ingestion_backend"):
        lines.append(f"- **Ingestion Backend**: {metadata['ingestion_backend']}")
    lines.append("")

    # ── 2. Slide-by-slide Summary ───────────────────────────────────
    lines.append("## 2. Slide-by-slide Summary\n")
    for s in slides:
        num = s.get("slide_number", "?")
        stitle = s.get("title", "(untitled)")
        bullets = s.get("bullets", [])
        lines.append(f"### Slide {num}: {stitle}\n")
        if bullets:
            for b in bullets[:8]:  # max 8 bullets per slide
                lines.append(f"- {b}")
            if len(bullets) > 8:
                lines.append(f"  *... and {len(bullets) - 8} more bullets*")
        else:
            content_preview = s.get("content", "")[:200].strip().replace("\n", " ")
            if content_preview:
                lines.append(f"*{content_preview}*")
            else:
                lines.append("*(no content)*")
        lines.append("")

    # ── 3. Research Questions ───────────────────────────────────────
    lines.append("## 3. Research Questions\n")
    lines.append(_or_na(topics.get("research_questions", []), "section"))
    lines.append("")

    # ── 4. Completed Work ───────────────────────────────────────────
    lines.append("## 4. Completed Work\n")
    lines.append(_or_na(topics.get("completed_work", []), "section"))
    lines.append("")

    # ── 5. Experiments and Results ──────────────────────────────────
    lines.append("## 5. Experiments and Results\n")
    experiments = topics.get("experiments", [])
    findings = topics.get("findings", [])
    if experiments:
        lines.append("### Experiments\n")
        lines.append(_or_na(experiments))
        lines.append("")
    if findings:
        lines.append("### Findings\n")
        lines.append(_or_na(findings))
        lines.append("")
    if not experiments and not findings:
        lines.append("*(Current PPT did not explicitly provide this section.)*")
    lines.append("")

    # ── 6. Issues / Limitations ─────────────────────────────────────
    lines.append("## 6. Issues / Limitations\n")
    lines.append(_or_na(topics.get("issues", []), "section"))
    lines.append("")

    # ── 7. Next Steps ───────────────────────────────────────────────
    lines.append("## 7. Next Steps\n")
    lines.append(_or_na(topics.get("next_steps", []), "section"))
    lines.append("")

    # ── 8. Long-term Memory Records ─────────────────────────────────
    lines.append("## 8. Long-term Memory Records\n")
    memory_records = _build_memory_records(metadata, slides, topics, today)
    if memory_records:
        for rec in memory_records:
            lines.append(f"- {rec}")
    else:
        lines.append("*(No memory records could be inferred.)*")
    lines.append("")

    # ── Keywords ────────────────────────────────────────────────────
    keywords = topics.get("keywords", [])
    if keywords:
        lines.append("## Keywords\n")
        lines.append(", ".join(keywords))
        lines.append("")

    return "\n".join(lines)


def _build_memory_records(
    metadata: Dict,
    slides: List[Dict],
    topics: Dict,
    today: str,
) -> List[str]:
    """Build tagged long-term memory records."""
    records: List[str] = []
    title = metadata.get("title", "Untitled")

    # Research progress record
    questions = topics.get("research_questions", [])
    if questions:
        q_summary = "; ".join(questions[:3])
        records.append(f"[research_progress] ({today}) {title}: {q_summary}")

    # Experiment update
    experiments = topics.get("experiments", [])
    if experiments:
        exp_summary = "; ".join(experiments[:3])
        records.append(f"[experiment_update] ({today}) {title}: {exp_summary}")

    # Completed work
    completed = topics.get("completed_work", [])
    if completed:
        c_summary = "; ".join(completed[:3])
        records.append(f"[task_completed] ({today}) {title}: {c_summary}")

    # Open issues
    issues = topics.get("issues", [])
    if issues:
        i_summary = "; ".join(issues[:3])
        records.append(f"[open_issue] ({today}) {title}: {i_summary}")

    # Next steps
    next_steps = topics.get("next_steps", [])
    if next_steps:
        n_summary = "; ".join(next_steps[:3])
        records.append(f"[next_step] ({today}) {title}: {n_summary}")

    # Slide count
    records.append(f"[slide_summary] ({today}) {title}: {len(slides)} slides presented")

    return records


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def generate_progress_memory(path: Path, use_llm: bool = False, save_memory: bool = False) -> Dict[str, Any]:
    """
    Generate a Research Progress Memory from a slide markdown file.

    Parameters
    ----------
    path : Path
        Path to a slide_doc ``.md`` file (from ingestion or hand-written).
    use_llm : bool
        If True, attempt LLM enhancement of the progress memory.
        Falls back gracefully to rule-based output when LLM is unavailable.
    save_memory : bool
        If True, write the result to the Memory Store via
        memory.adapters.save_progress_memory_result().

    Returns
    -------
    dict
        Keys: *source_path*, *metadata*, *slides*, *topics*,
        *progress_memory* (str), *memory_records* (list),
        *used_llm* (bool), *llm_error* (str),
        *memory_saved* (bool), *memory_result* (dict | None).
    """
    loaded = load_slide_markdown(path)
    metadata = loaded["metadata"]
    content = loaded["content"]

    slides = parse_slides_from_markdown(content)
    topics = infer_progress_topics(slides)
    progress_memory = build_progress_memory_doc({
        "metadata": metadata,
        "slides": slides,
        "topics": topics,
    })

    # Build memory records list
    today = date.today().isoformat()
    memory_records = _build_memory_records(metadata, slides, topics, today)

    # LLM enhancement (if requested and available)
    used_llm = False
    llm_error = ""
    if use_llm:
        try:
            from research_agent.llm.enhancers import enhance_progress_memory as _epm
            enhanced = _epm(metadata, slides, topics, progress_memory)
            if enhanced["used_llm"]:
                progress_memory = enhanced["text"]
                used_llm = True
            if enhanced.get("error"):
                llm_error = enhanced["error"]
        except Exception as e:
            llm_error = f"{type(e).__name__}: {e}"

    result = {
        "source_path": str(path),
        "metadata": metadata,
        "slides": slides,
        "topics": topics,
        "progress_memory": progress_memory,
        "memory_records": memory_records,
        "used_llm": used_llm,
        "llm_error": llm_error,
    }

    # Optional memory write-back
    if save_memory:
        try:
            from research_agent.memory.adapters import save_progress_memory_result
            memory_result = save_progress_memory_result(result, auto_write=True)
        except Exception as e:
            memory_result = {"ok": False, "error": str(e)}

        result["memory_saved"] = bool(memory_result.get("ok"))
        result["memory_result"] = memory_result
    else:
        result["memory_saved"] = False
        result["memory_result"] = None

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6. Batch processing
# ═══════════════════════════════════════════════════════════════════════════

def batch_generate_progress_memory(
    input_dir: Path,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Scan *input_dir* for ``.md`` files, generate a progress-memory
    document for each, and write results to *output_dir*.

    Returns a list of result dicts (one per file).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for md_path in sorted(input_dir.glob("*.md")):
        try:
            result = generate_progress_memory(md_path)
        except Exception as exc:
            results.append({
                "source_path": str(md_path),
                "status": "error",
                "error": str(exc),
            })
            continue

        # Write progress memory markdown
        out_path = output_dir / f"{md_path.stem}_progress.md"
        out_path.write_text(result["progress_memory"], encoding="utf-8")

        result["output_path"] = str(out_path)
        result["status"] = "success"
        results.append(result)

    return results
