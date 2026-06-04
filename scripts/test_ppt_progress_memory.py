"""
Test suite for PPT / Research Progress Memory module.

Usage::

    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_ppt_progress_memory.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Fix encoding on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.progress.ppt_progress_memory import (
    load_slide_markdown,
    parse_slides_from_markdown,
    infer_progress_topics,
    build_progress_memory_doc,
    generate_progress_memory,
    batch_generate_progress_memory,
)


PASS = "✓ PASS"
FAIL = "✗ FAIL"


def check(condition: bool, label: str) -> bool:
    status = PASS if condition else FAIL
    print(f"  {status}: {label}")
    return condition


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════
# Sample slide markdown builder
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_SLIDES = """---
source_type: slide_doc
title: Multimodal Bias Evaluation Progress Report
doc_type: pptx
original_path: raw_docs/slides_pptx/group_meeting_20260605.pptx
created_from: ingestion_pipeline
ingestion_backend: local
tags: [slides_pptx]
---

# Multimodal Bias Evaluation — Group Meeting 2026-06-05

## Slide 1

研究背景与目标

- 研究问题：视觉语言模型（VLM）中的社会偏见如何系统评估？
- 当前 VLM 偏见评估主要依赖直接提问，容易触发安全护栏导致拒答
- 目标：构建不依赖护栏的隐式偏见审计框架
- 本次汇报总结近两周实验进展

## Slide 2

已完成工作

- 完成了 Guardrail-Agnostic 论文的精读与笔记整理
- 构建了 VIGNETTE 基准的本地评估 pipeline
- 实现了基于 COCO 2017 的幻觉筛选实验 coco_val_n100_g1 和 coco_val_n300_g1
- 开发了 SD3.5 Gender Swap 受控图像生成脚本

## Slide 3

实验结果与指标

- coco_val_n300_g1 的 mean_extra_object_rate 为 0.23
- hrs_v1 综合幻觉风险排序分数可用于筛选高风险图像
- SD3.5 Gender Swap 实验中，不同性别条件下的对象幻觉存在差异
- 初步发现：male→female swap 场景下 extra_objects 数量略高于反向

## Slide 4

当前问题与局限

- COCO 数据集对象类别有限，无法覆盖开放世界所有对象
- 当前幻觉检测 pipeline 未考虑场景上下文
- SD3.5 生成图像质量不稳定，部分 seed 组合出现 artifacts
- 缺少人工标注的 ground truth 用于验证幻觉检测准确率
- 现有偏见评估依赖单一维度（性别），缺乏交叉性分析

## Slide 5

下一步计划

- 构建 stereotype library 用于多维度偏见审计
- 引入 OpenImages-MIAP 进行 bbox 级属性偏见分析
- 扩展实验到更多 VLM 模型（LLaVA, InstructBLIP）
- 开发自动化实验报告生成模块
- 准备下次组会汇报的 cross-model comparison 结果
"""


def create_sample_slide_md() -> Path:
    """Create a sample slide markdown for testing. Returns the path."""
    out_path = PROJECT_ROOT / "data" / "ingested" / "sample_progress_test_slides.md"
    out_path.write_text(SAMPLE_SLIDES, encoding="utf-8")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_load_slide_markdown():
    section("Test 1: load_slide_markdown")
    path = create_sample_slide_md()
    loaded = load_slide_markdown(path)
    ok = True
    ok &= check("metadata" in loaded, "has metadata key")
    ok &= check("content" in loaded, "has content key")
    ok &= check("path" in loaded, "has path key")
    ok &= check(loaded["metadata"].get("source_type") == "slide_doc",
                f"source_type = {loaded['metadata'].get('source_type')}")
    ok &= check("## Slide 1" in loaded["content"], "content contains ## Slide 1")
    return ok


def test_parse_slides():
    section("Test 2: parse_slides_from_markdown")
    path = create_sample_slide_md()
    loaded = load_slide_markdown(path)
    slides = parse_slides_from_markdown(loaded["content"])

    ok = True
    ok &= check(len(slides) == 5, f"Found {len(slides)} slides (expected 5)")
    for s in slides:
        ok &= check("slide_number" in s, f"Slide {s.get('slide_number')}: has slide_number")
        ok &= check("title" in s, f"Slide {s.get('slide_number')}: has title")
        ok &= check("bullets" in s, f"Slide {s.get('slide_number')}: has bullets")
        ok &= check("content" in s, f"Slide {s.get('slide_number')}: has content")

    # Check titles
    titles = [s["title"] for s in slides]
    ok &= check(any("研究背景" in t for t in titles), "Slide 1 title detected")
    ok &= check(any("已完成" in t for t in titles), "Slide 2 title detected")
    ok &= check(any("实验" in t for t in titles), "Slide 3 title detected")
    ok &= check(any("问题" in t for t in titles), "Slide 4 title detected")
    ok &= check(any("下一步" in t for t in titles), "Slide 5 title detected")

    # Check bullets
    total_bullets = sum(len(s["bullets"]) for s in slides)
    ok &= check(total_bullets > 10, f"Total bullets: {total_bullets} (expected > 10)")
    return ok


def test_infer_topics():
    section("Test 3: infer_progress_topics")
    path = create_sample_slide_md()
    loaded = load_slide_markdown(path)
    slides = parse_slides_from_markdown(loaded["content"])
    topics = infer_progress_topics(slides)

    ok = True
    ok &= check(len(topics.get("research_questions", [])) > 0,
                f"research_questions: {len(topics.get('research_questions', []))} items")
    ok &= check(len(topics.get("completed_work", [])) > 0,
                f"completed_work: {len(topics.get('completed_work', []))} items")
    ok &= check(len(topics.get("experiments", [])) > 0,
                f"experiments: {len(topics.get('experiments', []))} items")
    ok &= check(len(topics.get("issues", [])) > 0,
                f"issues: {len(topics.get('issues', []))} items")
    ok &= check(len(topics.get("next_steps", [])) > 0,
                f"next_steps: {len(topics.get('next_steps', []))} items")
    ok &= check(len(topics.get("keywords", [])) > 0,
                f"keywords: {len(topics.get('keywords', []))} items")

    # Print inferred items
    for cat in ["research_questions", "completed_work", "experiments", "issues", "next_steps", "keywords"]:
        items = topics.get(cat, [])
        print(f"  [{cat}]: {len(items)} items")
        for item in items[:3]:
            print(f"    - {item[:100]}")

    return ok


def test_build_doc():
    section("Test 4: build_progress_memory_doc")
    path = create_sample_slide_md()
    loaded = load_slide_markdown(path)
    slides = parse_slides_from_markdown(loaded["content"])
    topics = infer_progress_topics(slides)

    doc = build_progress_memory_doc({
        "metadata": loaded["metadata"],
        "slides": slides,
        "topics": topics,
    })

    ok = True
    ok &= check("Research Progress Memory" in doc, "Title present")
    ok &= check("Presentation Metadata" in doc, "Section 1 present")
    ok &= check("Slide-by-slide Summary" in doc, "Section 2 present")
    ok &= check("Research Questions" in doc, "Section 3 present")
    ok &= check("Completed Work" in doc, "Section 4 present")
    ok &= check("Experiments and Results" in doc, "Section 5 present")
    ok &= check("Issues / Limitations" in doc, "Section 6 present")
    ok &= check("Next Steps" in doc, "Section 7 present")
    ok &= check("Long-term Memory Records" in doc, "Section 8 present")
    ok &= check("[research_progress]" in doc, "Memory record: research_progress")
    ok &= check("[experiment_update]" in doc, "Memory record: experiment_update")
    ok &= check("[open_issue]" in doc, "Memory record: open_issue")
    ok &= check("[next_step]" in doc, "Memory record: next_step")

    print(f"  Doc length: {len(doc)} chars")
    print(f"\n  --- First 2000 chars ---")
    print(doc[:2000])
    return ok


def test_generate_progress_memory():
    section("Test 5: generate_progress_memory (main entry)")
    path = create_sample_slide_md()
    result = generate_progress_memory(path)

    ok = True
    ok &= check("source_path" in result, "has source_path")
    ok &= check("metadata" in result, "has metadata")
    ok &= check("slides" in result, "has slides")
    ok &= check("topics" in result, "has topics")
    ok &= check("progress_memory" in result, "has progress_memory")
    ok &= check("memory_records" in result, "has memory_records")

    # Slides
    ok &= check(len(result["slides"]) == 5, f"slide_count = {len(result['slides'])}")

    # Memory records
    records = result["memory_records"]
    ok &= check(len(records) >= 4, f"memory_records: {len(records)} (expected >= 4)")
    for r in records:
        print(f"  {r[:120]}")

    return ok


def test_batch_generate():
    section("Test 6: batch_generate_progress_memory")
    # Use the sample file created earlier
    input_dir = PROJECT_ROOT / "data" / "ingested"
    output_dir = PROJECT_ROOT / "data" / "progress_memory"

    results = batch_generate_progress_memory(input_dir, output_dir)

    ok = True
    ok &= check(len(results) > 0, f"Batch processed {len(results)} files")

    # Find our test file in results
    test_results = [r for r in results if "sample_progress_test_slides" in r.get("source_path", "")]
    if test_results:
        tr = test_results[0]
        ok &= check(tr.get("status") == "success", f"batch status: {tr.get('status')}")
        out_path = tr.get("output_path", "")
        ok &= check(Path(out_path).exists(), f"Output file exists: {out_path}")
        if Path(out_path).exists():
            size = Path(out_path).stat().st_size
            ok &= check(size > 500, f"Output size: {size} chars (expected > 500)")
            print(f"  Output: {out_path} ({size} bytes)")
    else:
        ok &= check(False, "Test file not found in batch results")

    ok &= check(output_dir.exists(), f"Output dir exists: {output_dir}")
    return ok


def test_no_slide_markers():
    section("Test 7: parse_slides without ## Slide N markers")
    plain_md = """# My Presentation

## Introduction

- Background info
- Research goal

## Methods

- Approach A
- Approach B

## Results

- Finding 1
- Finding 2
"""

    slides = parse_slides_from_markdown(plain_md)
    ok = True
    # H1 "My Presentation" + 3 H2 sections = 4 sections
    ok &= check(len(slides) >= 3, f"Fallback: {len(slides)} sections (expected >= 3)")
    if slides:
        titles = [s["title"] for s in slides]
        ok &= check("Introduction" in titles, "Introduction section found")
        ok &= check("Methods" in titles, "Methods section found")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 60)
    print("  PPT Progress Memory — Test Suite")
    print("=" * 60)

    results = {}
    results["load"] = test_load_slide_markdown()
    results["parse"] = test_parse_slides()
    results["topics"] = test_infer_topics()
    results["build_doc"] = test_build_doc()
    results["generate"] = test_generate_progress_memory()
    results["batch"] = test_batch_generate()
    results["no_markers"] = test_no_slide_markers()

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    all_passed = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}: {name}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("  All tests passed!")
    else:
        print("  Some tests FAILED — see details above.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
