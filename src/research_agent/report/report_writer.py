from typing import Dict, List


def build_report_from_context(
    query: str,
    retrieved_docs: List[Dict],
    report_type: str = "group_meeting",
) -> str:
    """
    最小版 Report Writer。

    当前版本不调用 LLM，只根据 retrieved_docs 拼接结构化科研汇报草稿。
    后续可以替换为 LLM-based Report Writer。
    """
    if not retrieved_docs:
        return _build_no_evidence_report(query)

    context_text = _format_context(retrieved_docs)

    return f"""
# 科研汇报草稿

## 0. 用户需求

{query}

## 1. 研究背景

本次汇报围绕用户问题中涉及的科研任务展开。根据本地资料库检索结果，该任务主要与多模态模型的实验分析、数据集使用或偏见 / 幻觉评估相关。

## 2. 汇报主题

可以将本次汇报主题概括为：

> 基于本地科研资料库的实验 / 数据集 / 论文信息整理与分析。

## 3. 资料依据

以下内容来自 ResearchAgent 检索到的本地资料片段：

{context_text}

## 4. 可放入 PPT 的结构

建议将汇报组织为以下几个部分：

1. **研究背景**：说明为什么该问题值得研究。
2. **任务目标**：说明当前实验或资料分析想解决什么问题。
3. **数据与设置**：介绍使用的数据集、实验样本、run_tag 或文档来源。
4. **方法流程**：说明数据如何被处理、如何评估、如何得到结论。
5. **结果与分析**：总结关键指标、观察结果和潜在问题。
6. **不足与后续工作**：说明当前 demo 或实验还存在什么限制。

## 5. 初步汇报文本

本项目围绕科研场景中的资料检索、实验分析与证据追踪展开。系统首先根据用户问题判断任务类型，再基于本地知识库检索相关资料，并结合工具分析结果生成结构化回答。

从检索资料来看，当前问题可以从实验目标、数据来源、关键指标和后续分析价值四个方面展开。若用于组会汇报，建议重点说明该实验或资料在 ResearchAgent 项目中的作用，以及它如何支持后续的多模态偏见、幻觉评估或数据集构建工作。

## 6. 后续可补充内容

- 加入更具体的实验指标数值。
- 补充对应 CSV / JSONL 工具分析结果。
- 增加图表或表格展示。
- 将本草稿进一步压缩成 PPT 页面文案。
""".strip()


def _format_context(retrieved_docs: List[Dict], max_docs: int = 3, max_chars: int = 500) -> str:
    lines = []

    for i, doc in enumerate(retrieved_docs[:max_docs], start=1):
        metadata = doc.get("metadata", {})
        content = doc.get("content", "")

        path = metadata.get("path", "unknown")
        source_type = metadata.get("source_type", "unknown")
        title = metadata.get("title", "")
        dataset = metadata.get("dataset", "")
        run_tag = metadata.get("run_tag", "")

        header = f"[{i}] source_type={source_type} | path={path}"

        if title:
            header += f" | title={title}"
        if dataset:
            header += f" | dataset={dataset}"
        if run_tag:
            header += f" | run_tag={run_tag}"

        preview = content[:max_chars].replace("\n", " ")

        lines.append(f"{header}\n{preview}")

    return "\n\n".join(lines)


def _build_no_evidence_report(query: str) -> str:
    return f"""
# 科研汇报草稿

## 用户需求

{query}

## 当前状态

暂未从本地资料库检索到足够相关的资料，因此无法生成有明确证据支撑的科研汇报草稿。

## 建议

请补充以下信息之一：

1. 具体论文名称；
2. 实验 run_tag；
3. 数据集名称；
4. CSV / JSONL 实验文件路径；
5. 希望汇报的主题或页面结构。
""".strip()