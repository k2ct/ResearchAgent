# Multimodal Hallucination Evaluation Notes

## 基本信息

- source_type: paper_note
- title: Multimodal Hallucination Evaluation Notes
- topic: hallucination_evaluation
- year: 2025
- path: data/papers/hallucination_evaluation_note.md

## 研究问题

多模态大模型在图像描述、视觉问答和开放式生成任务中容易产生幻觉，即生成图像中不存在的对象、属性、关系或事件。

幻觉评估的核心问题是：如何判断模型输出是否被图像证据支持，以及如何量化幻觉程度。

## 方法概括

常见幻觉评估方法包括对象级幻觉检测、CHAIR-like 指标、人工标注、GPT judge、基准问答评估和 caption 与 ground truth object 的匹配分析。

对象级方法通常会从模型生成文本中抽取对象，再与图像标注对象进行比较。如果生成对象不在图像标注或人工确认范围内，就可能被计为 extra object 或 hallucinated object。

## 使用数据集 / 评估任务

常见数据来源包括 COCO、Visual Genome、VQA、GQA、AMBER、POPE 等。不同数据集适合不同粒度的幻觉评估，例如对象、属性、关系、问答一致性等。

## 主要指标

常见指标包括 hallucination rate、extra object rate、precision、recall、CHAIRs、CHAIRi、object-level error rate、GPT judge consistency 等。

## 与 ResearchAgent 项目的关系

该文档用于支撑 ResearchAgent 中 experiment_analysis 和 paper_question 任务。它可以帮助 Agent 回答“幻觉评估有哪些指标”“CHAIR-like 方法如何理解”“为什么 extra_objects 可以作为风险信号”等问题。