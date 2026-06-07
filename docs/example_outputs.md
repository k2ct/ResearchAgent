# ResearchAgent Example Outputs

> 本文档展示 ResearchAgent v0.5 Memory + Multi-Agent Preview 在 6 种任务类型（共 8 个示例）上的典型输出示例。
>
> 每个示例均基于项目实际功能（LangGraph 工作流、Agentic RAG、CSV/JSONL 工具调用、Evidence Checker、Report Writer / LLM-assisted Report Writer）生成。

---

## 目录

- [Task 1: paper_question — 论文问答](#task-1-paper_question)
- [Task 2: experiment_analysis — 实验分析（CSV 工具）](#task-2-experiment_analysis)
- [Task 3: experiment_analysis — 实验分析（JSONL 工具）](#task-3-experiment_analysis-jsonl)
- [Task 4: dataset_recommendation — 数据集推荐](#task-4-dataset_recommendation)
- [Task 5: report_generation — 组会汇报（模板版）](#task-5-report_generation)
- [Task 6: report_generation — PPT 文案（LLM 版）](#task-6-report_generation-llm)
- [Task 7: code_help — 代码辅助](#task-7-code_help)
- [Task 8: general — 通用科研助手](#task-8-general)

---

## Task 1: paper_question

论文问答任务。系统根据问题从论文笔记资料库（`paper_note`）中检索相关内容，生成基于检索资料的初步回答。

### 示例 1.1：论文方法理解

**Input:**
```
Guardrail-Agnostic 这篇论文的核心方法是什么？如何避免直接触发模型安全拒答？
```

**task_type:** `paper_question`

**classifier_source:** `rule`

**route_reason:** `命中了论文、paper 或 related work 相关关键词。`

**tool_used:** `none`

**evidence_status:** `passed`

**evidence_reason:** `回答包含 RAG 检索来源，可追踪到本地资料库。`

**Sources:**
1. [paper_note] data/papers/guardrail_agnostic.md (title=Guardrail-Agnostic Societal Bias Evaluation in Large Vision-Language Models)
2. [paper_note] data/papers/multimodal_bias_survey.md (title=Multimodal Bias and Stereotype Evaluation Survey)

**Output:**

```
这是论文问答任务。系统已从论文笔记资料库中检索到相关内容。

基于检索资料的初步回答：

用户问题：Guardrail-Agnostic 这篇论文的核心方法是什么？如何避免直接触发模型安全拒答？

检索到的相关资料摘要如下：
[1] | source_type=paper_note | path=data/papers/guardrail_agnostic.md | title=Guardrail-Agnostic Societal Bias Evaluation in Large Vision-Language Models
论文的核心思想是设计 guardrail-agnostic evaluation，即尽量不直接触发模型安全拒答，而是通过感知、事实性、决策等任务观察模型输出中的系统性偏差。这种方法强调偏见评估不一定依赖直接询问敏感属性，而可以通过任务表现差异、选项偏好、描述倾向等方式进行测量...

[2] | source_type=paper_note | path=data/papers/multimodal_bias_survey.md | title=Multimodal Bias and Stereotype Evaluation Survey
常见方法包括构造对照图像、替换人物属性、控制场景和对象、设计问答模板、比较不同群体条件下的输出差异，以及分析 embedding space 中的偏见方向...

说明：当前版本在未启用 LLM 时使用检索内容摘要作为回答，启用 LLM 后会自动切换为 LLM-based 综合生成。
```

---

### 示例 1.2：论文概念解释

**Input:**
```
多模态幻觉评估中的 CHAIR 指标是什么？额外对象率是怎么算的？
```

**task_type:** `paper_question`

**classifier_source:** `llm`

**route_reason:** `用户询问多模态幻觉评估中的 CHAIR 指标和额外对象率计算方法，属于论文相关问题。`

**tool_used:** `none`

**evidence_status:** `passed`

**evidence_reason:** `回答包含 RAG 检索来源，可追踪到本地资料库。`

**Sources:**
1. [paper_note] data/papers/hallucination_evaluation_note.md (title=Multimodal Hallucination Evaluation Notes)

**Output:**

```
这是论文问答任务。系统已从论文笔记资料库中检索到相关内容。

基于检索资料的初步回答：

用户问题：多模态幻觉评估中的 CHAIR 指标是什么？额外对象率是怎么算的？

检索到的相关资料摘要如下：
[1] | source_type=paper_note | path=data/papers/hallucination_evaluation_note.md | title=Multimodal Hallucination Evaluation Notes
常见幻觉评估方法包括对象级幻觉检测、CHAIR-like 指标、人工标注、GPT judge、基准问答评估和 caption 与 ground truth object 的匹配分析。对象级方法通常会从模型生成文本中抽取对象，再与图像标注对象进行比较。如果生成对象不在图像标注或人工确认范围内，就可能被计为 extra object 或 hallucinated object。常见指标包括 hallucination rate、extra object rate、precision、recall、CHAIRs、CHAIRi、object-level error rate、GPT judge consistency 等...

说明：当前版本在未启用 LLM 时使用检索内容摘要作为回答，启用 LLM 后会自动切换为 LLM-based 综合生成。
```

---

## Task 2: experiment_analysis

实验分析任务。系统同时进行 RAG 检索（实验资料库）和本地文件工具调用（CSV/JSONL 分析），综合两者生成回答。

### 示例 2.1：CSV 文件分析

**Input:**
```
请分析 data/experiments/sample_metrics.csv
```

**task_type:** `experiment_analysis`

**classifier_source:** `rule`

**route_reason:** `命中了实验、文件路径、CSV/JSONL、run_tag、coco_val、幻觉或 benchmark 相关关键词。`

**tool_used:** `csv_analyzer`

**evidence_status:** `passed`

**evidence_reason:** `回答同时包含 RAG 检索来源和成功的工具分析结果。`

**Sources:**
1. [experiment_doc] data/experiments/coco_val_n100_g1.md (run_tag=coco_val_n100_g1, dataset=COCO2017)
2. [experiment_doc] data/experiments/coco_val_n300_g1.md (run_tag=coco_val_n300_g1, dataset=COCO2017)
3. [experiment_doc] data/experiments/sd35_gender_swap.md (run_tag=sd35_gender_swap, dataset=synthetic_sd35)

**Output:**

```
这是实验分析任务。系统已完成 RAG 检索，并根据需要尝试调用本地实验文件分析工具。

一、工具分析

本地实验文件分析结果：
文件：data/experiments/sample_metrics.csv
类型：csv
行数：3
列数：7

列名：
- run_tag (object)
- dataset (object)
- mean_extra_object_rate (float64)
- mean_precision (float64)
- mean_recall (float64)
- hrs_v1 (float64)
- Unnamed: 6 (float64)

缺失值字段：
- Unnamed: 6: 3

数值列摘要：
- mean_extra_object_rate: mean=0.1867, min=0.1500, max=0.2300, median=0.1800
- mean_precision: mean=0.8167, min=0.7800, max=0.8500, median=0.8200
- mean_recall: mean=0.7133, min=0.6900, max=0.7400, median=0.7100
- hrs_v1: mean=0.3267, min=0.2800, max=0.3900, median=0.3100

低基数字段分布：
- run_tag: {'coco_val_n100_g1': 1, 'coco_val_n300_g1': 1, 'sd35_gender_swap': 1}
- dataset: {'COCO2017': 2, 'synthetic_sd35': 1}

前几行预览：
[1] {'run_tag': 'coco_val_n100_g1', 'dataset': 'COCO2017', 'mean_extra_object_rate': 0.18, 'mean_precision': 0.82, 'mean_recall': 0.74, 'hrs_v1': 0.31, 'Unnamed: 6': nan}
[2] {'run_tag': 'coco_val_n300_g1', 'dataset': 'COCO2017', 'mean_extra_object_rate': 0.23, 'mean_precision': 0.78, 'mean_recall': 0.71, 'hrs_v1': 0.39, 'Unnamed: 6': nan}
[3] {'run_tag': 'sd35_gender_swap', 'dataset': 'synthetic_sd35', 'mean_extra_object_rate': 0.15, 'mean_precision': 0.85, 'mean_recall': 0.69, 'hrs_v1': 0.28, 'Unnamed: 6': nan}

二、相关实验资料检索

用户问题：请分析 data/experiments/sample_metrics.csv

检索到的相关资料摘要如下：
[1] | source_type=experiment_doc | path=data/experiments/coco_val_n100_g1.md | run_tag=coco_val_n100_g1 | dataset=COCO2017
实验是一个小规模 COCO validation 幻觉风险筛选实验，目标是在 100 张图像样本上测试图像描述模型是否会生成图像中不存在的对象。该实验适合作为 smoke test 或小规模调试，不适合得出稳定统计结论...

说明：当前版本在未启用 LLM 时使用检索内容摘要作为回答，启用 LLM 后会自动切换为 LLM-based 综合生成。
```

---

## Task 3: experiment_analysis (JSONL)

### 示例 3.1：JSONL 文件分析

**Input:**
```
请分析 data/experiments/sample_generations.jsonl，看看幻觉标注的整体情况
```

**task_type:** `experiment_analysis`

**classifier_source:** `rule`

**route_reason:** `命中了实验、文件路径、CSV/JSONL、run_tag、coco_val、幻觉或 benchmark 相关关键词。`

**tool_used:** `jsonl_analyzer`

**evidence_status:** `passed`

**evidence_reason:** `回答同时包含 RAG 检索来源和成功的工具分析结果。`

**Sources:**
1. [experiment_doc] data/experiments/coco_val_n100_g1.md (run_tag=coco_val_n100_g1, dataset=COCO2017)
2. [experiment_doc] data/experiments/coco_val_n300_g1.md (run_tag=coco_val_n300_g1, dataset=COCO2017)
3. [experiment_doc] data/experiments/sd35_gender_swap.md (run_tag=sd35_gender_swap, dataset=synthetic_sd35)

**Output:**

```
这是实验分析任务。系统已完成 RAG 检索，并根据需要尝试调用本地实验文件分析工具。

一、工具分析

本地实验文件分析结果：
文件：data/experiments/sample_generations.jsonl
类型：jsonl
记录数：5
解析错误数：0

字段列表：
- image_id: {'str': 5}
- run_tag: {'str': 5}
- dataset: {'str': 5}
- prompt: {'str': 5}
- response: {'str': 5}
- extra_objects: {'list': 5}
- hallucination: {'bool': 5}

列表字段长度统计：
- extra_objects: count=5, mean_length=0.60, min=0, max=1

布尔字段分布：
- hallucination: true=2, false=3

低基数字段分布：
- run_tag: {'coco_val_n100_g1': 2, 'coco_val_n300_g1': 2, 'sd35_gender_swap': 1}
- dataset: {'COCO2017': 4, 'synthetic_sd35': 1}
- hallucination: {'False': 3, 'True': 2}
- prompt: {'Describe the image.': 5}

前几条记录预览：
[1] {'image_id': '000001', 'run_tag': 'coco_val_n100_g1', 'dataset': 'COCO2017', 'prompt': 'Describe the image.', 'response': 'A man riding a bicycle on the street.', 'extra_objects': [], 'hallucination': False}
[2] {'image_id': '000002', 'run_tag': 'coco_val_n100_g1', 'dataset': 'COCO2017', 'prompt': 'Describe the image.', 'response': 'A woman holding an umbrella in a kitchen.', 'extra_objects': ['umbrella'], 'hallucination': True}
[3] {'image_id': '000003', 'run_tag': 'coco_val_n300_g1', 'dataset': 'COCO2017', 'prompt': 'Describe the image.', 'response': 'A dog sitting beside a child in a park.', 'extra_objects': [], 'hallucination': False}
[4] {'image_id': '000004', 'run_tag': 'coco_val_n300_g1', 'dataset': 'COCO2017', 'prompt': 'Describe the image.', 'response': 'A doctor standing in a hospital room with a stethoscope.', 'extra_objects': ['stethoscope'], 'hallucination': True}
[5] {'image_id': '000005', 'run_tag': 'sd35_gender_swap', 'dataset': 'synthetic_sd35', 'prompt': 'Describe the image.', 'response': 'A person working in an office with a laptop.', 'extra_objects': ['laptop'], 'hallucination': False}

二、相关实验资料检索

用户问题：请分析 data/experiments/sample_generations.jsonl，看看幻觉标注的整体情况

检索到的相关资料摘要如下：
[1] | source_type=experiment_doc | path=data/experiments/coco_val_n100_g1.md | run_tag=coco_val_n100_g1 | dataset=COCO2017
核心指标包括 mean_extra_object_rate、mean_precision、mean_recall、repeat_rate、prompt_consistency、hrs_v1 和 union_extra_objects...

说明：当前版本在未启用 LLM 时使用检索内容摘要作为回答，启用 LLM 后会自动切换为 LLM-based 综合生成。
```

---

## Task 4: dataset_recommendation

数据集推荐/数据集说明任务。系统根据问题从数据集资料库（`dataset_doc`）中检索相关内容。

### 示例 4.1：数据集属性查询

**Input:**
```
OpenImages-MIAP 的性别标注是图像级还是 bbox 级？
```

**task_type:** `dataset_recommendation`

**classifier_source:** `rule`

**route_reason:** `命中了数据集、dataset、OpenImages、MIAP、FairFace、GQA 或 COCO 相关关键词。`

**tool_used:** `none`

**evidence_status:** `passed`

**evidence_reason:** `回答包含 RAG 检索来源，可追踪到本地资料库。`

**Sources:**
1. [dataset_doc] data/datasets/openimages_miap.md (dataset=OpenImages-MIAP)

**Output:**

```
这是数据集推荐 / 数据集说明任务。系统已从数据集资料库中检索到相关内容。

基于检索资料的初步回答：

用户问题：OpenImages-MIAP 的性别标注是图像级还是 bbox 级？

检索到的相关资料摘要如下：
[1] | source_type=dataset_doc | path=data/datasets/openimages_miap.md | dataset=OpenImages-MIAP
OpenImages-MIAP 是基于 OpenImages 的人物属性相关标注数据，包含人物 bounding box 以及与人物相关的属性标注，例如感知性别表达和年龄表达等。该数据集的关键特点是属性更接近 person-level 或 bbox-level，而不是简单的 image-level 标签。该数据集提供人物框级别信息，因此比单纯图像级标签更细。对于一张图中有多个人物的情况，可以分别分析不同人物框的属性。局限：OpenImages-MIAP 的属性不应被粗暴地当作整张图像的唯一属性标签。如果将 bbox-level 属性直接聚合为 image-level 标签，可能会误解多人物图像的真实情况...

说明：当前版本在未启用 LLM 时使用检索内容摘要作为回答，启用 LLM 后会自动切换为 LLM-based 综合生成。
```

---

### 示例 4.2：数据集适用场景推荐

**Input:**
```
我想做视觉问答和场景图关系推理，应该用什么数据集？
```

**task_type:** `dataset_recommendation`

**classifier_source:** `llm`

**route_reason:** `用户询问视觉问答和场景图关系推理的适用数据集，属于数据集推荐问题。`

**tool_used:** `none`

**evidence_status:** `passed`

**evidence_reason:** `回答包含 RAG 检索来源，可追踪到本地资料库。`

**Sources:**
1. [dataset_doc] data/datasets/gqa.md (dataset=GQA)
2. [dataset_doc] data/datasets/coco.md (dataset=COCO2017)
3. [dataset_doc] data/datasets/fairface.md (dataset=FairFace)

**Output:**

```
这是数据集推荐 / 数据集说明任务。系统已从数据集资料库中检索到相关内容。

基于检索资料的初步回答：

用户问题：我想做视觉问答和场景图关系推理，应该用什么数据集？

检索到的相关资料摘要如下：
[1] | source_type=dataset_doc | path=data/datasets/gqa.md | dataset=GQA
GQA 是基于真实图像和 scene graph 构建的视觉问答数据集，包含对象、属性、关系以及结构化问答信息。它强调组合推理和视觉关系理解，比单纯对象检测数据集更适合分析场景图和关系推理。GQA 提供结构化 scene graph 和问题答案，有助于研究对象关系、属性组合和复杂视觉推理...

[2] | source_type=dataset_doc | path=data/datasets/coco.md | dataset=COCO2017
COCO 是常用的图像理解数据集，包含图像、对象类别、bounding box、segmentation 和 caption 标注。在幻觉评估中，COCO 常用于对象级幻觉检测，因为它提供较清晰的对象类别标注...

说明：当前版本在未启用 LLM 时使用检索内容摘要作为回答，启用 LLM 后会自动切换为 LLM-based 综合生成。
```

---

## Task 5: report_generation

汇报生成任务。系统对 report_generation 问题做全库 RAG 检索，优先尝试 LLM Report Writer，不可用时回退到模板版 Report Writer。

### 示例 5.1：组会汇报草稿（模板版 Template Fallback）

> **说明：** 此示例展示未启用 LLM Report Writer（`ENABLE_LLM_REPORT_WRITER=false`）或 API Key 未配置时的模板版输出。

**Input:**
```
帮我生成 coco_val_n300_g1 实验的组会汇报文本
```

**task_type:** `report_generation`

**classifier_source:** `rule`

**route_reason:** `命中了汇报、PPT、组会、文案、草稿、报告或 presentation 相关关键词。`

**tool_used:** `none`

**evidence_status:** `passed`

**evidence_reason:** `回答包含 RAG 检索来源，可追踪到本地资料库。`

**Sources:**
1. [experiment_doc] data/experiments/coco_val_n300_g1.md (run_tag=coco_val_n300_g1, dataset=COCO2017)
2. [experiment_doc] data/experiments/coco_val_n100_g1.md (run_tag=coco_val_n100_g1, dataset=COCO2017)
3. [paper_note] data/papers/hallucination_evaluation_note.md (title=Multimodal Hallucination Evaluation Notes)
4. [dataset_doc] data/datasets/coco.md (dataset=COCO2017)

**Output:**

```
这是汇报生成任务。系统已根据本地科研资料库生成结构化汇报草稿。

生成方式：
Template fallback Report Writer (LLM unavailable: LLM Report Writer is disabled.)

# 科研汇报草稿

## 0. 用户需求

帮我生成 coco_val_n300_g1 实验的组会汇报文本

## 1. 研究背景

本次汇报围绕用户问题中涉及的科研任务展开。根据本地资料库检索结果，该任务主要与多模态模型的实验分析、数据集使用或偏见 / 幻觉评估相关。

## 2. 汇报主题

可以将本次汇报主题概括为：

> 基于本地科研资料库的实验 / 数据集 / 论文信息整理与分析。

## 3. 资料依据

以下内容来自 ResearchAgent 检索到的本地资料片段：

[1] source_type=experiment_doc | path=data/experiments/coco_val_n300_g1.md | run_tag=coco_val_n300_g1 | dataset=COCO2017
coco_val_n300_g1 是基于 COCO validation 样本的幻觉风险筛选实验。实验目标是通过模型生成描述与图像对象标注之间的差异，识别 hallucination-prone images...

[2] source_type=experiment_doc | path=data/experiments/coco_val_n100_g1.md | run_tag=coco_val_n100_g1 | dataset=COCO2017
该实验是一个小规模 COCO validation 幻觉风险筛选实验，目标是在 100 张图像样本上测试图像描述模型是否会生成图像中不存在的对象...

[3] source_type=paper_note | path=data/papers/hallucination_evaluation_note.md | title=Multimodal Hallucination Evaluation Notes
常见幻觉评估方法包括对象级幻觉检测、CHAIR-like 指标、人工标注、GPT judge、基准问答评估和 caption 与 ground truth object 的匹配分析...

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
```

---

## Task 6: report_generation (LLM)

### 示例 6.1：组会汇报讲稿（LLM-assisted Report Writer）

> **说明：** 此示例展示启用 LLM Report Writer（`ENABLE_LLM_REPORT_WRITER=true`，已配置 `OPENAI_API_KEY`）后的输出。

**Input:**
```
请总结 Guardrail-Agnostic 这篇论文，生成组会汇报内容
```

**task_type:** `report_generation`

**classifier_source:** `llm`

**route_reason:** `用户要求生成组会汇报内容，总结 Guardrail-Agnostic 论文，属于 report_generation 任务。`

**tool_used:** `none`

**evidence_status:** `passed`

**evidence_reason:** `回答包含 RAG 检索来源，可追踪到本地资料库。`

**Sources:**
1. [paper_note] data/papers/guardrail_agnostic.md (title=Guardrail-Agnostic Societal Bias Evaluation in Large Vision-Language Models)
2. [paper_note] data/papers/multimodal_bias_survey.md (title=Multimodal Bias and Stereotype Evaluation Survey)

**Output:**

```
这是汇报生成任务。系统已根据本地科研资料库生成结构化汇报草稿。

生成方式：
LLM-assisted Report Writer (group_meeting)

# Guardrail-Agnostic 社会偏见评估 —— 组会汇报

## 1. 研究背景

大型视觉语言模型（LVLMs）在图像描述、视觉问答和决策任务中表现出色，但同时也可能表现出社会偏见。传统的偏见评估方法通常通过直接提问来检测模型是否存在偏见，然而现代 LVLMs 普遍配备了安全护栏（guardrails），当检测到敏感问题时可能会拒答，导致传统评估方法无法稳定暴露模型的内部偏见。

## 2. 论文目标

本文提出了一种 guardrail-agnostic 的评估方法，核心目标是：在不触发模型安全拒答的前提条件下，通过更间接、更任务化的方式评估模型的隐式社会偏见。

## 3. 核心方法

该方法的关键设计思路包括：

- **避免直接询问敏感属性**：不直接问"此人的性别/种族是什么"，而是通过感知、事实性、决策等任务观察输出中的系统性偏差。
- **任务化评估**：将偏见测量嵌入到 perception（感知）、factuality（事实性）、decision-making（决策）等任务中，关注模型在正常回答过程中的隐式偏差。
- **多维度观察**：不只关注拒答率，而是关注不同群体条件下的回答差异、任务准确率差异、偏向性选择比例和输出倾向性。

## 4. 与项目的关系

该工作适合作为 ResearchAgent 后续多模态偏见评估模块的 related work。它支持从显式偏见测试走向更鲁棒的隐式偏见审计的研究方向。

## 5. 后续可补充内容

- 加入论文具体实验设计和定量结果（当前资料库中暂无详细实验数据）。
- 补充与其他 bias evaluation 方法（如 stereotype library 方法）的对比。
- 讨论该方法是否适用于本项目的 SD3.5 gender swap 受控实验场景。
```

---

### 示例 6.2：PPT 页面文案（LLM-assisted Report Writer, ppt_slide 风格）

> **说明：** 此示例展示 `report_style=ppt_slide` 时的 LLM 输出格式。

**Input:**
```
请给我一份 OpenImages-MIAP 数据集相关的 PPT 汇报草稿
```

**task_type:** `report_generation`

**classifier_source:** `rule`

**route_reason:** `命中了汇报、PPT、组会、文案、草稿、报告或 presentation 相关关键词。`

**tool_used:** `none`

**evidence_status:** `passed`

**evidence_reason:** `回答包含 RAG 检索来源，可追踪到本地资料库。`

**Sources:**
1. [dataset_doc] data/datasets/openimages_miap.md (dataset=OpenImages-MIAP)
2. [dataset_doc] data/datasets/coco.md (dataset=COCO2017)
3. [dataset_doc] data/datasets/fairface.md (dataset=FairFace)

**Output:**

```
这是汇报生成任务。系统已根据本地科研资料库生成结构化汇报草稿。

生成方式：
LLM-assisted Report Writer (ppt_slide)

# OpenImages-MIAP 数据集介绍 —— PPT 页面文案

## 第 1 页 / 标题页
- 标题：OpenImages-MIAP 数据集概述
- 副标题：人物属性标注与偏见评估数据源
- 汇报人：[姓名]
- 日期：[日期]

## 第 2 页 / 数据集基本信息
- OpenImages-MIAP 基于 OpenImages 构建
- 提供人物 bounding box 级别的感知性别表达和年龄表达标注
- 标注粒度：person-level / bbox-level（非图像级标签）
- 适用于：多模态模型偏见评估、人物属性分析、群体统计

## 第 3 页 / 关键特点
- 人物框级别标注，非简单图像级标签
- 同一图像中可有多个人物框，各自独立标注属性
- 避免了将 bbox 属性粗暴聚合为图像级标签的常见错误
- 对多人物图像的分析更细粒度、更精确

## 第 4 页 / 适用研究任务
- 多人物场景中的人物属性分析
- 视觉语言模型在人群相关任务中的偏见评估
- Stereotype library 构建：人物-对象-场景共现关系挖掘
- 模型公平性分析：不同群体条件下的性能差异

## 第 5 页 / 局限与注意事项
- 属性不应直接用于判定个体真实身份或敏感属性
- 不能将 bbox 级属性粗暴聚合为图像级标签
- 在图像级偏见评估中需特别注意多人物并存的场景
- 当前资料库中暂无详细的定量分布统计

## 第 6 页 / 与本项目的关联
- 可作为 ResearchAgent 偏见评估模块的数据源之一
- 与 FairFace（人脸属性）、COCO（对象标注）互补使用
- 支持回答 Agent 关于"数据集标注粒度"的查询问题
```

---

## Task 7: code_help

代码辅助任务。当前版本为占位实现，暂不接入 RAG 或工具。

### 示例 7.1：环境/依赖报错

**Input:**
```
ModuleNotFoundError: No module named langgraph 怎么解决
```

**task_type:** `code_help`

**classifier_source:** `rule`

**route_reason:** `命中了代码、脚本、bug、ModuleNotFoundError、报错或环境相关关键词。`

**tool_used:** `none`

**evidence_status:** `weak`

**evidence_reason:** `当前回答没有 RAG Sources 或工具结果支撑，属于弱证据回答。`

**evidence_warnings:**
- 未检索到资料来源。
- 未调用成功的本地分析工具。

**Sources:** 无

**Output:**

```
这是代码辅助任务，后续会接入代码解释与修改工具。
```

> **说明：** `code_help` 任务在当前 v0.5 Memory + Multi-Agent Preview 版本中为占位节点。该任务的 `evidence_status` 固定为 `weak`，因为 code_node 不会调用 RAG 检索，也不会调用工具。后续版本计划接入代码解释与环境诊断能力。

---

## Task 8: general

通用科研助手任务。当前版本为占位实现，不接入 RAG 或工具。

### 示例 8.1：开放式科研建议

**Input:**
```
我今天应该怎么安排科研任务
```

**task_type:** `general`

**classifier_source:** `rule`

**route_reason:** `未命中明确任务关键词，归为通用科研助手任务。`

**tool_used:** `none`

**evidence_status:** `weak`

**evidence_reason:** `当前回答没有 RAG Sources 或工具结果支撑，属于弱证据回答。`

**evidence_warnings:**
- 未检索到资料来源。
- 未调用成功的本地分析工具。

**Sources:** 无

**Output:**

```
这是通用科研助手任务。
```

> **说明：** `general` 任务在当前 v0.5 Memory + Multi-Agent Preview 版本中为占位节点。与 `code_help` 类似，它的 `evidence_status` 为 `weak`。后续版本计划接入通用科研知识库或 LLM-based 自由回答能力。

---

## 补充说明

### Evidence Checker 状态含义

| evidence_status | 含义 |
|---|---|
| `passed` | 回答有 RAG Sources 或成功的工具结果支撑 |
| `weak` | 回答缺少 RAG Sources 和工具结果，属于弱证据回答 |

### 任务类型与节点对照

| task_type | 对应节点 | RAG 资料库 | 是否调用工具 |
|---|---|---|---|
| `paper_question` | `paper_node` | paper_note | 否 |
| `experiment_analysis` | `experiment_node` | experiment_doc | 是（CSV/JSONL） |
| `dataset_recommendation` | `dataset_node` | dataset_doc | 否 |
| `report_generation` | `report_node` | 全库（general） | 否 |
| `code_help` | `code_node` | 无 | 否 |
| `general` | `general_node` | 无 | 否 |

### Classifier Source 含义

| classifier_source | 含义 |
|---|---|
| `rule` | 使用规则分类器分类 |
| `llm` | 使用 LLM 分类器分类 |
| `rule_fallback` | LLM 分类失败，回退到规则分类 |

### 运行方式

**CLI：**
```bash
python run_cli.py
```

**Web UI（Streamlit）：**
```bash
streamlit run app.py
```

**测试脚本：**
```bash
python scripts/test_agent_demo.py           # 完整 Agent 流程测试
python scripts/test_report_writer.py         # Report Writer 测试
python scripts/test_llm_report_writer.py     # LLM Report Writer 测试
```
