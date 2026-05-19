# GQA Dataset

## 基本信息

- source_type: dataset_doc
- dataset: GQA
- annotation_type: scene_graph_question_answer
- task: relationship_reasoning
- path: data/datasets/gqa.md

## 数据集内容

GQA 是基于真实图像和 scene graph 构建的视觉问答数据集，包含对象、属性、关系以及结构化问答信息。

它强调组合推理和视觉关系理解，比单纯对象检测数据集更适合分析场景图和关系推理。

## 适合的研究任务

GQA 适合视觉问答、场景图分析、对象关系推理、属性理解和多跳视觉推理。

在 stereotype library 构建中，GQA 的 scene graph 信息可以帮助分析人物、对象、动作和场景之间的共现关系。

## 优点

GQA 提供结构化 scene graph 和问题答案，有助于研究对象关系、属性组合和复杂视觉推理。

它可以补充 COCO 只关注对象类别的不足。

## 局限

GQA 的问答和 scene graph 结构较复杂，使用前需要进行字段解析和清洗。

如果只是做最小 RAG demo，不建议直接加载完整 GQA 原始数据，而是先整理成小型说明文档。

## 在 ResearchAgent 项目中的用途

GQA 可用于支持关系推理、场景图分析和 stereotype library 构建相关问题。它可以帮助 Agent 回答“GQA 对场景图和关系分析有什么价值”这类问题。