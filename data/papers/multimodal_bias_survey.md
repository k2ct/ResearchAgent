# Multimodal Bias and Stereotype Evaluation Survey

## 基本信息

- source_type: paper_note
- title: Multimodal Bias and Stereotype Evaluation Survey
- topic: multimodal_bias
- year: 2025
- path: data/papers/multimodal_bias_survey.md

## 研究问题

多模态模型可能在图像理解、图像描述、视觉问答和决策类任务中表现出社会偏见。偏见可能与性别、年龄、种族、职业、宗教、场景和对象共现关系有关。

多模态偏见评估的核心问题是：模型是否会基于视觉线索之外的社会先验进行错误推断。

## 方法概括

常见方法包括构造对照图像、替换人物属性、控制场景和对象、设计问答模板、比较不同群体条件下的输出差异，以及分析 embedding space 中的偏见方向。

另一类方法关注 stereotype library，即从数据集中挖掘人物、场景、对象、动作之间的刻板共现关系，并观察模型是否会放大这些关系。

## 使用数据集 / 评估任务

常见数据来源包括 COCO、FairFace、OpenImages-MIAP、GQA、Visual Genome、Flickr30K Entities 和自建生成图像数据集。

评估任务包括 captioning、VQA、attribute prediction、role recognition、relationship reasoning 和 decision-making。

## 主要指标

常见指标包括群体间准确率差异、预测分布差异、敏感属性误判率、stereotype score、bias amplification、embedding projection score 和 hallucination correlation。

## 与 ResearchAgent 项目的关系

该文档为 ResearchAgent 的偏见评估方向提供背景知识。它可以帮助后续 Agent 根据用户问题检索多模态偏见、刻板印象、场景对象共现和 hallucination-bias 关联等资料。