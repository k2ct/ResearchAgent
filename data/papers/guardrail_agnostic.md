# Guardrail-Agnostic Societal Bias Evaluation in LVLMs

## 基本信息

- source_type: paper_note
- title: Guardrail-Agnostic Societal Bias Evaluation in Large Vision-Language Models
- topic: multimodal_bias
- year: 2026
- path: data/papers/guardrail_agnostic.md

## 研究问题

这篇论文关注大型视觉语言模型中的社会偏见评估问题，尤其是当模型存在安全护栏或拒答机制时，传统直接提问式偏见测试可能无法稳定暴露模型内部偏见。

论文希望通过更间接、更任务化的方式评估模型在人物无关或弱人物相关任务中的隐式社会偏见。

## 方法概括

论文的核心思想是设计 guardrail-agnostic evaluation，即尽量不直接触发模型安全拒答，而是通过感知、事实性、决策等任务观察模型输出中的系统性偏差。

这种方法强调：偏见评估不一定依赖直接询问敏感属性，而可以通过任务表现差异、选项偏好、描述倾向等方式进行测量。

## 使用数据集 / 评估任务

该类工作通常会构造多种人物相关或人物无关的视觉语言任务，包括 perception、factuality、decision-making 等任务类型。

任务重点不只是判断模型是否拒答，而是观察模型在正常回答过程中的隐式偏差。

## 主要指标

可能关注的指标包括不同群体条件下的回答差异、任务准确率差异、偏向性选择比例、拒答率、输出倾向性等。

## 与 ResearchAgent 项目的关系

这篇论文适合作为 ResearchAgent 后续多模态偏见评估模块的 related work。它可以帮助解释为什么当前项目需要从显式偏见测试走向更鲁棒的隐式偏见审计。

在阶段二中，该文档用于测试 paper_question 类型问题的 RAG 检索。