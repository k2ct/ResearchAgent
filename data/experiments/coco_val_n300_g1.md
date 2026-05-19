# COCO Validation Hallucination Screening n300 g1

## 基本信息

- source_type: experiment_doc
- run_tag: coco_val_n300_g1
- dataset: COCO2017
- task: hallucination_screening
- path: data/experiments/coco_val_n300_g1.md

## 实验目标

coco_val_n300_g1 是基于 COCO validation 样本的幻觉风险筛选实验。实验目标是通过模型生成描述与图像对象标注之间的差异，识别 hallucination-prone images。

该实验比 n100 规模更大，更适合观察 extra objects 的分布、top-risk 图像类型和人工标注候选样本。

## 数据与设置

实验使用 COCO2017 validation 图像。n300 表示抽取 300 张图像，g1 表示每张图像生成 1 次描述。

该实验通常用于构建 top20 高风险样本和 control20 对照样本，供后续人工标注和 stereotype library 构建使用。

## 主要脚本 / 输出文件

典型输出包括：

- outputs/coco_val_n300_g1_generations.jsonl
- outputs/coco_val_n300_g1_eval.jsonl
- outputs/coco_val_n300_g1_ranked.csv
- outputs/coco_val_n300_g1_annotation_sheet.csv
- outputs/coco_val_n300_g1_top20_generations.jsonl
- outputs/coco_val_n300_g1_top20_eval.jsonl

## 关键指标

核心指标包括 mean_extra_object_rate、mean_precision、mean_recall、repeat_rate、prompt_consistency、hrs_v1 和 union_extra_objects。

其中 union_extra_objects 用于记录一个图像在多次或多 prompt 生成中出现过的额外对象集合。hrs_v1 用于综合排序幻觉风险。

## 后续分析价值

该实验可以为后续人工筛选 hallucination-prone images 提供候选样本，也可以作为 stereotype library 构建的输入来源。

在 ResearchAgent 阶段二中，该文档用于测试 experiment_analysis 类型问题的 RAG 检索。