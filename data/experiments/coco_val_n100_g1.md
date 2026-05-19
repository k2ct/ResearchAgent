# COCO Validation Hallucination Screening n100 g1

## 基本信息

- source_type: experiment_doc
- run_tag: coco_val_n100_g1
- dataset: COCO2017
- task: hallucination_screening
- path: data/experiments/coco_val_n100_g1.md

## 实验目标

该实验是一个小规模 COCO validation 幻觉风险筛选实验，目标是在 100 张图像样本上测试图像描述模型是否会生成图像中不存在的对象。

该实验主要用于验证完整 pipeline 是否能跑通，包括样本 manifest 构建、caption 生成、CHAIR-like 评估和风险排序。

## 数据与设置

实验使用 COCO2017 validation 图像作为候选数据源。n100 表示样本规模为 100，g1 表示每张图像生成 1 次描述。

该实验适合作为 smoke test 或小规模调试，不适合得出稳定统计结论。

## 主要脚本 / 输出文件

典型流程包括：

- build manifest
- generate captions
- chair-like evaluation
- rank images by hallucination risk

典型输出包括：

- outputs/coco_val_n100_g1_generations.jsonl
- outputs/coco_val_n100_g1_eval.jsonl
- outputs/coco_val_n100_g1_ranked.csv

## 关键指标

实验关注 mean_extra_object_rate、mean_precision、mean_recall、repeat_rate、prompt_consistency 和 hrs_v1 等指标。

其中 hrs_v1 可作为综合幻觉风险排序分数，用于挑选高风险图像进入人工复核。

## 后续分析价值

该实验主要用于验证 pipeline 正确性。它可以帮助 ResearchAgent 回答与小规模 COCO 幻觉筛选流程、输出文件和指标含义相关的问题。