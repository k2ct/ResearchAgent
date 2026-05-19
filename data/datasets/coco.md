# COCO Dataset

## 基本信息

- source_type: dataset_doc
- dataset: COCO2017
- annotation_type: object_bbox_caption
- task: hallucination_evaluation
- path: data/datasets/coco.md

## 数据集内容

COCO 是常用的图像理解数据集，包含图像、对象类别、bounding box、segmentation 和 caption 标注。

在幻觉评估中，COCO 常用于对象级幻觉检测，因为它提供较清晰的对象类别标注，可以与模型生成描述中的对象词进行比较。

## 适合的研究任务

COCO 适合图像描述、对象识别、视觉问答预处理、对象级幻觉评估和 hallucination-prone image screening。

在 ResearchAgent 项目中，COCO 可用于构建候选图像池，并基于对象标注计算 extra object rate、precision、recall 和 CHAIR-like 指标。

## 优点

COCO 标注质量较高，类别体系清晰，相关工具成熟，适合作为幻觉评估的基础数据源。

它也适合做小规模 demo，因为样本可以按数量、场景或对象类别进行筛选。

## 局限

COCO 的对象类别有限，不能覆盖开放世界中的所有对象。对于 caption 中出现但 COCO 未标注的合理对象，可能会产生误判。

因此在严格幻觉评估中，需要区分 core ground truth 和 extended ground truth。

## 在 ResearchAgent 项目中的用途

COCO 是当前项目中幻觉筛选实验的重要数据源。它支持 coco_val_n100_g1、coco_val_n300_g1 等实验文档的解释和检索。