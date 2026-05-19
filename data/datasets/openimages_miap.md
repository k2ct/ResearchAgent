# OpenImages-MIAP Dataset

## 基本信息

- source_type: dataset_doc
- dataset: OpenImages-MIAP
- annotation_type: bbox_level_sensitive_attribute
- task: bias_evaluation
- path: data/datasets/openimages_miap.md

## 数据集内容

OpenImages-MIAP 是基于 OpenImages 的人物属性相关标注数据，包含人物 bounding box 以及与人物相关的属性标注，例如感知性别表达和年龄表达等。

该数据集的关键特点是属性更接近 person-level 或 bbox-level，而不是简单的 image-level 标签。

## 适合的研究任务

OpenImages-MIAP 适合用于人物属性分析、多人物图像中的群体属性统计、偏见评估和视觉语言模型在人群相关任务中的表现分析。

它尤其适合研究模型是否会在人物属性判断中产生系统性偏差。

## 优点

该数据集提供人物框级别信息，因此比单纯图像级标签更细。对于一张图中有多个人物的情况，可以分别分析不同人物框的属性。

这对于多模态偏见研究很重要，因为同一张图像中可能同时出现不同性别表达或年龄表达的人物。

## 局限

OpenImages-MIAP 的属性不应被粗暴地当作整张图像的唯一属性标签。如果将 bbox-level 属性直接聚合为 image-level 标签，可能会误解多人物图像的真实情况。

在做图像级偏见评估时，需要特别注意一张图中多个人物属性并存的问题。

## 在 ResearchAgent 项目中的用途

该数据集用于支持偏见评估、人物属性分析和 stereotype library 构建。它也可以帮助 ResearchAgent 回答“OpenImages-MIAP 的性别标注是图像级还是 bbox 级”这类问题。