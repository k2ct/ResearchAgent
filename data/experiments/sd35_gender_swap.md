# SD3.5 Gender Swap Controlled Image Generation

## 基本信息

- source_type: experiment_doc
- run_tag: sd35_gender_swap
- dataset: synthetic_sd35
- task: bias_hallucination_controlled_generation
- path: data/experiments/sd35_gender_swap.md

## 实验目标

该实验使用 SD3.5 生成受控图像，通过共享 seed 的方式尽量保持场景、动作、构图和对象一致，同时切换人物性别条件。

实验目标是构造可控的 gender swap 图像对，用于分析视觉语言模型在相似场景下是否会因为人物性别变化而产生不同描述、不同对象幻觉或不同角色判断。

## 数据与设置

实验使用本地 SD3.5 生成模型，通常设置为 1024x1024 图像、固定 steps 和 guidance，并对同一组 prompt 使用共享 seed。

图像输出会按照性别、场景和对象条件进行组织。部分实验还会加入男性或女性刻板关联对象，用于观察对象共现是否影响模型判断。

## 主要脚本 / 输出文件

典型脚本包括 SD3.5 批量生成脚本、gender swap prompt 配置文件、图像后处理脚本和后续偏见评估脚本。

典型输出路径包括：

- outputs/HalluciationTest_Images
- outputs/HalluciationTest_Images_objects_mf_singlelib_aggressive
- person_blacked 图像变体

## 关键指标

后续评估可能关注模型对人物性别的判断准确率、不同场景下的偏见倾向、对象幻觉率、gender-object-scene 交互效应，以及 person-blacked 条件下模型是否仍然依赖背景或对象进行性别推断。

## 后续分析价值

该实验用于构建自建受控数据集，可以帮助分析多模态模型是否会受到场景、对象和社会刻板印象的影响。

在 ResearchAgent 中，该资料可用于回答 SD3.5 gender swap 实验设计、受控变量和后续偏见分析用途等问题。