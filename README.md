# ResearchAgent v0.1

ResearchAgent 是一个基于 LangGraph 构建的科研智能体项目。

当前版本是 v0.1，主要目标是搭建一个清晰、可扩展、可运行的 Agent 工作流骨架。它支持用户问题输入、任务分类、条件路由，并根据不同科研任务进入对应的处理节点。

本版本暂时不接入真实 RAG、论文数据库、实验文件分析或报告生成工具，而是先完成 Agent 的核心流程设计，为后续扩展打基础。

---

## 项目目标

ResearchAgent 的长期目标是构建一个面向科研场景的多功能智能体，支持：

- 论文问答与文献总结
- 实验结果分析
- 数据集推荐
- 组会汇报生成
- 代码与环境问题辅助
- 证据检查与回答可信度评估

当前 v0.1 版本聚焦于：

> 使用 LangGraph 跑通一个最小科研 Agent 工作流。

---

## 当前功能

ResearchAgent v0.1 已实现以下功能：

- 基于 LangGraph 的 Agent 工作流
- 使用 `AgentState` 维护流程状态
- 支持规则分类器
- 支持可选的 LLM 分类器
- LLM 分类失败时自动回退到规则分类
- 根据任务类型进行条件路由
- 支持命令行交互 Demo
- 支持 6 类科研任务节点

---

## 支持的任务类型

| 任务类型 | 说明 |
|---|---|
| `paper_question` | 论文问答、文献总结、论文解释、论文对比 |
| `experiment_analysis` | 实验结果分析、CSV/JSONL 输出分析、幻觉指标解释 |
| `dataset_recommendation` | 数据集推荐、数据集选择、数据集构建建议 |
| `report_generation` | 组会汇报、PPT 文案、阶段总结、实验汇报文本 |
| `code_help` | 代码解释、报错排查、脚本修改、环境问题 |
| `general` | 其他通用科研助手问题 |

---

## 工作流结构

当前 Agent 工作流如下：

```text
用户输入
↓
classify_task
↓
route_task
├── paper_node
├── experiment_node
├── dataset_node
├── report_node
├── code_node
└── general_node
↓
final_answer
↓
END
````

其中：

* `classify_task`：判断用户问题属于哪种任务类型
* `route_task`：根据任务类型选择下一个节点
* `paper_node`：论文问答任务节点
* `experiment_node`：实验分析任务节点
* `dataset_node`：数据集推荐任务节点
* `report_node`：汇报生成任务节点
* `code_node`：代码辅助任务节点
* `general_node`：通用科研助手任务节点
* `final_answer`：汇总最终回答

---

## 项目结构

```text
ResearchAgent
├── run_cli.py
├── test_demo.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── notebooks
│   └── day1_langgraph_minimal.ipynb
└── src
    └── research_agent
        ├── __init__.py
        └── graph
            ├── __init__.py
            ├── state.py
            ├── nodes.py
            ├── router.py
            ├── workflow.py
            └── llm_classifier.py
```

---

## 核心文件说明

| 文件                  | 作用                |
| ------------------- | ----------------- |
| `run_cli.py`        | 命令行运行入口           |
| `test_demo.py`      | 一键测试多个示例输入        |
| `requirements.txt`  | 项目依赖              |
| `.env.example`      | 环境变量示例文件          |
| `state.py`          | 定义 AgentState     |
| `nodes.py`          | 定义各个 LangGraph 节点 |
| `router.py`         | 定义任务路由逻辑          |
| `workflow.py`       | 组装 LangGraph 工作流  |
| `llm_classifier.py` | LLM 分类器逻辑         |

---

## 环境安装

推荐使用 Python 3.11。

### 1. 创建本地 Conda 环境

在项目根目录执行：

```powershell
cd F:\ResearchAgent
conda create --prefix .\.conda python=3.11 -y
```

### 2. 激活环境

```powershell
conda activate .\.conda
```

如果 PowerShell 中激活失败，也可以直接使用本地环境里的 Python：

```powershell
.\.conda\python.exe
```

### 3. 安装依赖

```powershell
.\.conda\python.exe -m pip install -r requirements.txt
```

---

## 环境变量配置

项目使用 `.env` 文件管理环境变量。

请复制 `.env.example`：

```powershell
Copy-Item .env.example .env
```

`.env.example` 示例：

```env
# 是否启用 LLM 分类器
# true：优先使用 LLM 分类，失败时回退规则分类
# false：只使用规则分类
USE_LLM_CLASSIFIER=false

# OpenAI-compatible API 配置
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

如果暂时不使用 LLM 分类器，可以保持：

```env
USE_LLM_CLASSIFIER=false
```

这样系统会只使用规则分类器，仍然可以正常运行。

如果希望启用 LLM 分类器，需要配置：

```env
USE_LLM_CLASSIFIER=true
OPENAI_API_KEY=你的 API Key
OPENAI_BASE_URL=你的 API Base URL
OPENAI_MODEL=模型名称
```

---

## 运行方式

### 方式一：运行 CLI

```powershell
cd F:\ResearchAgent
.\.conda\python.exe run_cli.py
```

运行后会进入命令行交互模式：

```text
请输入你的科研问题：
>
```

输入：

```text
请帮我分析 coco_val_n300_g1 的幻觉风险
```

示例输出：

```text
任务类型：experiment_analysis
分类来源：rule
分类原因：命中了实验、COCO、幻觉或 benchmark 相关关键词。
处理结果：这是实验分析任务，后续会接入 CSV / JSONL 分析工具。
```

---

## 示例输入

你可以测试以下输入：

```text
请帮我总结一篇多模态偏见论文
```

```text
请帮我分析 coco_val_n300_g1 的幻觉风险
```

```text
请推荐适合做幻觉评估的数据集
```

```text
帮我生成组会汇报文本
```

```text
ModuleNotFoundError: No module named langgraph 怎么解决
```

```text
我今天应该怎么安排科研任务
```

---

## 一键测试

如果项目中包含 `test_demo.py`，可以运行：

```powershell
.\.conda\python.exe test_demo.py
```

该脚本会自动测试多个示例输入，检查不同任务类型是否能够正确分类和路由。

---

## 当前版本说明

当前版本是 `v0.1`，主要完成了科研 Agent 的基础骨架。

已经完成：

* LangGraph 最小工作流
* State / Node / Edge / Conditional Edge 基础结构
* 规则分类
* 可选 LLM 分类
* LLM 失败自动回退规则分类
* 多任务路由
* CLI 运行入口
* 项目工程化结构

尚未完成：

* 真实论文 RAG
* 本地论文库检索
* 实验 CSV / JSONL 自动分析
* 报告生成器
* Evidence Checker
* Streamlit / Web 前端
* 多 Agent 协作

---

## 后续计划

后续版本计划逐步加入以下能力：

### v0.2：Paper RAG 最小版本

* 支持读取本地 Markdown / TXT 文献笔记
* 支持简单文本检索
* 支持基于文献内容回答问题

### v0.3：实验分析节点

* 支持读取 CSV / JSONL 实验结果
* 支持输出指标摘要
* 支持解释幻觉相关指标

### v0.4：报告生成节点

* 根据论文、实验结果和阶段进展生成组会汇报文本
* 支持生成 PPT 大纲

### v0.5：证据检查节点

* 检查回答是否有证据支撑
* 标注回答中的不确定部分
* 提高科研回答可信度

---

## 技术栈

当前版本使用：

* Python 3.11
* LangGraph
* LangChain
* langchain-openai
* python-dotenv
* Pydantic
* Rich
* tqdm

---

## 设计思想

本项目采用 LangGraph 而不是简单线性 Chain，主要原因是科研 Agent 后续会涉及：

* 多任务路由
* 多节点协作
* 条件分支
* 失败回退
* 证据检查
* 循环重试
* 多 Agent 扩展

因此，当前版本使用 `AgentState` 作为共享状态，让不同节点围绕同一个状态对象进行读取和更新。

简单来说：

```text
LangGraph 负责流程编排
LangChain / LLM 负责节点内部能力
AgentState 负责保存上下文
```

---

## 安全说明

请不要上传真实 `.env` 文件。

`.gitignore` 中应包含：

```gitignore
.env
.conda/
__pycache__/
.ipynb_checkpoints/
```

其中：

* `.env` 可能包含 API Key
* `.conda/` 是本地 Python 环境，体积大且不适合上传
* `__pycache__/` 是 Python 缓存文件

GitHub 仓库中只应保留 `.env.example`，不要上传真实 `.env`。

---

## 版本

当前版本：

```text
ResearchAgent v0.1
```

这是一个最小可运行科研 Agent Demo，主要用于展示 LangGraph 工作流、任务分类、条件路由和项目工程化结构。

```

---

如果你现在仓库里其实只有 `app.py`，而不是 `run_cli.py + src/` 这套结构，那 README 里有些项目结构需要对应改一下。按照你前面 Day 4 的版本，这份就是比较合适的中文 README。
```

## Streamlit Web Demo

Run:

```powershell
.\.conda\python.exe -m streamlit run app.py

Example queries:

请分析 data/experiments/sample_metrics.csv
请分析 data/experiments/sample_generations.jsonl
OpenImages-MIAP 的性别标注是图像级还是 bbox 级？
