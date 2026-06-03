# ResearchAgent 项目亮点

## 1. 项目定位

ResearchAgent 是一个面向科研场景的智能体 Demo，目标是辅助研究者完成资料检索、实验文件分析、证据追踪和结果解释。

项目基于 LangGraph 构建，采用显式状态流转和条件路由，将用户问题分发到不同任务节点，并结合 RAG、工具调用和证据检查生成回答。

## 2. 技术栈

- Python
- LangGraph
- LangChain
- Chroma
- HuggingFace Embeddings
- Streamlit
- pandas
- JSONL / CSV 文件分析

## 3. 核心能力

### 3.1 LangGraph Workflow

项目使用 LangGraph 管理 Agent 工作流，包括：

- classify_task：任务分类
- route_task：条件路由
- paper / experiment / dataset / code / report / general 节点
- evidence_check：证据检查
- final_answer：最终回答生成

### 3.2 Agentic RAG

项目不是简单地对所有资料统一检索，而是根据任务类型选择不同检索范围：

- paper_question → paper_note
- experiment_analysis → experiment_doc
- dataset_recommendation → dataset_doc

这种方式让 RAG 检索更贴合任务语义。

### 3.3 Tool Calling

实验分析节点支持调用本地工具：

- CSV Analyzer：分析实验指标表
- JSONL Analyzer：分析逐样本生成 / 评测日志

工具返回结构化结果，并写入 AgentState，供后续回答和证据检查使用。

### 3.4 Evidence Checker

项目加入最小证据检查节点，根据以下信息判断回答可靠性：

- 是否存在 RAG Sources
- 是否存在成功工具调用结果
- 是否存在证据警告

该模块让 Agent 不仅能回答，还能说明回答是否有依据。

### 3.5 Streamlit Web UI

项目提供最小 Web Demo，支持：

- 输入科研问题
- 展示任务类型
- 展示工具调用结果
- 展示证据状态
- 展示 Sources
- 展示 Debug State

## 4. 工程亮点

- 使用 src 结构组织项目
- 使用 .env.example 管理环境变量示例
- 使用 .gitignore 排除本地环境、向量库和缓存
- 使用 scripts/ 提供构建索引和测试脚本
- 提供 sample documents、CSV 和 JSONL 数据用于快速演示
- 提供 Streamlit 配置，关闭 watcher 提升 Windows 稳定性

## 5. 当前限制

- 当前仍是 Demo，不是完整生产系统
- RAG 使用 sample documents
- Evidence Checker 是规则版，不是深度事实核查
- CSV / JSONL 工具只做基础统计
- 尚未实现复杂 Report Writer 和多 Agent 协作

## 6. 后续计划

- 增加 Report Writer 节点
- 增加更细粒度的实验指标分析
- 增强 Evidence Checker
- 支持更多文件类型
- 增加多 Agent 协作
- 优化 Web UI 展示