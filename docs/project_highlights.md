# ResearchAgent 项目亮点

> **版本：ResearchAgent v0.5 Memory + Multi-Agent Preview**

## 1. 项目定位

ResearchAgent 是一个面向科研场景的本地智能助手，目标是辅助研究者完成资料检索、实验文件分析、论文阅读、长期记忆管理和多 Agent 协作研究。

项目基于 LangGraph 构建，采用显式状态流转和条件路由，将用户问题分发到不同任务节点，并结合 RAG、工具调用、Memory 检索、Multi-Agent 协调和证据检查生成回答。

## 2. 技术栈

- **Python 3.11**
- **LangGraph** — Agent 工作流编排
- **LangChain** — RAG 组件
- **Chroma** — 本地向量数据库
- **HuggingFace Embeddings** — `sentence-transformers` 嵌入
- **Streamlit** — Web UI
- **pandas** — 实验数据分析
- **pymupdf / python-docx / python-pptx** — 文档解析
- **MinerU API**（可选）— 复杂 PDF 解析

## 3. 核心能力

### 3.1 LangGraph 多节点工作流

项目使用 LangGraph 管理 Agent 工作流，包括：

- `classify_task`：任务分类（paper / experiment / dataset / report / code / general）
- `route_task`：条件路由，根据 task_type 分发到对应节点
- `retrieve_memory`：Memory 检索节点，注入历史上下文
- `multi_agent_router`：可选 Multi-Agent 分发节点
- `evidence_check`：证据质量检查
- `final_answer`：最终回答生成

### 3.2 Agentic RAG + Hybrid Retrieval

项目不是简单地对所有资料统一检索，而是根据任务类型选择不同检索范围：

- `paper_question` → `paper_note`
- `experiment_analysis` → `experiment_doc`
- `dataset_recommendation` → `dataset_doc`

支持两种检索模式：
- **vector**：纯 Chroma 向量相似度搜索
- **hybrid**：向量检索 + 关键词检索 + 分数融合 + 启发式重排序

### 3.3 Markdown-aware Chunking

基于 Markdown 标题层级进行语义分块，保留文档结构信息。支持可配置的 chunk 大小、最小合并阈值和段落重叠参数。

### 3.4 Tool Calling

实验分析节点支持调用本地工具：

- **CSV Analyzer**：分析实验指标表（行数、列名、数值列摘要、低基数字段分布）
- **JSONL Analyzer**：分析逐样本生成/评测日志（记录数、字段类型、布尔分布、列表统计）

工具返回结构化结果，注入 AgentState，供后续回答和证据检查使用。

### 3.5 Evidence Checker

最小证据检查节点，根据以下信息判断回答可信度：

- 是否存在 RAG Sources
- 是否存在成功工具调用结果
- 是否存在证据警告

该模块让 Agent 不仅能回答，还能透明说明回答是否有依据。

### 3.6 Report Writer

支持三种汇报风格的草稿生成：

- **组会汇报**（group_meeting）：结构化研究讲稿
- **PPT 大纲**（ppt_slide）：按页组织的 PPT 文案
- **论文摘要**（summary）：论文核心内容总结

支持双模式：
- **模板版（默认）**：使用规则模板生成结构化草稿，无需 LLM
- **LLM 版（可选）**：LLM 生成自然语言汇报，自动检测汇报风格，严格基于 RAG Sources

### 3.7 Memory System

完整的七层记忆子系统：

| 层 | 说明 |
|---|---|
| `schema` | MemoryRecord 数据结构定义与校验 |
| `store` | JSONL + Markdown summary 文件存储，支持增删改查 |
| `writer` | 规则驱动的记忆写入决策引擎，含去重和合并 |
| `retriever` | 基于关键词 + 时间衰减的多维度记忆检索 |
| `privacy_scope` | 三层作用域控制（private / shared / global） |
| `consolidation` | 记忆压缩 / 合并 / 过期 / 阶段摘要，dry-run 安全操作 |
| `adapters` | Claim / Paper / Progress 模块与 Memory 的桥接层 |

支持三层记忆时间线：`long_term`（跨会话持久化）、`mid_term`（会话内中间记忆）、`short_term`（当前查询上下文）。

### 3.8 Memory-aware Agent

主 Agent 可配置记忆感知层：在推理前自动检索相关历史记忆，将上下文注入最终回答。启用方式：`ENABLE_MEMORY_AWARE_AGENT=true`。

### 3.9 Multi-Agent System

9 个 Specialist Agent 的协作架构：

| Agent | 职责 |
|---|---|
| `coordinator_agent` | 任务分析、Agent 选择、结果汇总 |
| `paper_agent` | 论文检索、阅读、总结 |
| `experiment_agent` | 实验数据分析（CSV / JSONL / 指标） |
| `claim_agent` | 声明/论断支撑检索 |
| `progress_agent` | 组会进展追踪、PPT 大纲 |
| `report_agent` | 科研汇报生成 |
| `code_agent` | 代码/环境问题辅助 |
| `memory_agent` | 记忆存取管理 |
| `general_agent` | 通用科研问答 |

核心协作机制：
- **Orchestrator**：Agent 注册、任务分析、分发协调
- **Handoff**：Agent 间任务交接、结果传递、来源追踪
- **Tracing & Evaluation**：多 Agent 调用链追踪、质量评估
- **Arbitration**：多 Agent 分歧仲裁

### 3.10 Shared Workspace

本地 Markdown / JSONL 协作模型，支持 section 级权限：

- 所有 Agent 可读全文
- 每个 Agent 只能写入自己的 section
- 越权写入转为 suggestion，等待 coordinator 审批
- 数据存储在 `data/workspaces/`（不提交 Git）

### 3.11 Document Ingestion + MinerU Fallback

文档摄入管道支持 PDF / DOCX / PPTX / MD → YAML-front-matter Markdown：

- **本地模式（默认）**：pymupdf / python-docx / python-pptx，无需 API Key
- **MinerU 模式（可选）**：复杂 PDF 解析（公式/表格/OCR），失败自动 fallback 本地
- 摄入结果记录 `ingestion_backend`、`mineru_used`、`mineru_error` 字段，可审计

### 3.12 LLM Enhancement Layer

统一的 LLM 增强层，覆盖 Claim Support / Paper Reading / PPT Progress Memory / Report Polish：

- 所有模块**总是先生成规则版输出**
- LLM 增强严格约束：只能使用提供的证据/内容，不得虚构，必须标注缺失信息
- LLM 不可用时**自动回退规则版**，不报错

### 3.13 Streamlit Web UI

完整 Debug 可视化界面：

- Task Classification — 任务类型识别
- Tool Analysis — 工具调用及结果
- RAG Sources — 检索来源文档
- Memory Metrics — 记忆检索记录数/上下文
- Multi-Agent Handoff — Agent 交接详情
- Arbitration — 分歧仲裁结果
- Debug State — 完整状态快照

## 4. 工程亮点

- 使用 `src/` 结构组织项目，模块职责清晰
- 使用 `.env.example` 管理环境变量模板，含完整中文注释
- 使用 `.gitignore` 排除敏感配置、本地环境、向量库、记忆数据、追踪日志等
- 使用 `scripts/` 提供构建索引、测试、文档摄入、Memory 合并等脚本
- 提供 sample documents、CSV 和 JSONL 数据用于快速演示
- 提供 `.streamlit/config.toml` 预设端口和 watcher 配置，Windows 稳定运行
- 所有 LLM / API 功能默认关闭，规则版可独立运行，失败自动 fallback

## 5. 当前限制

- 当前仍是 Demo/Preview，不是完整生产系统
- Shared Workspace 是本地文件协作模型，不是数据库级并发控制
- Multi-Agent execution v1 以规则分发为主，尚未接入 LLM-based 动态编排
- Reranker 是 heuristic 实现，后续可引入 CrossEncoder 模型
- HuggingFace embedding 首次运行需联网下载或提前缓存
- MinerU 需要单独申请 API Key
- Web UI 是 demo 级 Streamlit

## 6. 后续计划

- CrossEncoder Reranker 替换 heuristic reranker
- Chroma Memory Collection 向量化检索
- Workspace 接入 Orchestrator 深度集成
- Paper Figure / Table Understanding 增强
- Evaluation Dashboard 可视化面板
- Docker Packaging 容器化部署
