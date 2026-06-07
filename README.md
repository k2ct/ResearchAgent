# ResearchAgent v0.5 Memory + Multi-Agent Preview

ResearchAgent is a local research assistant for paper reading, experiment analysis, RAG retrieval, long-term memory, and multi-agent research workflow. 项目基于 LangGraph 构建，面向科研工作流提供可解释、可扩展的 Agent 能力，同时提供 CLI 和 Streamlit Web UI。

> **本版本（v0.5）为 Memory + Multi-Agent 预览版**，在 v0.4 的 RAG + Report Writer 基础上新增了长期记忆系统和多智能体协作框架。

---

## 快速开始 Quick Start

以下步骤假设你已经 clone 了本项目，并在 Windows 上使用 PowerShell。

### 1. 确认 Python 版本

```powershell
python --version   # 需要 Python 3.11
```

### 2. 创建 Conda 环境（推荐）

```powershell
cd F:\ResearchAgent
conda create --prefix .\.conda python=3.11 -y
```

### 3. 安装依赖

```powershell
.\.conda\python.exe -m pip install -r requirements.txt
```

### 4. 复制环境变量模板

```powershell
copy .env.example .env
```

然后用文本编辑器编辑 `.env`，可选填入你的 API Key。**如果不使用 LLM / MinerU 等高级功能，可以保持默认值不变**——项目的所有核心流程都可以在纯规则模式下运行。

> ⚠️ `.env` 已加入 `.gitignore`，**绝对不要提交到 Git**。

### 5. 构建 RAG 索引

```powershell
.\.conda\python.exe scripts\build_index.py
```

首次运行会自动下载 HuggingFace embedding model（`sentence-transformers`），需要联网。如果模型已缓存，可设 `HF_HUB_OFFLINE=1` 跳过下载。

### 6. 运行 CLI

```powershell
.\.conda\python.exe run_cli.py
```

### 7. 运行 Streamlit Web UI

```powershell
# 默认端口 8503
.\.conda\python.exe -m streamlit run app.py --server.port 8503

# Windows 稳定启动（关闭 file watcher）
.\.conda\python.exe -m streamlit run app.py --server.port 8504 --server.fileWatcherType none
```

### 8. 运行测试（可选）

```powershell
# 端到端集成测试
.\.conda\python.exe scripts\test_agent_demo.py

# Memory 系统测试
.\.conda\python.exe scripts\test_memory_store.py

# Multi-Agent 测试
.\.conda\python.exe scripts\test_agent_profiles.py
.\.conda\python.exe scripts\test_agent_handoff.py
.\.conda\python.exe scripts\test_multi_agent_orchestrator.py
.\.conda\python.exe scripts\test_agent_workspace.py
```

如果 HuggingFace 网络不稳定，先在 PowerShell 中设置 `$env:HF_HUB_OFFLINE="1"` 再运行测试。

---

## 版本演进

| 版本 | 主要变更 |
|---|---|
| v0.1 | LangGraph 最小骨架 + CLI Demo |
| v0.2 | Agentic RAG 接入 LangGraph + Retriever Routing + Metadata Schema |
| v0.3 | CSV/JSONL Tool Calling + Evidence Checker + Template Report Writer |
| v0.4 Preview | Optional LLM-assisted Report Writer + Streamlit Web UI + 完整多场景 Demo |
| **v0.5 Preview** | **Memory System + Multi-Agent Orchestrator + Shared Workspace + MinerU Backend + Paper/Claim/Progress Pipelines** |

---

## 核心功能概览

| 模块 | 功能 | 状态 |
|---|---|---|
| RAG Knowledge Base | 基于 Chroma 的本地向量检索，支持 vector / hybrid 两种模式 | ✅ |
| Document Ingestion | PDF / DOCX / PPTX / MD 文档解析入库，YAML front matter 可审计 | ✅ |
| MinerU PDF Backend | 可选 MinerU API 后端，用于复杂 PDF 公式 / 表格 / OCR 解析，失败自动 fallback local | ✅ |
| Markdown-aware Chunking | 基于标题层级的语义分块，保留文档结构信息 | ✅ |
| Heuristic Reranker | 规则版重排序，支持基于 metadata 字段的打分和过滤 | ✅ |
| LLM Enhancement Layer | 可选 LLM 增强层（Claim / Paper / Progress / Report），严格约束不得虚构，失败自动回退规则版 | ✅ |
| Claim Support Retrieval | 声明 / 论断支撑检索，模板版 academic wording + 可选 LLM 增强 | ✅ |
| Paper Reading Pipeline | 论文章节提取 + 结构化阅读笔记生成 | ✅ |
| PPT Progress Memory | 组会进展追踪，基于 heuristic topic inference + 可选 LLM 增强 | ✅ |
| Memory System | Schema / Store / Writer / Retriever / Privacy & Scope / Consolidation / Adapters 完整记忆子系统 | ✅ |
| Memory-aware Agent | 主 Agent 接入记忆检索，支持查询相关记忆上下文 | ✅ |
| Memory Write-back | 重要交互后自动写入记忆（可配置） | ✅ |
| Multi-Agent Orchestrator | 9 个 Specialist Agent 的注册、分发、协调 | ✅ |
| Agent Handoff | Agent 之间的任务交接与结果汇总 | ✅ |
| Agent Tracing & Evaluation | 多 Agent 调用链追踪、质量评估 | ✅ |
| Agent Arbitration | 多 Agent 分歧仲裁 | ✅ |
| Shared Workspace | 所有 Agent 可读全文、每人仅可写自己 section 的协作文档模型 | ✅ |
| CLI | 命令行交互入口，显示 memory / multi-agent metrics | ✅ |
| Streamlit Web UI | 可视化 Web 页面，展示 Task / Tool / RAG Sources / Memory / Handoff / Arbitration / Debug State | ✅ |

---

## 系统架构

```text
User Query
    ↓
Task Classification
    ↓
RAG Retriever  +  Memory Retriever
    ↓
Memory-aware Agent
    ↓
Multi-Agent Orchestrator
    ↓
Specialist Agents
    ↓
Handoff / Trace / Arbitration
    ↓
Final Answer
    ↓
Optional Memory Write-back
```

**流程说明：**

1. 用户输入进入 LangGraph 工作流，首先进行 Task Classification（paper / experiment / dataset / report / code / general）。
2. 根据 task_type 路由到对应 RAG Retriever，同时查询 Memory Retriever 获取相关历史上下文。
3. Memory-aware Agent 结合 RAG 结果和记忆上下文进行推理。
4. 如果启用了 Multi-Agent（`ENABLE_MULTI_AGENT=true`），Orchestrator 根据任务类型分发给 Specialist Agents 执行。
5. 各 Specialist Agent 执行后通过 Handoff 返回结果，Arbitration 处理分歧，Tracing 记录全链路。
6. 最终生成回答，并可选将重要交互写入 Memory。

---

## 目录结构

```text
ResearchAgent
├── app.py                          # Streamlit Web Demo 入口
├── run_cli.py                      # CLI 运行入口
├── README.md
├── requirements.txt
├── src/
│   └── research_agent/
│       ├── agents/                 # Multi-Agent System
│       │   ├── profiles.py         #   9 个 Agent 定义
│       │   ├── handoff.py          #   Agent 任务交接
│       │   ├── orchestrator.py     #   多 Agent 协调器
│       │   ├── executors.py        #   Specialist Executors
│       │   ├── tracing.py          #   调用链追踪
│       │   ├── arbitration.py      #   分歧仲裁
│       │   └── workspace.py        #   共享工作区
│       ├── memory/                 # Memory System
│       │   ├── schema.py           #   记忆数据结构
│       │   ├── store.py            #   记忆存储 (JSONL)
│       │   ├── writer.py           #   记忆写入
│       │   ├── retriever.py        #   记忆检索
│       │   ├── privacy_scope.py    #   隐私与作用域
│       │   ├── consolidation.py    #   记忆压缩 / 合并 / 过期
│       │   ├── adapters.py         #   模块适配器
│       │   └── memory_aware_agent.py
│       ├── rag/                    # RAG 子系统
│       │   ├── indexer.py
│       │   ├── retriever.py
│       │   ├── hybrid_retriever.py
│       │   ├── chunker.py
│       │   ├── reranker.py
│       │   ├── loaders.py
│       │   └── schemas.py
│       ├── ingestion/              # 文档摄入
│       │   ├── document_ingestor.py
│       │   └── mineru_client.py
│       ├── claim/                  # Claim Support 检索
│       ├── paper/                  # Paper Reading Pipeline
│       ├── progress/               # PPT Progress Memory
│       ├── report/                 # Report Writer
│       ├── graph/                  # LangGraph 工作流
│       ├── tools/                  # CSV / JSONL / Paper 工具
│       ├── llm/                    # LLM Enhancement Layer
│       ├── eval/                   # 评估模块
│       └── utils/                  # 工具函数
├── scripts/                        # 构建索引 / 摄入文档 / Memory 合并等脚本
├── data/
│   ├── papers/                     # 论文说明文档（示例）
│   ├── experiments/                # 实验说明文档（示例）
│   ├── datasets/                   # 数据集说明文档（示例）
│   ├── samples/                    # 示例数据
│   ├── memory/                     # 记忆存储（不提交）
│   ├── traces/                     # Multi-Agent 调用链（不提交）
│   ├── workspaces/                 # 共享工作区草稿（不提交）
│   ├── paper_notes/                # 论文笔记（不提交）
│   ├── ingested/                   # 摄入后的结构化文档（不提交）
│   └── progress_memory/            # 进展记忆（不提交）
├── storage/                        # Chroma 向量库（不提交）
├── reports/                        # 输出报告（不提交）
├── tests/                          # 测试
└── notebooks/                      # Jupyter Notebook（不提交）
```

**以下目录已加入 `.gitignore`，不会被提交到 Git：**

- `.env` — API Key 等敏感配置
- `.conda/` — Conda 环境
- `storage/` — Chroma 向量库
- `raw_docs/` — 原始私人文档
- `data/memory/*.jsonl` / `data/memory/*.md` / `data/memory/backups/` — 记忆数据
- `data/traces/` — Multi-Agent 调用链
- `data/workspaces/` — 共享工作区草稿
- `data/ingested/` — 摄入后的衍生文件
- `data/paper_notes/` — 论文笔记
- `data/progress_memory/` — 进展记忆
- `reports/` — 输出报告
- `notebooks/` — Jupyter Notebook

---

## 环境配置

### Python 版本

需要 **Python 3.11**。

### Conda 环境

```powershell
cd F:\ResearchAgent
conda create --prefix .\.conda python=3.11 -y
```

### 安装依赖

```powershell
.\.conda\python.exe -m pip install -r requirements.txt
```

### .env 配置

在项目根目录创建 `.env` 文件（不要提交到 Git）。以下为配置示例，**请替换为你的实际值，不要使用真实 key**：

```env
# =========================
# LLM API（可选，用于 LLM Enhancement / Report Writer / Memory-aware Agent / Multi-Agent）
# =========================
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-pro

# =========================
# Feature Toggles（默认全部关闭，规则版可独立运行）
# =========================
ENABLE_LLM_ENHANCEMENT=false
ENABLE_LLM_REPORT_WRITER=false
ENABLE_MEMORY_AWARE_AGENT=false
ENABLE_MULTI_AGENT=false

# =========================
# RAG 检索模式: vector（默认）| hybrid
# =========================
RAG_RETRIEVAL_MODE=vector

# =========================
# 文档摄入后端: local（默认）| mineru
# =========================
DOCUMENT_INGESTION_BACKEND=local

# =========================
# MinerU 配置（仅在 DOCUMENT_INGESTION_BACKEND=mineru 时需要）
# =========================
# MINERU_API_KEY=your_mineru_api_key_here
# MINERU_API_BASE_URL=https://mineru.net/api/v4/extract/task
# MINERU_API_MODE=precise
# MINERU_POLL_INTERVAL_SECONDS=3
# MINERU_MAX_WAIT_SECONDS=300

# =========================
# HuggingFace（首次运行需要联网下载 embedding model）
# =========================
# HF_HUB_OFFLINE=1    # 如果模型已缓存到本地，可设为 1 跳过联网
```

---

## Feature Toggles

| 功能 | 环境变量 | 默认值 | 说明 |
|---|---|---|---|
| LLM Enhancement（Claim / Paper / Progress / Report） | `ENABLE_LLM_ENHANCEMENT` | `false` | 开启后，规则版输出会交由 LLM 增强；关闭时所有模块使用纯规则版逻辑 |
| LLM-assisted Report Writer | `ENABLE_LLM_REPORT_WRITER` | `false` | 开启后 Report Writer 使用 LLM 生成汇报；关闭时使用模板版 |
| Memory-aware Agent | `ENABLE_MEMORY_AWARE_AGENT` | `false` | 开启后主 Agent 会检索记忆上下文并注入到推理中 |
| Multi-Agent Orchestrator | `ENABLE_MULTI_AGENT` | `false` | 开启后启用 9-Agent 协作；关闭时使用单 Agent 模式 |
| RAG Retrieval Mode | `RAG_RETRIEVAL_MODE` | `vector` | `vector` = 纯向量检索；`hybrid` = 向量 + 关键词混合检索 |
| Document Ingestion Backend | `DOCUMENT_INGESTION_BACKEND` | `local` | `local` = pymupdf / python-docx / python-pptx；`mineru` = MinerU API |
| MinerU API Mode | `MINERU_API_MODE` | `precise` | `precise` / `agent` / `local_fastapi` |
| HuggingFace Offline | `HF_HUB_OFFLINE` | 未设置 | 设为 `1` 可跳过 embedding model 联网下载（需已缓存） |

**关键设计原则：所有 LLM / API 功能默认关闭，规则版可独立运行，LLM 不可用时自动回退且不报错。**

---

## 构建索引

首次运行或更新 data 目录后，需要先构建本地向量索引：

```powershell
cd F:\ResearchAgent
.\.conda\python.exe scripts\build_index.py
```

说明：

- 首次运行会自动下载 HuggingFace embedding model（`sentence-transformers`），需要联网。
- 如果模型已缓存到本地，建议设置 `HF_HUB_OFFLINE=1` 跳过联网检查。
- 索引生成到 `storage/chroma_db`，该目录已在 `.gitignore` 中，不会提交到 Git。

---

## CLI 使用

```powershell
cd F:\ResearchAgent
.\.conda\python.exe run_cli.py
```

### 示例查询

```text
- OpenImages-MIAP 的性别标注是图像级还是 bbox 级？
- 帮我分析 data/experiments/sample_metrics.csv
- 帮我总结最近的共现关系幻觉研究进展
- 根据组会进展生成汇报大纲
```

### 输出信息

CLI 输出包含以下字段（根据启用的功能模块）：

| 字段 | 说明 |
|---|---|
| `final_answer` | Agent 最终回答 |
| `task_type` | 任务分类结果 |
| `tool_used` | 使用的工具名称 |
| `evidence_status` | 证据检查状态（passed / warning / failed） |
| `sources` | RAG 检索到的来源文档 |
| `memory_used` | 是否使用了记忆上下文 |
| `memory_count` | 检索到的记忆记录数 |
| `handoff_summary` | Multi-Agent 交接摘要（如果启用） |

---

## Streamlit Web UI

### 启动

```powershell
cd F:\ResearchAgent
.\.conda\python.exe -m streamlit run app.py --server.port 8503
```

### Windows 稳定启动

Windows 上推荐显式关闭 Streamlit file watcher，避免扫描大型依赖目录（如 `transformers`）时触发无关报错：

```powershell
.\.conda\python.exe -m streamlit run app.py --server.port 8504 --server.fileWatcherType none
```

项目已通过 `.streamlit/config.toml` 预置了默认端口和 watcher 配置。

### UI 展示内容

- **Task Classification** — 任务类型识别
- **Tool Analysis** — 工具调用及结果
- **RAG Sources** — 检索来源文档
- **Memory Metrics** — 记忆检索记录数 / 上下文
- **Multi-Agent Handoff** — Agent 交接详情
- **Arbitration** — 分歧仲裁结果
- **Debug State** — 完整状态快照

---

## Memory System

ResearchAgent v0.5 引入了完整的长期记忆子系统，支持三层记忆架构：

| 层级 | 存储 | 说明 |
|---|---|---|
| `long_term` | JSONL + Markdown summary | 跨会话持久化记忆 |
| `mid_term` | JSONL | 会话内中间记忆 |
| `short_term` | 运行态 | 当前查询上下文 |

### 作用域

| Scope | 说明 |
|---|---|
| `private` | 仅当前 Agent 可见 |
| `shared` | 同组 Agent 可见 |
| `global` | 所有 Agent 可见 |

### 核心模块

| 文件 | 功能 |
|---|---|
| `memory/schema.py` | 记忆数据结构定义（MemoryRecord） |
| `memory/store.py` | JSONL 文件存储，支持增删改查 |
| `memory/writer.py` | 记忆写入逻辑，含去重和合并 |
| `memory/retriever.py` | 基于关键词 + 时间衰减的记忆检索 |
| `memory/privacy_scope.py` | 隐私作用域控制 |
| `memory/consolidation.py` | 记忆压缩 / 合并 / 过期 / 摘要 |
| `memory/adapters.py` | 各模块（Claim / Paper / Progress）与 Memory 的适配器 |
| `memory/memory_aware_agent.py` | 主 Agent 的记忆感知层 |

### Memory Consolidation Safety

- **默认 dry-run**：运行 `scripts/run_memory_consolidation.py` 不加 `--apply` 只预览，不修改任何文件。
- **`--apply` 才写入**：显式传入 `--apply` 才会执行合并 / 压缩 / 过期。
- **自动备份**：写入前自动备份 `data/memory/*.jsonl` 到 `data/memory/backups/YYYYMMDD_HHMMSS/`。
- **幂等操作**：合并 / 压缩 / 过期只影响 `active` 状态记录，可安全重复执行。

```powershell
# 预览（安全 — 不写入）
python scripts/run_memory_consolidation.py

# 执行（带自动备份）
python scripts/run_memory_consolidation.py --apply

# 执行 + 周度摘要 + 过期清理
python scripts/run_memory_consolidation.py --apply --weekly --expire
```

> `data/memory/*.jsonl`、`data/memory/*.md`、`data/memory/backups/` 均在 `.gitignore` 中，不会提交到 Git。

---

## Multi-Agent System

v0.5 引入了一个 **9-Agent 协作架构**，由 Orchestrator 根据任务类型分发给对应的 Specialist Agent：

| Agent | 职责 |
|---|---|
| `coordinator_agent` | 任务分析、Agent 选择、结果汇总 |
| `paper_agent` | 论文检索、阅读、总结 |
| `experiment_agent` | 实验数据分析（CSV / JSONL / 指标） |
| `claim_agent` | 声明 / 论断支撑检索 |
| `progress_agent` | 组会进展追踪、PPT 大纲 |
| `report_agent` | 科研汇报生成 |
| `code_agent` | 代码 / 环境问题辅助 |
| `memory_agent` | 记忆存取管理 |
| `general_agent` | 通用科研问答 |

### 核心模块

| 文件 | 功能 |
|---|---|
| `agents/profiles.py` | 9 个 Agent 的能力定义、系统提示、工具绑定 |
| `agents/orchestrator.py` | Agent 注册、任务分析、分发协调 |
| `agents/handoff.py` | Agent 间任务交接、结果传递、来源追踪 |
| `agents/executors.py` | Specialist Agent 的具体执行逻辑 |
| `agents/tracing.py` | 多 Agent 调用链追踪、耗时统计 |
| `agents/arbitration.py` | 多 Agent 输出分歧时的仲裁逻辑 |
| `agents/workspace.py` | 共享工作区实现 |

### 关键设计

- **默认关闭**：`ENABLE_MULTI_AGENT=false`，系统以单 Agent 模式运行，不引入额外开销。
- **规则分发优先**：当前版本以规则路由为主，后续可接入 LLM-based Orchestrator。
- **失败隔离**：单个 Specialist Agent 失败不影响其他 Agent 和最终回答生成。

---

## Shared Workspace

Shared Workspace 是一个本地 Markdown / JSONL 协作模型，用于多 Agent 之间的信息共享：

| 特性 | 说明 |
|---|---|
| 可读性 | 所有 Agent 可读取全文 |
| 写入权限 | 每个 Agent 只能写入自己的 section |
| 越权处理 | 无权限写入转为 suggestion，等待审批 |
| 审批 | Coordinator 可 approve suggestion |
| 存储 | `data/workspaces/`（已在 `.gitignore` 中） |

---

## Document Ingestion / MinerU

### 本地模式（默认）

```powershell
.\.conda\python.exe scripts\ingest_documents.py --backend local
```

使用 `pymupdf` / `python-docx` / `python-pptx` 解析 PDF / DOCX / PPTX / MD 文件，无需 API key。

### MinerU 模式

适用于包含复杂公式、表格、OCR 内容的 PDF：

```powershell
# 通过环境变量
DOCUMENT_INGESTION_BACKEND=mineru

# 或通过 CLI
.\.conda\python.exe scripts\ingest_documents.py --backend mineru --input raw_docs/papers_pdf/test.pdf
```

### Fallback 机制

MinerU API 调用失败时，系统自动 fallback 到 local extraction，不会中断批量摄入流程。摄入后的文档在 YAML front matter 中记录 `ingestion_backend`、`mineru_used`、`mineru_error` 字段，便于审计。

---

## 测试覆盖

本项目在 `scripts/` 目录下维护了以下测试脚本，覆盖核心模块：

### Memory System

| 测试脚本 | 覆盖范围 |
|---|---|
| `test_memory_schema.py` | 记忆数据结构 |
| `test_memory_store.py` | JSONL 存储读写 |
| `test_memory_writer.py` | 记忆写入 + 去重 |
| `test_memory_retriever.py` | 记忆检索 |
| `test_memory_privacy.py` | 隐私作用域 |
| `test_memory_adapters.py` | 模块适配器 |
| `test_memory_consolidation.py` | 合并 / 压缩 / 过期 |
| `test_memory_writeback.py` | 回写集成 |
| `test_memory_aware_agent.py` | 记忆感知 Agent |

### Multi-Agent System

| 测试脚本 | 覆盖范围 |
|---|---|
| `test_agent_profiles.py` | Agent 定义 |
| `test_agent_handoff.py` | 任务交接 |
| `test_agent_executors.py` | Specialist Executors |
| `test_multi_agent_orchestrator.py` | 协调器 |
| `test_multi_agent_tracing.py` | 调用链追踪 |
| `test_agent_arbitration.py` | 分歧仲裁 |
| `test_agent_workspace.py` | 共享工作区 |
| `test_agent_demo.py` | 端到端集成 Demo |

### Core Pipelines

| 测试脚本 | 覆盖范围 |
|---|---|
| `test_claim_support.py` | Claim Support 检索 |
| `test_paper_reading.py` | Paper Reading Pipeline |
| `test_ppt_progress_memory.py` | PPT Progress Memory |

> **All core tests pass in the current local development environment.**

### 运行测试

```powershell
# 端到端集成 Demo（最推荐，覆盖全链路）
.\.conda\python.exe scripts\test_agent_demo.py

# Memory System
.\.conda\python.exe scripts\test_memory_store.py
.\.conda\python.exe scripts\test_memory_writer.py
.\.conda\python.exe scripts\test_memory_retriever.py

# Multi-Agent System
.\.conda\python.exe scripts\test_agent_profiles.py
.\.conda\python.exe scripts\test_agent_handoff.py
.\.conda\python.exe scripts\test_multi_agent_orchestrator.py
.\.conda\python.exe scripts\test_agent_workspace.py

# Core Pipelines
.\.conda\python.exe scripts\test_claim_support.py
.\.conda\python.exe scripts\test_paper_reading.py
.\.conda\python.exe scripts\test_ppt_progress_memory.py
```

> 部分测试依赖 HuggingFace embedding model 的本地缓存。如果网络不稳定，可先在 PowerShell 中执行 `$env:HF_HUB_OFFLINE="1"` 再运行测试。如果未配置 LLM API Key，LLM 相关模块会自动 fallback 到规则版，不影响测试通过。

---

## 常用脚本说明

| 脚本 | 用途 | 典型用法 |
|---|---|---|
| `scripts/build_index.py` | 构建 / 重建 Chroma 向量索引 | `.\.conda\python.exe scripts\build_index.py` |
| `scripts/ingest_documents.py` | 批量摄入 raw_docs/ 中的文档 | `.\.conda\python.exe scripts\ingest_documents.py --backend local` |
| `scripts/run_memory_consolidation.py` | Memory 合并 / 压缩 / 过期（默认 dry-run） | `.\.conda\python.exe scripts\run_memory_consolidation.py` |
| `scripts/test_agent_demo.py` | 端到端集成测试（CLI + Streamlit 流程） | `.\.conda\python.exe scripts\test_agent_demo.py` |
| `scripts/test_multi_agent_orchestrator.py` | Multi-Agent 协调器测试 | `.\.conda\python.exe scripts\test_multi_agent_orchestrator.py` |
| `scripts/test_agent_workspace.py` | Shared Workspace 权限读写测试 | `.\.conda\python.exe scripts\test_agent_workspace.py` |
| `scripts/run_benchmark.py` | 评估基准运行 | `.\.conda\python.exe scripts\run_benchmark.py` |
| `scripts/stop_streamlit.ps1` | 关闭 Windows 上的 Streamlit 进程 | `.\scripts\stop_streamlit.ps1` |
| `scripts/export_demo_logs.py` | 导出 Demo 运行日志 | `.\.conda\python.exe scripts\export_demo_logs.py` |

---

## 安全与 Git Ignore

以下内容**绝对不能**提交到 GitHub：

| 内容 | 说明 |
|---|---|
| `.env` | 包含 API Key 等敏感配置 |
| `.conda/` | Conda 本地环境 |
| `storage/` | Chroma 向量库 |
| `raw_docs/` | 原始私人文档 |
| `data/memory/*.jsonl` | 记忆数据（含私人研究记录） |
| `data/memory/*.md` | 记忆摘要 |
| `data/memory/backups/` | 记忆备份 |
| `data/traces/` | Multi-Agent 调用链 |
| `data/workspaces/` | 共享工作区草稿 |
| `data/ingested/` | 摄入后的衍生文件 |
| `data/paper_notes/` | 论文笔记 |
| `data/progress_memory/` | 进展记忆 |
| `reports/` | 输出报告 |

以上路径均已写入 `.gitignore`。项目提供了 `.env.example` 作为环境变量模板，便于他人快速上手。

---

## Troubleshooting / 常见问题

### SOCKS proxy / socksio issue on Windows

如果运行 `test_agent_demo.py`、Streamlit 或 LLM 相关功能时出现以下错误：

- `No module named 'socksio'`
- `SOCKS proxy error`
- `ProxyError`
- `WinError 10054`
- `ConnectionResetError`
- `httpx.ConnectError`

**这不是 ResearchAgent 核心代码 bug**，通常是 Windows 系统代理或 Python HTTP 客户端缺少 SOCKS 支持导致。核心 RAG / Memory / Multi-Agent 流程不受影响。

**解决方案（按推荐顺序）：**

1. **临时关闭系统代理**，重新运行。

2. **安装 SOCKS 支持**：

   ```powershell
   .\.conda\python.exe -m pip install "httpx[socks]" socksio
   ```

3. **使用离线 embedding**（如果模型已缓存）：

   ```powershell
   $env:HF_HUB_OFFLINE="1"
   .\.conda\python.exe scripts\test_agent_demo.py
   ```

4. **在 `.env` 中关闭 LLM 相关 toggles**（不需要 LLM 时可完全跳过网络请求）：

   ```env
   ENABLE_LLM_ENHANCEMENT=false
   ENABLE_LLM_REPORT_WRITER=false
   ```

5. **规则版 RAG / Memory / Multi-Agent 流程仍可正常运行**，无需任何 API key 或网络连接（首次 embedding 模型缓存之后）。

### HuggingFace 模型首次下载失败

首次运行 `build_index.py` 时需要从 HuggingFace Hub 下载 `sentence-transformers` embedding model。如果网络受限：

- 在有网络的机器上先跑一次 `build_index.py`，模型会自动缓存到本地。
- 之后在离线环境中设置 `$env:HF_HUB_OFFLINE="1"` 即可。
- 或手动下载模型放到 HuggingFace cache 目录（通常为 `C:\Users\<user>\.cache\huggingface\`）。

### MinerU PDF 解析失败

如果 `DOCUMENT_INGESTION_BACKEND=mineru` 但 PDF 解析返回空或失败：

- 系统会自动 **fallback 到 local extraction**（pymupdf），不会中断摄入流程。
- 检查 `.env` 中的 `MINERU_API_KEY` 和 `MINERU_API_BASE_URL` 是否正确。
- 如果只是偶尔失败，可调大 `MINERU_MAX_WAIT_SECONDS`（默认 300 秒）。
- 仅需简单 PDF/DOCX/PPTX/MD 解析时，使用默认的 `local` backend 即可，无需 API key。

### Streamlit 启动时 torch/torchvision watcher 警告

Windows 上启动 Streamlit 时可能看到 `torch` 或 `torchvision` 相关的 file watcher 错误：

```powershell
# 使用 --server.fileWatcherType none 关闭 watcher
.\.conda\python.exe -m streamlit run app.py --server.port 8504 --server.fileWatcherType none
```

项目已通过 `.streamlit/config.toml` 预置了默认配置，也可在启动时显式指定端口和 watcher 参数。

---

## 当前限制

需要诚实说明的是，这仍然是一个 **preview / demo 级别**的项目：

1. **不是生产级多人并发系统** — 所有 Agent 在同一进程中运行，无消息队列、无分布式调度。
2. **Shared Workspace 是本地协作模型** — 基于 Markdown / JSONL 文件，不是数据库级别的并发控制。
3. **Multi-Agent execution v1 以规则分发为主** — 当前以本地模块调用和规则路由为核心，尚未接入 LLM-based 动态编排。
4. **Reranker 是 heuristic 实现** — 目前是规则打分，后续可替换为 CrossEncoder 模型以获得更好的重排序效果。
5. **HuggingFace embedding 首次运行需联网** — `sentence-transformers` 模型首次使用需要下载，或提前缓存并设置 `HF_HUB_OFFLINE=1`。
6. **MinerU 需要 API key** — 不属于项目自带能力，失败时自动 fallback 到 local extraction。
7. **Web UI 是 demo 级 Streamlit** — 重点是功能跑通和 Debug 可视化，不是生产级交互设计。
8. **LLM Enhancement 依赖第三方 API** — 所有 LLM 增强功能默认关闭，规则版可独立运行，开启后依赖 API key 和网络。

---

## Roadmap

| 计划 | 说明 |
|---|---|
| CrossEncoder Reranker | 用 CrossEncoder 替换当前 heuristic reranker，提升检索精度 |
| Chroma Memory Collection | 为 Memory System 增加向量化检索能力 |
| Workspace → Orchestrator 接入 | 将 Shared Workspace 更深度集成到多 Agent 协调流程 |
| Paper Figure / Table Understanding | 增强论文图表理解能力 |
| Evaluation Dashboard | 构建评估指标的可视化面板 |
| Docker Packaging | 提供 Docker 镜像，降低环境配置门槛 |

---

## 运行建议

推荐按以下顺序操作，便于定位问题：

1. **创建 Conda 环境 + 安装依赖**
2. **构建 RAG 索引** — `.\.conda\python.exe scripts\build_index.py`
3. **运行 CLI 或 Streamlit Demo** — 确认核心流程正常
4. **按需开启 Feature Toggle** — 在 `.env` 中逐步启用 LLM Enhancement / Memory-aware Agent / Multi-Agent

---

## 项目亮点

- **LangGraph 多节点工作流** — 显式状态管理和条件路由组织 Agent 流程。
- **Agentic RAG** — 根据任务类型选择不同资料库检索范围，而不是全库混合检索。
- **完整 Memory 子系统** — Schema / Store / Writer / Retriever / Privacy / Consolidation / Adapters 七层架构，支持 dry-run 安全操作。
- **9-Agent 协作架构** — Orchestrator + Handoff + Arbitration + Tracing 完整链路。
- **Shared Workspace** — Section 级权限的本地协作模型。
- **可选 LLM 增强** — 所有 LLM 功能默认关闭、可独立运行、失败自动回退。
- **MinerU Fallback** — 复杂 PDF 解析失败时自动回退 local extraction。
- **实验工具调用** — 支持 CSV 和 JSONL 实验文件分析。
- **Evidence Checker** — 在最终回答前检查 Sources 或工具结果支撑。
- **Report Writer** — 支持组会讲稿、PPT 大纲、论文摘要三种风格，模板 / LLM 双模式。
- **Streamlit Web Demo** — 完整 Debug 可视化，含 Task / Tool / RAG / Memory / Handoff / Arbitration。
- **工程化结构** — `src/`、`scripts/`、`data/`、`tests/` 等目录组织，便于扩展和展示。
