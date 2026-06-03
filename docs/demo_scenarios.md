# ResearchAgent Demo Scenarios

This document describes the six core demo scenarios of ResearchAgent, with expected behaviors verified in the Day21 integration test.

All scenarios are runnable via CLI (`python run_cli.py`) or Streamlit Web UI (`streamlit run app.py`).

---

## 1. CSV Experiment Analysis

**User Input:**

```
请分析 data/experiments/sample_metrics.csv
```

**Expected Behavior:**

| Field | Expected Value |
|-------|----------------|
| `task_type` | `experiment_analysis` |
| `tool_used` | `csv_analyzer` |
| `evidence_status` | `passed` |

**Demo Output Highlights:**

- CSV file detected: `data/experiments/sample_metrics.csv`
- Rows: 3, Columns: 6
- Columns: `run_tag` (str), `dataset` (str), `mean_extra_object_rate` (float64), `mean_precision` (float64), `mean_recall` (float64), `hrs_v1` (float64)
- Numeric summary:
  - `mean_extra_object_rate`: mean=0.1867, min=0.1500, max=0.2300
  - `mean_precision`: mean=0.8167, min=0.7800, max=0.8500
  - `mean_recall`: mean=0.7133, min=0.6900, max=0.7400
  - `hrs_v1`: mean=0.3267, min=0.2800, max=0.3900
- Top rows previewed (3 rows from `coco_val_n100_g1`, `coco_val_n300_g1`, `sd35_gender_swap`)
- RAG retrieved 3 experiment docs; Sources include `data/experiments/coco_val_n100_g1.md`
- Evidence reason: "回答同时包含 RAG 检索来源和成功的工具分析结果"

**What this demonstrates:**
- Agent correctly classifies the task as `experiment_analysis`
- Tool router detects `.csv` path and dispatches `csv_analyzer`
- CSV analyzer returns structured stats (row count, column types, numeric distributions)
- RAG enriches the answer with experiment documentation
- Evidence Checker confirms both RAG sources and tool results are present

---

## 2. JSONL Generation Log Analysis

**User Input:**

```
请分析 data/experiments/sample_generations.jsonl
```

**Expected Behavior:**

| Field | Expected Value |
|-------|----------------|
| `task_type` | `experiment_analysis` |
| `tool_used` | `jsonl_analyzer` |
| `evidence_status` | `passed` |

**Demo Output Highlights:**

- JSONL file detected: `data/experiments/sample_generations.jsonl`
- Record count: 5, Malformed lines: 0
- Fields detected:
  - `dataset` (str), `extra_objects` (list), `hallucination` (bool)
  - `image_id` (str), `prompt` (str), `response` (str), `run_tag` (str)
- Boolean field distribution: `hallucination`: true=2, false=3
- List field length stats: `extra_objects` — count=5, mean_length=0.60, min=0, max=1
- Categorical distributions:
  - `run_tag`: {coco_val_n100_g1: 2, coco_val_n300_g1: 2, sd35_gender_swap: 1}
  - `dataset`: {COCO2017: 4, synthetic_sd35: 1}
- First 5 records previewed with image_id, run_tag, hallucination flags
- RAG retrieved 3 experiment docs; Sources from `coco_val_n100_g1.md`, `sd35_gender_swap.md`, `coco_val_n300_g1.md`
- Evidence reason: "回答同时包含 RAG 检索来源和成功的工具分析结果"

**What this demonstrates:**
- Agent correctly dispatches `jsonl_analyzer` for `.jsonl` files
- JSONL analyzer provides field type inference, boolean distributions, list-length stats
- Cross-references with RAG knowledge base for experiment context

---

## 3. Dataset RAG QA

**User Input:**

```
OpenImages-MIAP 的性别标注是图像级还是 bbox 级？
```

**Expected Behavior:**

| Field | Expected Value |
|-------|----------------|
| `task_type` | `dataset_recommendation` |
| `tool_used` | `none` |
| `evidence_status` | `passed` |
| Sources must include | `data/datasets/openimages_miap.md` |

**Demo Output Highlights:**

- Task classified as `dataset_recommendation` — a dataset-specific knowledge question
- No tool invoked (no CSV/JSONL path detected)
- RAG retrieves 3 dataset docs; top chunks from `openimages_miap.md` and `gqa.md`
- Answer draws from the key metadata: `annotation_type: bbox_level_sensitive_attribute`
- Answer explains: "OpenImages-MIAP 的关键特点是属性更接近 person-level 或 bbox-level，而不是简单的 image-level 标签"
- Sources: `data/datasets/openimages_miap.md` (primary), `data/datasets/gqa.md` (secondary)
- Evidence reason: "回答基于 RAG 检索来源，可追踪到本地资料库"

**What this demonstrates:**
- Pure knowledge QA without tool invocation
- RAG routes to `dataset_doc` source type based on task classification
- Metadata-driven answer (annotation_type, dataset properties)
- Sources are traceable to specific markdown files

---

## 4. Report Generation

**User Input:**

```
帮我生成 coco_val_n300_g1 实验的组会汇报文本
```

**Expected Behavior:**

| Field | Expected Value |
|-------|----------------|
| `task_type` | `report_generation` |
| `evidence_status` | `passed` |
| Sources | Not empty (4 sources observed) |
| LLM enabled | Uses "LLM-assisted Report Writer" with style detection |
| LLM disabled | Falls back to Template Report Writer |

**Demo Output Highlights (LLM-assisted mode):**

- Task classified as `report_generation`, no tool invoked
- Report style auto-detected: `group_meeting`
- RAG retrieves 4 documents: `coco.md`, `coco_val_n300_g1.md`, `coco_val_n100_g1.md`, `sd35_gender_swap.md`
- Structured report generated with sections:
  1. 研究背景与动机
  2. 实验目标
  3. 数据与方法流程
  4. 关键指标与现阶段结果
  5. 局限与后续工作
- Sources: 4 documents with paths and titles
- Evidence status: `passed`

**Additional report styles tested:**

| User Input (paraphrase) | Detected Style |
|--------------------------|----------------|
| "请给我一份 OpenImages-MIAP 数据集相关的 PPT 汇报草稿" | `ppt_slide` |
| "请总结 Guardrail-Agnostic 这篇论文，生成组会汇报内容" | `summary` |

**What this demonstrates:**
- Report Writer node generates structured, sectioned output
- LLM-assisted mode auto-detects report style (group_meeting / ppt_slide / summary)
- All claims are grounded in RAG-retrieved documents
- Template fallback available when LLM is not configured

---

## 5. General Advice (Weak Evidence)

**User Input:**

```
我今天应该怎么安排科研任务
```

**Expected Behavior:**

| Field | Expected Value |
|-------|----------------|
| `task_type` | `general` |
| `tool_used` | `none` |
| `evidence_status` | `weak` |
| Sources | Empty |

**Demo Output Highlights:**

- Task classified as `general` — open-ended, no specific domain
- No RAG documents retrieved (no matching source type)
- No tool invoked
- Answer: generic research planning advice
- Evidence status: `weak`
- Evidence reason: "当前回答没有 RAG Sources 或工具结果支撑，属于弱证据回答"
- Evidence warnings:
  - "未检索到相关资料源"
  - "未调用成功的本地分析工具"

**What this demonstrates:**
- Evidence Checker correctly identifies unsupported answers
- Agent is transparent about weak evidence — doesn't fabricate sources
- `general` task type gracefully handles open-ended queries

---

## 6. Code Help (Weak Evidence / Mock)

**User Input:**

```
ModuleNotFoundError: No module named langgraph 怎么解决
```

**Expected Behavior:**

| Field | Expected Value |
|-------|----------------|
| `task_type` | `code_help` |
| `tool_used` | `none` |
| `evidence_status` | `weak` |

**Demo Output Highlights:**

- Task classified as `code_help`
- No RAG documents retrieved (code help not in knowledge base)
- No tool invoked
- Answer: generic code environment troubleshooting advice
- Evidence status: `weak`
- Evidence reason: "当前回答没有 RAG Sources 或工具结果支撑，属于弱证据回答"

**What this demonstrates:**
- Code help intent is recognized even without domain-specific knowledge
- Current implementation is mock/weak — transparent about limitations
- Extensible: future versions can integrate real code assistant or Stack Overflow retrieval

---

## Agent Architecture Overview

All six scenarios flow through a common LangGraph pipeline:

```text
User Input
    │
    ▼
[classify_task] ──► task_type + classification_reason
    │
    ▼
[route_task] ──► paper | experiment | dataset | code | report | general
    │
    ├── RAG retrieval (filtered by source_type)
    ├── Tool calling (csv_analyzer / jsonl_analyzer if path detected)
    │
    ▼
[evidence_check] ──► evidence_status (passed | weak)
    │
    ▼
[final_answer] ──► answer + sources + debug info
```

Key design decisions:
- **Agentic RAG**: source_type filtering per task type (e.g., experiment_analysis → experiment_doc)
- **Tool Router**: detects file paths in user input and dispatches the correct analyzer
- **Evidence Checker**: rule-based check on RAG sources + tool results
- **Report Writer**: pluggable — template fallback or LLM-assisted generation

---

## Demo Checklist

- [x] CLI runnable — `python run_cli.py` accepts user input and prints structured output
- [x] Web UI runnable — `streamlit run app.py` on port 8503
- [x] RAG Sources displayed — each answer includes source file paths and metadata
- [x] CSV / JSONL tools invokable — file paths in user input trigger analyzers
- [x] Evidence Checker shows status — `passed` or `weak` with reasoning and warnings
- [x] Report Writer generates drafts — structured meeting notes, PPT outlines, paper summaries

---

## Running the Demo

### CLI Mode

```bash
# Build the RAG index first
python scripts/build_index.py

# Run the CLI
python run_cli.py
```

### Web UI Mode

```bash
# Start Streamlit (configured for port 8503)
streamlit run app.py
```

Open `http://localhost:8503` in your browser.

### Individual Component Tests

```bash
python scripts/test_agent_demo.py        # Full workflow test
python scripts/test_tool_router.py        # CSV/JSONL dispatch
python scripts/test_tool_augmented_agent.py  # Tool + RAG integration
python scripts/test_evidence_checker.py   # Evidence grading
python scripts/test_report_writer.py      # Template report writer
python scripts/test_llm_report_writer.py  # LLM-assisted report writer
```
