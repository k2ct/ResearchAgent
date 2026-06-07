# ResearchAgent Project Structure and Module Guide

> **Version**: v0.5 — Memory + Multi-Agent Preview  
> **Last Updated**: 2026-06-07

---

## 1. Project Overview

**ResearchAgent** is a local-first, memory-aware research assistant agent system built with LangGraph. It supports:

- **RAG retrieval** — vector + keyword hybrid search over a local paper/experiment/dataset knowledge base
- **Paper reading** — structured note extraction from academic papers
- **Experiment analysis** — CSV/JSONL tool-augmented analysis of experiment outputs
- **Claim support** — evidence-grounded argument building from local literature
- **Long-term memory** — JSONL-backed persistent memory with privacy scoping, consolidation, and multi-dimensional retrieval
- **Multi-agent collaboration** — 9 specialist agents with handoff, arbitration, shared workspace, and tracing
- **Report writing** — structured group-meeting report and PPT slide generation
- **Streamlit Web UI** — interactive query interface with debug state display
- **CLI interface** — lightweight terminal entry point for rapid testing

All processing is **local** — no proprietary data leaves the machine unless the user explicitly configures a remote LLM API or MinerU API.

---

## 2. High-Level Architecture

```
                        ┌─────────────────────┐
                        │     User Query       │
                        │  (Web UI / CLI)      │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   LangGraph          │
                        │   Workflow Engine    │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   Task Classification│
                        │   (LLM + Rule fallback)│
                        └──────────┬──────────┘
                                   │
          ┌────────────┬───────────┼───────────┬──────────┐
          ▼            ▼           ▼           ▼          ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Paper    │ │Experiment│ │ Dataset  │ │ Report   │ │ Code /   │
    │ Node     │ │ Node     │ │ Node     │ │ Node     │ │ General  │
    └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
         └──────────────┴────────────┴────────────┴────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  Memory Retrieval   │
                        │  (Memory-Aware Node) │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │  ENABLE_MULTI_AGENT?        │
                    └──────────┬─────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │ YES            │                │ NO
              ▼                │                ▼
    ┌──────────────────┐       │      ┌──────────────────┐
    │ Multi-Agent       │       │      │ Evidence Check   │
    │ Router            │       │      │ (sources + tools)│
    │ ┌──────────────┐  │       │      └────────┬─────────┘
    │ │ Orchestrator │  │       │               │
    │ │ → Handoff    │  │       │               ▼
    │ │ → Executors  │  │       │      ┌──────────────────┐
    │ │ → Arbitration│  │       │      │ Final Answer     │
    │ │ → Trace      │  │       │      │ (merge memory    │
    │ └──────┬───────┘  │       │      │  context if any) │
    └────────┼──────────┘       │      └────────┬─────────┘
             │                  │               │
             ▼                  │               │
    ┌──────────────────┐       │               │
    │ Evidence Check   │◄──────┘               │
    └────────┬─────────┘                       │
             │                                 │
             └─────────────┬───────────────────┘
                           ▼
                  ┌──────────────────┐
                  │  Final Answer    │
                  │  + Sources       │
                  │  + Memory Context│
                  │  + Handoff Summary│
                  └──────────────────┘
```

---

## 3. Directory Structure

```
ResearchAgent/
├── src/research_agent/          # Main Python package
│   ├── agents/                  # Multi-Agent System (9 agents)
│   │   ├── profiles.py          # Agent role definitions
│   │   ├── handoff.py           # Agent-to-agent handoff structures
│   │   ├── orchestrator.py      # Multi-agent dispatch pipeline
│   │   ├── executors.py         # Real specialist agent executors
│   │   ├── tracing.py           # Multi-agent trace recording
│   │   ├── arbitration.py       # Conflict detection & coordinator ruling
│   │   └── workspace.py         # Shared workspace with section-level permissions
│   │
│   ├── memory/                  # Memory System
│   │   ├── schema.py            # MemoryRecord dataclass & validation
│   │   ├── store.py             # JSONL + Markdown summary storage
│   │   ├── writer.py            # Rule-based memory creation decision engine
│   │   ├── retriever.py         # Multi-dimensional memory search
│   │   ├── privacy_scope.py     # Access control & scope filtering
│   │   ├── consolidation.py     # Merge, compress, expire, stage summary
│   │   ├── adapters.py          # Phase-2 module → Memory System bridge
│   │   └── memory_aware_agent.py # Query-time memory augmentation
│   │
│   ├── rag/                     # Retrieval-Augmented Generation
│   │   ├── loaders.py           # Document loaders (Markdown → LangChain Document)
│   │   ├── schemas.py           # SourceType, DocumentMetadata definitions
│   │   ├── indexer.py           # Chroma vector store builder
│   │   ├── retriever.py         # Vector-based document retrieval
│   │   ├── hybrid_retriever.py  # Vector + keyword hybrid retrieval with fusion
│   │   ├── reranker.py          # Heuristic re-ranking of retrieved documents
│   │   └── chunker.py           # Configurable document chunking
│   │
│   ├── ingestion/               # Document Ingestion Pipeline
│   │   ├── document_ingestor.py # PDF/DOCX/PPTX/MD → YAML front-matter Markdown
│   │   └── mineru_client.py     # Optional MinerU API client for complex PDFs
│   │
│   ├── claim/                   # Claim Support Retrieval
│   │   └── claim_support.py     # Evidence-grounded argument building from RAG
│   │
│   ├── paper/                   # Paper Reading Pipeline
│   │   └── paper_reader.py      # Structured reading notes from paper Markdown
│   │
│   ├── progress/                # PPT Progress Memory
│   │   └── ppt_progress_memory.py # Research progress extraction from slide docs
│   │
│   ├── report/                  # Report Writer
│   │   ├── report_writer.py     # Template-based reporting (non-LLM fallback)
│   │   └── llm_report_writer.py # LLM-assisted report generation
│   │
│   ├── graph/                   # LangGraph Main Workflow
│   │   ├── state.py             # AgentState TypedDict definition
│   │   ├── nodes.py             # All workflow nodes (classify, task, memory, evidence, final)
│   │   ├── router.py            # Task routing logic
│   │   ├── workflow.py          # Graph builder (StateGraph construction)
│   │   ├── evidence_checker.py  # Evidence quality assessment
│   │   └── llm_classifier.py    # LLM-based task classification
│   │
│   ├── tools/                   # Tool-Augmented Analysis
│   │   ├── tool_router.py       # Query → tool dispatch
│   │   ├── csv_analyzer.py      # CSV metrics analysis
│   │   ├── jsonl_analyzer.py    # JSONL generation log analysis
│   │   ├── dataset_tools.py     # Dataset lookup & recommendation
│   │   ├── experiment_tools.py  # Experiment data analysis helpers
│   │   ├── paper_tools.py       # Paper-related utilities
│   │   ├── code_tools.py        # Code assistance utilities
│   │   └── file_utils.py        # Generic file I/O helpers
│   │
│   ├── llm/                     # LLM Integration
│   │   ├── client.py            # LLM client wrapper
│   │   ├── prompts.py           # Prompt templates
│   │   └── enhancers.py         # LLM-based text enhancement
│   │
│   ├── eval/                    # Evaluation
│   │   ├── benchmark.py         # Automated benchmark runner
│   │   └── metrics.py           # Quality metrics (evidence, memory, multi-agent)
│   │
│   └── utils/                   # Shared Utilities
│       ├── config.py            # Environment & configuration management
│       ├── file_utils.py        # Common file operations
│       └── logging_utils.py     # Logging configuration
│
├── app.py                       # Streamlit Web UI entry point
├── run_cli.py                   # CLI entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore rules
│
├── data/                        # Data directories
│   ├── papers/                  # Paper source markdown (committable samples)
│   ├── datasets/                # Dataset reference markdown (committable)
│   ├── experiments/             # Experiment description + sample data (committable)
│   ├── ingested/                # Ingested markdown output (git-ignored)
│   ├── memory/                  # Memory JSONL + MD summary (JSONL git-ignored)
│   ├── workspaces/              # Multi-agent workspace drafts (git-ignored)
│   ├── traces/                  # Multi-agent trace logs (git-ignored)
│   └── paper_notes/             # Paper reading notes output
│
├── scripts/                     # Testing & utility scripts
│   ├── build_index.py           # Build Chroma vector index
│   ├── ingest_documents.py      # Batch document ingestion
│   ├── run_benchmark.py         # Run evaluation benchmark
│   ├── run_memory_consolidation.py # Run memory consolidation
│   ├── test_*.py                # ~40+ test scripts for each module
│   └── stop_streamlit.ps1       # Windows Streamlit process killer
│
├── storage/                     # Chroma vector database (git-ignored)
├── raw_docs/                    # Raw input documents (git-ignored, private)
├── docs/                        # Project documentation
├── notebooks/                   # Jupyter notebooks (git-ignored)
├── reports/                     # Generated reports output
└── tests/                       # Unit & integration tests
```

---

## 4. graph/ — LangGraph Main Workflow

### 4.1 Files

| File | Purpose |
|---|---|
| [state.py](src/research_agent/graph/state.py) | Defines `AgentState` TypedDict — the shared state object flowing through all LangGraph nodes |
| [nodes.py](src/research_agent/graph/nodes.py) | All workflow node functions: classify, task-specific nodes, memory retrieval, evidence check, final answer, multi-agent router |
| [router.py](src/research_agent/graph/router.py) | Conditional edge routing: maps `task_type` to the correct downstream node |
| [workflow.py](src/research_agent/graph/workflow.py) | `build_graph()` — assembles the StateGraph, adds nodes and edges, supports conditional multi-agent path |
| [llm_classifier.py](src/research_agent/graph/llm_classifier.py) | LLM-based task classification with prompt engineering |
| [evidence_checker.py](src/research_agent/graph/evidence_checker.py) | Validates whether an answer has sufficient source/tool evidence |

### 4.2 AgentState Fields

`AgentState` carries all information through the workflow:

| Field | Type | Purpose |
|---|---|---|
| `query` | `str` | Raw user input |
| `task_type` | `str` | Classified task type (e.g. `paper_question`) |
| `result` | `str` | Intermediate answer from task node |
| `final_answer` | `str` | Final composed answer with sources, memory, evidence |
| `retrieved_docs` | `List[Dict]` | RAG-retrieved documents |
| `sources` | `List[Dict]` | Extracted source metadata |
| `tool_used` / `tool_result` | `str` / `Dict` | Tool invocation results |
| `evidence_status` | `str` | `passed` / `weak` — evidence quality |
| `memory_context` | `str` | Formatted memory snippets for the answer |
| `retrieved_memories` | `List[Dict]` | Raw memory records retrieved |
| `multi_agent_enabled` | `bool` | Whether multi-agent path was activated |
| `handoff_plan` / `handoff_results` | `Dict` / `List[Dict]` | Multi-agent orchestration results |
| `memory_written` | `bool` | Whether results were persisted to Memory Store |

### 4.3 Workflow Execution Flow

```
START → classify_task → [task_node] → retrieve_memory
  → [multi_agent_router if enabled] → evidence_check → final_answer → END
```

---

## 5. rag/ — Retrieval-Augmented Generation

### 5.1 Files

| File | Purpose |
|---|---|
| [loaders.py](src/research_agent/rag/loaders.py) | Loads Markdown files from `data/` subdirectories, parses YAML front matter, builds LangChain `Document` objects with metadata |
| [schemas.py](src/research_agent/rag/schemas.py) | Defines `SourceType` enum (`paper_note`, `experiment_doc`, `dataset_doc`, `slide_doc`, etc.) and `DocumentMetadata` TypedDict |
| [indexer.py](src/research_agent/rag/indexer.py) | Builds Chroma vector store from loaded documents; supports chunking configuration |
| [retriever.py](src/research_agent/rag/retriever.py) | Primary retrieval API — `retrieve_documents()`, `document_to_dict()`, `extract_sources_from_docs()`, `format_retrieved_docs()`. Supports task-type filtering and score retrieval |
| [hybrid_retriever.py](src/research_agent/rag/hybrid_retriever.py) | Vector + keyword hybrid retrieval with score fusion. Tokenizes Chinese/English text, performs keyword search on raw documents, fuses with vector similarity scores (default 70/30 weighting) |
| [reranker.py](src/research_agent/rag/reranker.py) | Heuristic re-ranking of fused results (title match bonus, task-type match bonus) |
| [chunker.py](src/research_agent/rag/chunker.py) | Configurable document chunking (by headings, paragraphs, or fixed size) |

### 5.2 Data Flow

```
data/{papers,experiments,datasets}/*.md
  → loaders.py (parse front matter, build Document)
  → indexer.py (chunk → embed → Chroma)
  → retriever.py / hybrid_retriever.py (query → search → filter → format)
  → graph/nodes.py (inject into AgentState)
```

### 5.3 Retrieval Modes

- **Vector** (default): Chroma similarity search with optional metadata filtering by `source_type`
- **Hybrid**: vector search + keyword token search → score fusion → optional heuristic rerank
- Controlled by `RAG_RETRIEVAL_MODE` env var or explicit parameter

---

## 6. ingestion/ — Document Ingestion

### 6.1 Files

| File | Purpose |
|---|---|
| [document_ingestor.py](src/research_agent/ingestion/document_ingestor.py) | Converts PDF, DOCX, PPTX, Markdown to YAML-front-matter Markdown. Local backend uses pymupdf/python-docx/python-pptx |
| [mineru_client.py](src/research_agent/ingestion/mineru_client.py) | Optional MinerU API client for complex PDFs (tables, formulas, OCR). Falls back to local extraction on failure |

### 6.2 Pipeline

```
raw_docs/*.{pdf,docx,pptx,md}
  → detect file type
  → extract text (pymupdf / python-docx / python-pptx / mineru API)
  → build YAML front matter (title, source_type, date, etc.)
  → write to data/ingested/*.md
  → ready for RAG indexing (scripts/build_index.py)
```

---

## 7. memory/ — Memory System

### 7.1 Files

| File | Purpose |
|---|---|
| [schema.py](src/research_agent/memory/schema.py) | `MemoryRecord` dataclass with full field specification. Defines 4 enums (`MemoryLevel`, `MemoryScope`, `MemoryType`, `AgentRole`) + validation + serialisation (dict/JSONL) |
| [store.py](src/research_agent/memory/store.py) | JSONL-based persistent storage. Organised by level (long/mid/short) and scope (shared). Supports append, query, markdown summary, archive, clear short-term |
| [writer.py](src/research_agent/memory/writer.py) | Rule-based decision engine: classifies content into `memory_level` (long/mid/short), `memory_scope` (private/shared/global), `memory_type`, `tags`, and `shared_with`. Handles explicit user commands and auto-promotion |
| [retriever.py](src/research_agent/memory/retriever.py) | Multi-dimensional memory search with filters: `memory_type`, `memory_level`, `owner_agent`, `tags`, `keyword`, `time_range`, `importance`, `status`. Store-backed with JSONL fallback |
| [privacy_scope.py](src/research_agent/memory/privacy_scope.py) | Access control: classifies scope (private/shared/global), checks `can_agent_read()`, filters records by agent, validates scope on write |
| [consolidation.py](src/research_agent/memory/consolidation.py) | Maintenance operations: `merge_duplicates`, `compress_long_memories`, `mark_expired_memories`, `generate_stage_summary`, `run_consolidation` (all-in-one) |
| [adapters.py](src/research_agent/memory/adapters.py) | Bridge functions per source module: `save_claim_support_result()`, `save_paper_reading_result()`, `save_progress_result()`, `save_experiment_result()`, `save_report_result()` → delegates to `writer.write_memory_from_source()` |
| [memory_aware_agent.py](src/research_agent/memory/memory_aware_agent.py) | Query-time augmentation: `retrieve_memories_for_query()`, `format_memory_context()`. Filters by relevance, level, and status. Called by `retrieve_memory_node` in the LangGraph workflow |

### 7.2 Memory Levels & Lifecycle

| Level | Scope | Typical TTL | Examples |
|---|---|---|---|
| `short_term` | private | session | Current query context, tool output |
| `mid_term` | private/shared | days–weeks | Weekly experiment results, paper notes |
| `long_term` | shared/global | indefinite | Research direction, project decisions, user preferences |

### 7.3 Memory Types

`paper_note`, `experiment_result`, `claim_support`, `progress_update`, `report_summary`, `code_note`, `research_direction`, `project_decision`, `user_preference`, `todo`, `issue`, `meeting_note`, `general_note`

### 7.4 Data Flow

```
Module Output (claim/paper/progress/report/tools)
  → adapters.py (extract content, set source metadata)
  → writer.py (classify level/scope/type, validate scope)
  → store.py (append to JSONL, regen summary)
  → retriever.py (multi-dimensional query)
  → memory_aware_agent.py (retrieve + format for query)
  → graph/nodes.py (inject into AgentState.memory_context)
```

---

## 8. agents/ — Multi-Agent System

### 8.1 Files

| File | Purpose |
|---|---|
| [profiles.py](src/research_agent/agents/profiles.py) | Defines 9 `AgentProfile` dataclasses with role, responsibilities, task types, memory permissions, tool permissions, system prompts, handoff targets. Includes `select_agent_for_task()` with keyword heuristics |
| [handoff.py](src/research_agent/agents/handoff.py) | `HandoffRequest` / `HandoffResult` / `HandoffPlan` dataclasses. `build_handoff_plan()`, `aggregate_handoff_results()`, JSONL logging |
| [orchestrator.py](src/research_agent/agents/orchestrator.py) | `run_multi_agent_pipeline()` — ties profiles, handoff, executors, arbitration, tracing, and memory write-back into a single dispatch pipeline |
| [executors.py](src/research_agent/agents/executors.py) | Real specialist agent executors — each `_execute_*_agent()` dispatches to the actual module (paper_reader, claim_support, ppt_progress, report_writer, etc.) |
| [tracing.py](src/research_agent/agents/tracing.py) | `MultiAgentTrace` dataclass. Records orchestrator runs to `data/traces/multi_agent_traces.jsonl`. Provides quality evaluation per run |
| [arbitration.py](src/research_agent/agents/arbitration.py) | Conflict detection across agent outputs: classifies stances (support/oppose/uncertain/neutral/failed), detects contradictions, produces coordinator-level final arbitration text |
| [workspace.py](src/research_agent/agents/workspace.py) | Shared Markdown workspace with section-level permissions. Each agent owns sections; Coordinator has universal write. Unauthorised writes become suggestions. All actions logged to JSONL patch log |

### 8.2 The 9 Specialist Agents

| Agent ID | Display Name | Primary Task Types | Handoff Targets |
|---|---|---|---|
| `coordinator_agent` | Coordinator | general, routing, multi_agent_task, project_planning | All 8 specialists |
| `paper_agent` | Paper Reader | paper_question, paper_reading, related_work, literature_review | claim_agent, report_agent, coordinator_agent |
| `experiment_agent` | Experiment Analyst | experiment_analysis, metric_analysis, result_interpretation | report_agent, progress_agent, claim_agent |
| `claim_agent` | Claim Supporter | claim_support, argument_building, evidence_synthesis | paper_agent, experiment_agent, report_agent |
| `progress_agent` | Progress Tracker | progress_memory, meeting_summary, next_step_planning | experiment_agent, report_agent, coordinator_agent |
| `report_agent` | Report Writer | report_generation, ppt_generation, summary_generation | paper_agent, experiment_agent, progress_agent, claim_agent |
| `code_agent` | Code Assistant | code_question, debugging, development_task | coordinator_agent, memory_agent |
| `memory_agent` | Memory Manager | memory_query, memory_write, memory_consolidation, memory_audit | All agents |
| `general_agent` | General Assistant | general, casual, lightweight_advice | coordinator_agent |

### 8.3 Multi-Agent Orchestration Flow

```
User Query + task_type + RAG docs + memory context
  → orchestrator.run_multi_agent_pipeline()
    → 1. select_agent_for_task() — choose primary specialist
    → 2. build_handoff_plan() — decompose into sub-tasks
    → 3. execute_handoffs() — each specialist processes its task
         (real executors → module dispatch, or simulated fallback)
    → 4. aggregate_handoff_results() — combine into single answer
    → 5. arbitration.arbitrate_results() — detect conflicts, build ruling
    → 6. save results to Memory Store (via adapters)
    → 7. trace_and_evaluate() — record trace + quality score
  → return merged state dict
```

---

## 9. claim/ — Claim Support Retrieval

### 9.1 File

[claim_support.py](src/research_agent/claim/claim_support.py)

### 9.2 Purpose

Given a scientific claim or hypothesis, retrieves structured evidence from the RAG knowledge base, organised into:

1. Claim intent classification (theoretical / empirical / method / related_work)
2. Theoretical support from papers
3. Empirical support from experiments
4. Related work positioning
5. Limitations and counter-evidence
6. Academic wording suggestions
7. Recommended follow-up experiments

Output can be written back to memory as `claim_support` type via [memory/adapters.py](src/research_agent/memory/adapters.py).

---

## 10. paper/ — Paper Reading Pipeline

### 10.1 File

[paper_reader.py](src/research_agent/paper/paper_reader.py)

### 10.2 Purpose

Reads paper Markdown (ingested or hand-written) and produces a structured reading note:

1. Title & basic information
2. Research background
3. Research problem
4. Method overview
5. Experimental setup
6. Key findings
7. Contributions
8. Limitations
9. Relevance to my research
10. PPT outline
11. Suggested follow-up questions

Rule-based section detection for Chinese + English headings. YAML front matter parsing. No LLM dependency in v1 (optional `use_llm` flag reserved).

---

## 11. progress/ — PPT Progress Memory

### 11.1 File

[ppt_progress_memory.py](src/research_agent/progress/ppt_progress_memory.py)

### 11.2 Purpose

Reads slide_doc Markdown (from PPTX ingestion or hand-written slides) and produces a **Research Progress Memory** document:

1. Presentation metadata
2. Slide-by-slide summary
3. Inferred research questions
4. Completed work
5. Experiments and results
6. Issues / limitations
7. Next steps
8. Long-term memory records (tagged bullets)

All heuristic-based keyword matching — no LLM dependency.

---

## 12. report/ — Report Writer

### 12.1 Files

| File | Purpose |
|---|---|
| [report_writer.py](src/research_agent/report/report_writer.py) | Template-based report construction from RAG context. Generates structured group-meeting reports with background, findings, PPT structure, and next steps. Non-LLM fallback |
| [llm_report_writer.py](src/research_agent/report/llm_report_writer.py) | LLM-assisted report generation. Supports multiple report styles. Falls back to template writer on LLM failure |

### 12.2 Data Flow

```
User query ("生成本周组会汇报")
  → RAG retrieval (top_k=4, task_type=general)
  → try LLM Report Writer
  → fallback to Template Report Writer
  → structured markdown report
  → optional memory write-back (report_summary)
```

---

## 13. tools/ — Tool-Augmented Analysis

### 13.1 Files

| File | Purpose |
|---|---|
| [tool_router.py](src/research_agent/tools/tool_router.py) | Extracts file paths from query, maps to tool by extension (.csv → csv_analyzer, .jsonl → jsonl_analyzer), dispatches analysis |
| [csv_analyzer.py](src/research_agent/tools/csv_analyzer.py) | CSV metrics file analysis — column stats, min/max/mean, top-N rows |
| [jsonl_analyzer.py](src/research_agent/tools/jsonl_analyzer.py) | JSONL generation log analysis — key frequency, distribution stats |
| [dataset_tools.py](src/research_agent/tools/dataset_tools.py) | Dataset lookup, recommendation, metadata retrieval |
| [experiment_tools.py](src/research_agent/tools/experiment_tools.py) | Experiment data helpers — run_tag parsing, metric comparison |
| [paper_tools.py](src/research_agent/tools/paper_tools.py) | Paper-related utility functions |
| [code_tools.py](src/research_agent/tools/code_tools.py) | Code reading and explanation utilities |
| [file_utils.py](src/research_agent/tools/file_utils.py) | Shared file I/O utilities |

---

## 14. app.py — Streamlit Web UI

Entry point: `streamlit run app.py`

Features:
- Query input with example buttons
- Final answer display with formatted markdown
- Sidebar showing: `task_type`, `classifier_source`, `tool_used`, `evidence_status`, `evidence_warnings`
- Sources list with clickable paths
- Memory metrics panel: `memory_used`, `memory_count`, `retrieved_memories`, `memory_error`
- Multi-agent panel: `multi_agent_enabled`, `primary_agent`, `handoff_count`, `handoff_summary`, handoff results per agent
- Arbitration display when conflicts detected
- Workspace summary section
- Debug state JSON expander
- Configuration via `.streamlit/config.toml`

---

## 15. run_cli.py — CLI Interface

Entry point: `python run_cli.py`

Features:
- Interactive REPL loop
- `q` / `quit` / `exit` to stop
- Runs `build_graph().invoke()` directly
- Prints `final_answer` including memory and multi-agent sections
- Suitable for rapid testing without the Streamlit UI

---

## 16. Data Directories

### 16.1 Directory Map

| Directory | Content | Git Status |
|---|---|---|
| `data/papers/` | Paper source Markdown files | ✅ Committable (samples) |
| `data/datasets/` | Dataset reference Markdown | ✅ Committable |
| `data/experiments/` | Experiment descriptions + sample CSV/JSONL | ✅ Committable (samples) |
| `data/ingested/` | Ingested Markdown output from raw docs | ❌ Git-ignored |
| `data/memory/` | Memory JSONL + Markdown summary | ❌ JSONL git-ignored; `.gitkeep` tracked |
| `data/memory/backups/` | Consolidation backups | ❌ Git-ignored |
| `data/workspaces/` | Multi-agent workspace drafts | ❌ Git-ignored |
| `data/traces/` | Multi-agent trace logs (JSONL) | ❌ Git-ignored |
| `data/paper_notes/` | Paper reading notes output | ❌ Git-ignored |
| `data/progress_memory/` | Progress memory output | ❌ Git-ignored |
| `raw_docs/` | Raw input documents (PDF, DOCX, PPTX) | ❌ Git-ignored (private) |
| `storage/` | Chroma vector database | ❌ Git-ignored |
| `reports/` | Generated reports output | ❌ Git-ignored |
| `scripts/` | Test & utility scripts | ✅ Committable |
| `docs/` | Project documentation | ✅ Committable |

### 16.2 Git-Ignore Summary

| Category | Patterns |
|---|---|
| Private documents | `raw_docs/`, `raw_data/` |
| Vector store | `storage/`, `chroma_db/`, `*.sqlite` |
| Memory data | `data/memory/*.jsonl`, `data/memory/*.md` (except `.gitkeep`) |
| Agent runtime data | `data/traces/`, `data/workspaces/` |
| Python artifacts | `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.mypy_cache/` |
| Environment | `.env`, `.conda/`, `venv/` |
| OS files | `.DS_Store`, `Thumbs.db` |
| Notebooks | `notebooks/*.ipynb` |

---

## 17. End-to-End Workflows

### 17.1 Paper Reading Workflow

```
PDF / MD file
  → Ingestion (raw_docs/ → data/ingested/*.md)
  → Paper Reading (paper_reader.py: section detection, structured note)
  → Memory Write-back (adapters.py → writer.py → store.py)
  → Report (optional: include in weekly report)
```

### 17.2 Experiment Analysis Workflow

```
User query mentioning CSV/JSONL
  → Task Classification → experiment_analysis
  → RAG Retrieval (experiment_doc filter)
  → Tool Router: extract file path, dispatch CSV/JSONL analyzer
  → Experiment Node: merge RAG context + tool output
  → Memory Store (optional: experiment_result memory)
  → Report Agent (optional: include findings in report)
```

### 17.3 Memory-Aware Answering Workflow

```
User Query
  → Task Classification
  → Task Node (RAG retrieval)
  → Memory Retrieval Node:
      → memory_aware_agent.retrieve_memories_for_query()
      → filters by relevance, level, status
      → format_memory_context()
  → Evidence Check
  → Final Answer (merged: RAG + Memory context + Sources)
```

### 17.4 Multi-Agent Report Workflow

```
User Query ("生成本周组会汇报，包括论文进展、实验结果和下一步计划")
  → Task Classification → report_generation
  → RAG Retrieval (general, top_k=4)
  → Memory Retrieval
  → Multi-Agent Router (ENABLE_MULTI_AGENT=true):
      → Coordinator: select primary = report_agent, build handoff plan
      → Handoff to paper_agent → paper reading + claim support
      → Handoff to experiment_agent → CSV/JSONL analysis
      → Handoff to progress_agent → progress summary
      → Handoff to report_agent → compose final report
      → Arbitration: detect conflicts, coordinator ruling
      → Aggregation: combined answer
      → Trace: record to traces JSONL
      → Memory: write to Memory Store
  → Evidence Check
  → Final Answer (with handoff summary + coordinator arbitration)
```

---

## 18. Module Dependency Summary

| Module | Depends On | Used By | Purpose |
|---|---|---|---|
| `graph/state.py` | *(none)* | `workflow.py`, `nodes.py`, `router.py` | Shared state schema |
| `graph/workflow.py` | `state`, `nodes`, `router` | `app.py`, `run_cli.py` | Build and compile LangGraph app |
| `graph/nodes.py` | `rag/retriever`, `tools/tool_router`, `memory/memory_aware_agent`, `agents/orchestrator`, `report/*` | `workflow.py` | All node implementations |
| `rag/loaders.py` | `rag/schemas` | `rag/indexer`, `rag/hybrid_retriever` | Document loading |
| `rag/indexer.py` | `rag/loaders`, `rag/chunker` | `scripts/build_index.py` | Chroma vector store |
| `rag/retriever.py` | `rag/indexer`, `rag/schemas`, `rag/hybrid_retriever` | `graph/nodes.py`, `claim/*`, `paper/*` | Vector retrieval |
| `rag/hybrid_retriever.py` | `rag/indexer`, `rag/loaders`, `rag/reranker` | `rag/retriever.py` | Hybrid retrieval |
| `ingestion/document_ingestor.py` | `rag/schemas`, `rag/loaders` | `scripts/ingest_documents.py` | Document ingestion |
| `memory/schema.py` | *(none)* | `memory/store`, `memory/writer`, `memory/retriever`, `memory/privacy_scope` | Memory data model |
| `memory/store.py` | `memory/schema` | `memory/writer`, `memory/retriever`, `memory/consolidation` | Persistent storage |
| `memory/writer.py` | `memory/schema`, `memory/store`, `memory/privacy_scope` | `memory/adapters`, write-back paths | Memory creation |
| `memory/retriever.py` | `memory/store` | `memory/memory_aware_agent`, `agents/executors` | Memory search |
| `memory/memory_aware_agent.py` | `memory/retriever`, `memory/consolidation` | `graph/nodes.py` | Query-time augmentation |
| `memory/adapters.py` | `memory/writer` | `claim/*`, `paper/*`, `progress/*`, `report/*`, `agents/orchestrator` | Module → Memory bridge |
| `memory/privacy_scope.py` | `memory/store` | `memory/writer`, `memory/retriever` | Access control |
| `memory/consolidation.py` | `memory/store` | `scripts/run_memory_consolidation.py` | Memory maintenance |
| `agents/profiles.py` | *(none)* | `agents/handoff`, `agents/orchestrator`, `agents/executors` | Agent definitions |
| `agents/handoff.py` | `agents/profiles` | `agents/orchestrator`, `agents/executors` | Handoff structures |
| `agents/orchestrator.py` | `agents/profiles`, `agents/handoff`, `agents/executors`, `agents/arbitration`, `agents/tracing`, `memory/adapters`, `memory/memory_aware_agent` | `graph/nodes.py` | Multi-agent pipeline |
| `agents/executors.py` | `agents/handoff`, `claim/*`, `paper/*`, `progress/*`, `report/*`, `tools/*` | `agents/orchestrator.py` | Real agent dispatch |
| `agents/arbitration.py` | *(none)* | `agents/orchestrator.py` | Conflict resolution |
| `agents/tracing.py` | *(none)* | `agents/orchestrator.py` | Trace recording |
| `agents/workspace.py` | *(none)* | `app.py`, `agents/orchestrator.py` | Shared workspace |
| `claim/claim_support.py` | `rag/retriever` | `agents/executors`, standalone use | Claim support |
| `paper/paper_reader.py` | *(standalone)* | `agents/executors`, standalone use | Paper reading |
| `progress/ppt_progress_memory.py` | *(standalone)* | `agents/executors`, standalone use | Progress memory |
| `report/report_writer.py` | *(standalone)* | `graph/nodes.py`, `agents/executors` | Report generation |
| `report/llm_report_writer.py` | `llm/client` | `graph/nodes.py` | LLM report |
| `tools/tool_router.py` | `tools/csv_analyzer`, `tools/jsonl_analyzer` | `graph/nodes.py` | Tool dispatch |
| `llm/client.py` | *(config)* | `llm/enhancers`, `graph/llm_classifier`, `report/llm_report_writer` | LLM API |
| `eval/benchmark.py` | `graph/workflow` | `scripts/run_benchmark.py` | Evaluation |
| `eval/metrics.py` | *(none)* | `eval/benchmark.py` | Quality metrics |

---

## 19. Current Development Stage

### v0.5 — Memory + Multi-Agent Preview

**✅ Completed:**
- LangGraph workflow with task classification (LLM + rule fallback)
- RAG pipeline: document loading, chunking, Chroma indexing, vector retrieval
- Hybrid retrieval: vector + keyword score fusion
- Heuristic reranking
- Document ingestion (PDF, DOCX, PPTX, MD) with optional MinerU API
- Tool-augmented experiment analysis (CSV, JSONL)
- Evidence checking (sources + tool results)
- Memory system: schema, JSONL store, rule-based writer, multi-dimensional retriever
- Memory privacy scoping (private / shared / global)
- Memory consolidation (merge, compress, expire, stage summary)
- Memory-aware agent (query-time memory retrieval)
- 9 specialist agent profiles with keyword-based selection
- Multi-agent handoff: plan building, task decomposition, result aggregation
- Real specialist executors (delegating to actual modules)
- Conflict detection & coordinator arbitration (stance classification)
- Shared workspace with section-level permissions
- Multi-agent tracing & quality evaluation
- Report writer (template + LLM-assisted)
- Claim support retrieval
- Paper reading pipeline
- PPT progress memory extraction
- Streamlit Web UI with debug state, memory metrics, multi-agent panels
- CLI interface
- 40+ test scripts per module

**🚧 In Progress / Planned:**
- Cross-Encoder reranker (replacing heuristic reranker)
- Chroma-based memory collection (vector search over memory records)
- Docker packaging for easier deployment
- More robust evaluation dashboard (metrics over time)
- Production concurrency control for multi-agent
- Agent-specific LLM prompting (each agent gets tailored system prompts)
- Persistent agent sessions across CLI/UI restarts
- Memory retrieval with semantic similarity (embedding-based)

---

## 20. Output Summary

| Item | Status |
|---|---|
| New file created | `docs/project_structure_and_modules.md` |
| Chapters | 20 (Overview, Architecture, Directory, Graph, RAG, Ingestion, Memory, Agents, Claim, Paper, Progress, Report, Tools, UI, CLI, Data, Workflows, Dependencies, Dev Stage, Summary) |
| Covers agents | ✅ All 7 module files + 9 agent profiles |
| Covers memory | ✅ All 8 module files (schema, store, writer, retriever, privacy, consolidation, adapters, memory_aware_agent) |
| Covers RAG | ✅ All 7 module files (loaders, schemas, indexer, retriever, hybrid_retriever, reranker, chunker) |
| Covers ingestion | ✅ Both files (document_ingestor, mineru_client) |
| Covers graph | ✅ All 6 files (state, nodes, router, workflow, evidence_checker, llm_classifier) |
| Covers UI | ✅ app.py (Streamlit) |
| Covers CLI | ✅ run_cli.py |
| Data directories & gitignore | ✅ Full table with git status per directory |
| Sensitive info exposed | ❌ None (no API keys, no private data paths, no real content) |
| Recommended git command | `git add docs/project_structure_and_modules.md` |
| Recommended commit message | `docs: add project structure and module guide` |
