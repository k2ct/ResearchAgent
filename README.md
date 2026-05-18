# ResearchAgent v0.1

ResearchAgent is a minimal LangGraph-based research assistant demo.

The current version focuses on building a clean and extensible Agent workflow skeleton. It supports task classification, conditional routing, and mock task nodes for different research scenarios.

## Features

- LangGraph-based workflow
- Shared AgentState
- Rule-based task classification
- Optional LLM-based task classification
- Fallback from LLM classifier to rule classifier
- Conditional routing to different task nodes
- CLI demo

## Supported Task Types

| Task Type | Description |
|---|---|
| `paper_question` | Paper Q&A, literature summary, paper comparison |
| `experiment_analysis` | Experiment result analysis, hallucination metric interpretation |
| `dataset_recommendation` | Dataset recommendation and dataset selection |
| `report_generation` | Group meeting report, PPT text, progress summary |
| `code_help` | Code explanation, bug fixing, environment issues |
| `general` | General research assistant task |

## Project Structure

```text
ResearchAgent
├── run_cli.py
├── requirements.txt
├── .env.example
└── src
    └── research_agent
        └── graph
            ├── state.py
            ├── nodes.py
            ├── router.py
            ├── workflow.py
            └── llm_classifier.**py**