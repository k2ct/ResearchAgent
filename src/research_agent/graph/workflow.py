"""
LangGraph workflow for ResearchAgent.

Supports two modes:
- **Single-Agent** (default): classify → task_node → memory → evidence → answer
- **Multi-Agent** (ENABLE_MULTI_AGENT=true): adds multi_agent_router_node
  between memory retrieval and evidence check.
"""

import os

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import (
    classify_task,
    paper_node,
    experiment_node,
    dataset_node,
    code_node,
    general_node,
    evidence_check_node,
    final_answer_node,
    report_node,
    retrieve_memory_node,
    multi_agent_router_node,
)
from .router import route_task


def _multi_agent_enabled() -> bool:
    return os.getenv("ENABLE_MULTI_AGENT", "false").strip().lower() == "true"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_task", classify_task)

    workflow.add_node("paper_node", paper_node)
    workflow.add_node("experiment_node", experiment_node)
    workflow.add_node("dataset_node", dataset_node)
    workflow.add_node("code_node", code_node)
    workflow.add_node("general_node", general_node)
    workflow.add_node("report_node", report_node)
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("evidence_check", evidence_check_node)
    workflow.add_node("final_answer", final_answer_node)

    workflow.add_edge(START, "classify_task")

    workflow.add_conditional_edges(
        "classify_task",
        route_task,
        {
            "paper_node": "paper_node",
            "experiment_node": "experiment_node",
            "dataset_node": "dataset_node",
            "code_node": "code_node",
            "general_node": "general_node",
            "report_node": "report_node",
        },
    )

    # Each task node → memory retrieval
    for node_name in ["paper_node", "experiment_node", "dataset_node",
                       "report_node", "code_node", "general_node"]:
        workflow.add_edge(node_name, "retrieve_memory")

    # ── Conditional: multi-agent router ────────────────────────────
    if _multi_agent_enabled():
        workflow.add_node("multi_agent_router", multi_agent_router_node)
        workflow.add_edge("retrieve_memory", "multi_agent_router")
        workflow.add_edge("multi_agent_router", "evidence_check")
    else:
        workflow.add_edge("retrieve_memory", "evidence_check")

    workflow.add_edge("evidence_check", "final_answer")
    workflow.add_edge("final_answer", END)

    return workflow.compile()
