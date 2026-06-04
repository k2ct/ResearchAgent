
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
)
from .router import route_task


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

    # Each task node → memory retrieval → evidence check → final answer
    workflow.add_edge("paper_node", "retrieve_memory")
    workflow.add_edge("experiment_node", "retrieve_memory")
    workflow.add_edge("dataset_node", "retrieve_memory")
    workflow.add_edge("report_node", "retrieve_memory")
    workflow.add_edge("code_node", "retrieve_memory")
    workflow.add_edge("general_node", "retrieve_memory")

    workflow.add_edge("retrieve_memory", "evidence_check")
    workflow.add_edge("evidence_check", "final_answer")
    workflow.add_edge("final_answer", END)

    return workflow.compile()