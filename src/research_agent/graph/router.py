from typing import Literal

from .state import AgentState


def route_task(state: AgentState) -> Literal[
    "paper_node",
    "experiment_node",
    "dataset_node",
    "code_node",
    "general_node",
    "report_node",
]:
    """
    根据 task_type 决定下一步进入哪个节点。
    """
    task_type = state["task_type"]

    if task_type == "paper_question":
        return "paper_node"
    elif task_type == "experiment_analysis":
        return "experiment_node"
    elif task_type == "dataset_recommendation":
        return "dataset_node"
    elif task_type == "code_help":
        return "code_node"
    elif task_type == "report_generation":
        return "report_node"
    else:
        return "general_node"
    