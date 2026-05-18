from typing import TypedDict


class AgentState(TypedDict):
    query: str
    task_type: str
    result: str
    final_answer: str

    # Day 3 新增：记录分类来源，方便调试
    classifier_source: str
    route_reason: str