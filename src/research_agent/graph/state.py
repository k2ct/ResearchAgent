'''
rom typing import TypedDict


class AgentState(TypedDict):
    query: str
    task_type: str
    result: str
    final_answer: str

    # Day 3 新增：记录分类来源，方便调试
    classifier_source: str
    route_reason: str
'''

from typing import TypedDict, List, Dict


class AgentState(TypedDict):
    query: str
    task_type: str
    result: str
    final_answer: str

    classifier_source: str
    route_reason: str

    # Day 9 新增：RAG 检索结果
    retrieved_docs: List[Dict]
    sources: List[Dict]