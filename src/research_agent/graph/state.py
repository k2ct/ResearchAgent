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

from typing import TypedDict, List, Dict, Any


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

    # Day 12 新增：工具调用结果
    tool_used: str
    tool_result: Dict[str, Any]
    tool_result_text: str

    # Day 13 新增：证据检查结果
    evidence_status: str
    evidence_reason: str
    evidence_warnings: List[str]