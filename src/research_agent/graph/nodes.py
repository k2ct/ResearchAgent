from .state import AgentState

from .llm_classifier import classify_with_llm, get_llm_classifier_enabled

from research_agent.rag.retriever import (
    retrieve_documents,
    document_to_dict,
    extract_sources_from_docs,
    format_retrieved_docs,
)

'''
def classify_task(state: AgentState) -> dict:
    """
    根据用户输入判断任务类型。
    Day 2 暂时继续使用规则分类，不接 LLM。
    """
    query = state["query"]

    if "论文" in query or "paper" in query.lower():
        task_type = "paper_question"
    elif "实验" in query or "coco" in query.lower() or "幻觉" in query:
        task_type = "experiment_analysis"
    elif "数据集" in query or "dataset" in query.lower():
        task_type = "dataset_recommendation"
    elif "代码" in query or "脚本" in query or "bug" in query.lower():
        task_type = "code_help"
    elif "汇报" in query or "PPT" in query.upper() or "组会" in query:
        task_type = "report_generation"
    else:
        task_type = "general"

    return {
        "task_type": task_type
    }
'''


def classify_task_by_rule(query: str) -> dict:
    """
    规则分类器：作为 LLM 分类失败后的兜底。
    """
    if "论文" in query or "paper" in query.lower():
        task_type = "paper_question"
        reason = "命中了论文 / paper 关键词。"
    elif "实验" in query or "coco" in query.lower() or "幻觉" in query or "benchmark" in query.lower():
        task_type = "experiment_analysis"
        reason = "命中了实验、COCO、幻觉或 benchmark 相关关键词。"
    elif "数据集" in query or "dataset" in query.lower():
        task_type = "dataset_recommendation"
        reason = "命中了数据集 / dataset 关键词。"
    elif "汇报" in query or "PPT" in query.upper() or "组会" in query or "总结" in query:
        task_type = "report_generation"
        reason = "命中了汇报、PPT、组会或总结相关关键词。"
    elif "代码" in query or "脚本" in query or "bug" in query.lower() or "报错" in query or "环境" in query:
        task_type = "code_help"
        reason = "命中了代码、脚本、bug、报错或环境相关关键词。"
    else:
        task_type = "general"
        reason = "未命中明确任务关键词，归为通用科研助手任务。"

    return {
        "task_type": task_type,
        "reason": reason,
    }


def classify_task(state: AgentState) -> dict:
    """
    Day 3 分类节点：
    优先使用 LLM 分类；
    如果关闭 LLM 或 LLM 失败，则回退到规则分类。
    """
    query = state["query"]

    if get_llm_classifier_enabled():
        try:
            llm_result = classify_with_llm(query)
            return {
                "task_type": llm_result["task_type"],
                "classifier_source": "llm",
                "route_reason": llm_result["reason"],
            }
        except Exception as e:
            rule_result = classify_task_by_rule(query)
            return {
                "task_type": rule_result["task_type"],
                "classifier_source": "rule_fallback",
                "route_reason": f"LLM 分类失败，回退规则分类。错误：{str(e)}；规则原因：{rule_result['reason']}",
            }

    rule_result = classify_task_by_rule(query)
    return {
        "task_type": rule_result["task_type"],
        "classifier_source": "rule",
        "route_reason": rule_result["reason"],
    }


def build_simple_answer_from_context(query: str, retrieved_context: str) -> str:
    """
    v0.2 最小回答生成器。

    当前版本不调用 LLM，只把检索上下文整理成可读回答。
    后续可以替换为 LLM-based answer generator。
    """
    if not retrieved_context or retrieved_context == "未检索到相关资料。":
        return "暂未在本地科研资料库中检索到足够相关的资料。"

    return f"""
用户问题：{query}

检索到的相关资料摘要如下：
{retrieved_context}

说明：当前 v0.2 版本暂时使用检索内容摘要作为回答，后续可以接入 LLM 对 retrieved_docs 进行综合生成。
""".strip()


# 以下函数一为新增
def run_rag_for_task(state: AgentState, task_type: str, top_k: int = 3) -> dict:
    """
    根据 query 和 task_type 调用 Retriever，
    返回 retrieved_docs、sources 和格式化后的 retrieved_context。
    """
    query = state["query"]

    docs = retrieve_documents(
        query=query,
        task_type=task_type,
        top_k=top_k,
        use_filter=True,
    )

    retrieved_docs = [document_to_dict(doc) for doc in docs]
    sources = extract_sources_from_docs(docs)
    retrieved_context = format_retrieved_docs(docs, max_chars_per_doc=500)

    return {
        "retrieved_docs": retrieved_docs,
        "sources": sources,
        "retrieved_context": retrieved_context,
    }


'''
def paper_node(state: AgentState) -> dict:
    return {
        "result": "这是论文问答任务，后续会接入论文 RAG。"
    }
'''
def paper_node(state: AgentState) -> dict:
    rag_result = run_rag_for_task(
        state=state,
        task_type="paper_question",
        top_k=3,
    )

    result = f"""
这是论文问答任务。系统已从论文笔记资料库中检索到相关内容。

基于检索资料的初步回答：
{build_simple_answer_from_context(state["query"], rag_result["retrieved_context"])}
""".strip()

    return {
        "retrieved_docs": rag_result["retrieved_docs"],
        "sources": rag_result["sources"],
        "result": result,
    }


'''
def experiment_node(state: AgentState) -> dict:
    return {
        "result": "这是实验分析任务，后续会接入 CSV / JSONL 分析工具。"
    }
'''
def experiment_node(state: AgentState) -> dict:
    rag_result = run_rag_for_task(
        state=state,
        task_type="experiment_analysis",
        top_k=3,
    )

    result = f"""
这是实验分析任务。系统已从实验说明资料库中检索到相关内容。

基于检索资料的初步回答：
{build_simple_answer_from_context(state["query"], rag_result["retrieved_context"])}
""".strip()

    return {
        "retrieved_docs": rag_result["retrieved_docs"],
        "sources": rag_result["sources"],
        "result": result,
    }


'''
def dataset_node(state: AgentState) -> dict:
    return {
        "result": "这是数据集推荐任务，后续会接入数据集资料库。"
    }
'''
def dataset_node(state: AgentState) -> dict:
    rag_result = run_rag_for_task(
        state=state,
        task_type="dataset_recommendation",
        top_k=3,
    )

    result = f"""
这是数据集推荐 / 数据集说明任务。系统已从数据集资料库中检索到相关内容。

基于检索资料的初步回答：
{build_simple_answer_from_context(state["query"], rag_result["retrieved_context"])}
""".strip()

    return {
        "retrieved_docs": rag_result["retrieved_docs"],
        "sources": rag_result["sources"],
        "result": result,
    }


'''
def code_node(state: AgentState) -> dict:
    return {
        "result": "这是代码辅助任务，后续会接入代码解释与修改工具。"
    }
'''
def code_node(state: AgentState) -> dict:
    return {
        "retrieved_docs": [],
        "sources": [],
        "result": "这是代码辅助任务，后续会接入代码解释与修改工具。"
    }


'''
def general_node(state: AgentState) -> dict:
    return {
        "result": "这是通用科研助手任务。"
    }
'''
def general_node(state: AgentState) -> dict:
    return {
        "retrieved_docs": [],
        "sources": [],
        "result": "这是通用科研助手任务。"
    }


'''
1
def final_answer_node(state: AgentState) -> dict:
    final_answer = f"""
任务类型：{state["task_type"]}
处理结果：{state["result"]}
""".strip()

    return {
        "final_answer": final_answer
    }
'''

'''
2
def final_answer_node(state: AgentState) -> dict:
    final_answer = f"""
任务类型：{state["task_type"]}
分类来源：{state["classifier_source"]}
分类原因：{state["route_reason"]}
处理结果：{state["result"]}
""".strip()

    return {
        "final_answer": final_answer
    }
'''
def format_sources(sources: list) -> str:
    """
    格式化 Sources。
    """
    if not sources:
        return "无"

    lines = []

    for i, source in enumerate(sources, start=1):
        path = source.get("path", "unknown")
        source_type = source.get("source_type", "unknown")
        title = source.get("title", "")
        dataset = source.get("dataset", "")
        run_tag = source.get("run_tag", "")

        extra_parts = []

        if title:
            extra_parts.append(f"title={title}")

        if dataset:
            extra_parts.append(f"dataset={dataset}")

        if run_tag:
            extra_parts.append(f"run_tag={run_tag}")

        extra = f" ({', '.join(extra_parts)})" if extra_parts else ""

        lines.append(f"{i}. [{source_type}] {path}{extra}")

    return "\n".join(lines)


def final_answer_node(state: AgentState) -> dict:
    final_answer = f"""
任务类型：{state["task_type"]}
分类来源：{state["classifier_source"]}
分类原因：{state["route_reason"]}

回答：
{state["result"]}

Sources:
{format_sources(state.get("sources", []))}
""".strip()

    return {
        "final_answer": final_answer
    }


'''
def report_node(state: AgentState) -> dict:
    return {
        "result": "这是报告生成任务，后续会接入自动 PPT 生成工具。"
    }
'''
def report_node(state: AgentState) -> dict:
    return {
        "retrieved_docs": [],
        "sources": [],
        "result": "这是汇报生成任务，后续会接入 Report Writer。"
    }