import os

from .state import AgentState
from research_agent.report.report_writer import build_report_from_context

from .llm_classifier import classify_with_llm, get_llm_classifier_enabled

from research_agent.rag.retriever import (
    retrieve_documents,
    document_to_dict,
    extract_sources_from_docs,
    format_retrieved_docs,
)

from research_agent.tools.tool_router import run_tool_from_query

from research_agent.report.llm_report_writer import generate_report_with_llm

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
    query_lower = query.lower()
    query_upper = query.upper()

    if (
        "汇报" in query
        or "组会" in query
        or "文案" in query
        or "草稿" in query
        or "报告" in query
        or "PPT" in query_upper
        or "presentation" in query_lower
        or "slide" in query_lower
    ):
        task_type = "report_generation"
        reason = "命中了汇报、PPT、组会、文案、草稿、报告或 presentation 相关关键词。"
    elif "论文" in query or "paper" in query_lower or "related work" in query_lower:
        task_type = "paper_question"
        reason = "命中了论文、paper 或 related work 相关关键词。"
    #elif "实验" in query or "coco" in query_lower or "幻觉" in query or "benchmark" in query_lower:
    #    task_type = "experiment_analysis"
    #    reason = "命中了实验、COCO、幻觉或 benchmark 相关关键词。"
    elif (
        "实验" in query
        or "coco_val" in query_lower
        or "run_tag" in query_lower
        or "幻觉" in query
        or "benchmark" in query_lower
        or ".csv" in query_lower
        or ".jsonl" in query_lower
    ):
        task_type = "experiment_analysis"
        reason = "命中了实验、文件路径、CSV/JSONL、run_tag、coco_val、幻觉或 benchmark 相关关键词。"

    #elif "数据集" in query or "dataset" in query_lower:
    #    task_type = "dataset_recommendation"
    #    reason = "命中了数据集 / dataset 关键词。"
    elif (
        "数据集" in query
        or "dataset" in query_lower
        or "OpenImages" in query
        or "MIAP" in query
        or "FairFace" in query
        or "GQA" in query
        or "COCO" in query_upper
    ):
        task_type = "dataset_recommendation"
        reason = "命中了数据集、dataset、OpenImages、MIAP、FairFace、GQA 或 COCO 相关关键词。"
    elif (
        "代码" in query
        or "脚本" in query
        or "bug" in query_lower
        or "modulenotfounderror" in query_lower
        or "no module named" in query_lower
        or "报错" in query
        or "环境" in query
    ):
        task_type = "code_help"
        reason = "命中了代码、脚本、bug、ModuleNotFoundError、报错或环境相关关键词。"
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
    模板回答生成器（非 LLM fallback）。

    当前版本在未启用 LLM 时使用检索内容摘要作为回答。
    启用 LLM 后会自动切换为 LLM-based 综合生成。
    """
    if not retrieved_context or retrieved_context == "未检索到相关资料。":
        return "暂未在本地科研资料库中检索到足够相关的资料。"

    return f"""
用户问题：{query}

检索到的相关资料摘要如下：
{retrieved_context}

说明：当前版本在未启用 LLM 时使用检索内容摘要作为回答，启用 LLM 后会自动切换为 LLM-based 综合生成。
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



def run_optional_tool_for_experiment(state: AgentState) -> dict:
    """
    对 experiment_analysis 问题尝试调用本地文件分析工具。

    如果用户 query 中没有 CSV / JSONL 文件路径，则不调用工具。
    """
    query = state["query"]
    tool_output = run_tool_from_query(query)

    return {
        "tool_used": tool_output.get("tool_used", "none"),
        "tool_result": tool_output.get("analysis", {}),
        "tool_result_text": tool_output.get("formatted_text", ""),
    }


'''
def experiment_node(state: AgentState) -> dict:
    return {
        "result": "这是实验分析任务，后续会接入 CSV / JSONL 分析工具。"
    }
'''

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

def experiment_node(state: AgentState) -> dict:
    # 1. 先做 RAG 检索
    rag_result = run_rag_for_task(
        state=state,
        task_type="experiment_analysis",
        top_k=3,
    )

    # 2. 再尝试调用 CSV / JSONL 工具
    tool_state = run_optional_tool_for_experiment(state)

    tool_used = tool_state["tool_used"]
    tool_text = tool_state["tool_result_text"]

    if tool_used != "none" and tool_text:
        tool_section = f"""
本地实验文件分析结果：
{tool_text}
""".strip()
    else:
        tool_section = "本次问题未检测到 CSV / JSONL 文件路径，因此未调用本地文件分析工具。"

    result = f"""
这是实验分析任务。系统已完成 RAG 检索，并根据需要尝试调用本地实验文件分析工具。

一、工具分析
{tool_section}

二、相关实验资料检索
{build_simple_answer_from_context(state["query"], rag_result["retrieved_context"])}
""".strip()

    return {
        "retrieved_docs": rag_result["retrieved_docs"],
        "sources": rag_result["sources"],
        "tool_used": tool_state["tool_used"],
        "tool_result": tool_state["tool_result"],
        "tool_result_text": tool_state["tool_result_text"],
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


def evidence_check_node(state: AgentState) -> dict:
    """
    Day 13 最小证据检查节点。

    目标：
    1. 检查是否有 RAG sources
    2. 检查是否有成功的 tool result
    3. 对缺少证据的回答给出 warning
    """
    sources = state.get("sources", [])
    retrieved_docs = state.get("retrieved_docs", [])

    tool_used = state.get("tool_used", "none")
    tool_result = state.get("tool_result", {})
    tool_ok = bool(tool_result.get("ok")) if isinstance(tool_result, dict) else False

    warnings = []

    has_sources = len(sources) > 0
    has_retrieved_docs = len(retrieved_docs) > 0
    has_successful_tool = tool_used != "none" and tool_ok

    if has_sources and has_successful_tool:
        evidence_status = "passed"
        evidence_reason = "回答同时包含 RAG 检索来源和成功的工具分析结果。"

    elif has_sources:
        evidence_status = "passed"
        evidence_reason = "回答包含 RAG 检索来源，可追踪到本地资料库。"

    elif has_successful_tool:
        evidence_status = "passed"
        evidence_reason = "回答包含成功的本地工具分析结果。"

    elif has_retrieved_docs:
        evidence_status = "weak"
        evidence_reason = "回答包含检索片段，但未提取到明确 Sources。"
        warnings.append("检索到文档片段，但 Sources 为空，请检查 metadata path 是否正常。")

    else:
        evidence_status = "weak"
        evidence_reason = "当前回答没有 RAG Sources 或工具结果支撑，属于弱证据回答。"
        warnings.append("未检索到资料来源。")
        warnings.append("未调用成功的本地分析工具。")

    # 如果工具被调用但失败，额外提示
    if tool_used != "none" and not tool_ok:
        warnings.append(f"工具 {tool_used} 被调用，但分析未成功。")

    return {
        "evidence_status": evidence_status,
        "evidence_reason": evidence_reason,
        "evidence_warnings": warnings,
    }

'''
def final_answer_node(state: AgentState) -> dict:
    final_answer = f"""
任务类型：{state["task_type"]}
分类来源：{state["classifier_source"]}
分类原因：{state["route_reason"]}
工具调用：{state.get("tool_used", "none")}

回答：
{state["result"]}

Sources:
{format_sources(state.get("sources", []))}
""".strip()

    return {
        "final_answer": final_answer
    }
'''
def format_warnings(warnings: list) -> str:
    if not warnings:
        return "无"

    return "\n".join(
        f"- {warning}"
        for warning in warnings
    )


def _is_memory_aware_enabled() -> bool:
    """Check whether memory-aware retrieval is enabled via env var."""
    from dotenv import load_dotenv
    load_dotenv()
    value = os.getenv("ENABLE_MEMORY_AWARE_AGENT", "false").lower()
    return value in ("1", "true", "yes", "y")


def retrieve_memory_node(state: AgentState) -> dict:
    """
    Retrieve relevant memories from the Memory Store for the current query.

    Safely no-ops when ENABLE_MEMORY_AWARE_AGENT is false or the Memory
    system is unavailable.  Never raises — failures return empty memory.
    """
    query = state.get("query", "")
    task_type = state.get("task_type", "general")

    if not _is_memory_aware_enabled():
        return {
            "memory_context": "",
            "retrieved_memories": [],
            "memory_count": 0,
            "memory_used": False,
            "memory_error": "Memory-aware agent disabled",
        }

    try:
        from research_agent.memory.memory_aware_agent import (
            retrieve_memories_for_query,
            format_memory_context,
        )

        top_k = int(os.getenv("MEMORY_TOP_K", "5"))
        include_short = os.getenv("MEMORY_INCLUDE_SHORT_TERM", "true").lower() in ("1", "true", "yes", "y")
        include_expired = os.getenv("MEMORY_INCLUDE_EXPIRED", "false").lower() in ("1", "true", "yes", "y")

        memories = retrieve_memories_for_query(
            query=query,
            task_type=task_type,
            max_results=top_k,
            include_short_term=include_short,
            include_expired=include_expired,
        )

        memory_context = format_memory_context(memories)

        return {
            "retrieved_memories": memories,
            "memory_context": memory_context,
            "memory_count": len(memories),
            "memory_used": len(memories) > 0,
            "memory_error": "",
        }

    except Exception as e:
        return {
            "memory_context": "",
            "retrieved_memories": [],
            "memory_count": 0,
            "memory_used": False,
            "memory_error": f"{type(e).__name__}: {e}",
        }


def _merge_memory_into_answer(answer: str, state: AgentState) -> str:
    """Append memory context to the final answer if available."""
    memory_context = state.get("memory_context", "")
    memory_count = state.get("memory_count", 0)

    if not memory_context or memory_count == 0:
        return answer

    memory_section = f"""
记忆检索：已使用 ({memory_count} 条)
记忆上下文：
{memory_context}
""".strip()

    return f"{answer}\n\n{memory_section}"


def format_memory_debug(state: AgentState) -> str:
    """Format memory debug info for CLI / Web UI display."""
    memory_used = state.get("memory_used", False)
    memory_count = state.get("memory_count", 0)
    memory_error = state.get("memory_error", "")
    memories = state.get("retrieved_memories", [])

    if not memory_used and not memory_error:
        return "Memory Used: False"

    lines = [
        f"Memory Used: {memory_used}",
        f"Memory Count: {memory_count}",
    ]

    if memory_error:
        lines.append(f"Memory Error: {memory_error}")

    if memories:
        lines.append("Memory IDs:")
        for m in memories[:5]:
            mid = m.get("memory_id", "?") if isinstance(m, dict) else "?"
            mtype = m.get("memory_type", "?") if isinstance(m, dict) else "?"
            summary = (m.get("summary", "") or "") if isinstance(m, dict) else ""
            lines.append(f"- {mid} ({mtype}) {summary[:80]}")

    return "\n".join(lines)


def final_answer_node(state: AgentState) -> dict:
    # Merge memory context into answer if available
    result = state.get("result", "")
    result = _merge_memory_into_answer(result, state)

    memory_debug = format_memory_debug(state)

    final_answer = f"""
任务类型：{state["task_type"]}
分类来源：{state["classifier_source"]}
分类原因：{state["route_reason"]}
工具调用：{state.get("tool_used", "none")}

证据检查：{state.get("evidence_status", "unknown")}
证据说明：{state.get("evidence_reason", "")}
证据警告：
{format_warnings(state.get("evidence_warnings", []))}

{memory_debug}

回答：
{result}

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
    """
    Day20: LLM-assisted Report Writer.

    逻辑：
    1. 对 report_generation 问题做全库 RAG 检索；
    2. 优先尝试 LLM Report Writer；
    3. 如果 LLM 不可用或失败，则 fallback 到模板版 Report Writer。
    """
    rag_result = run_rag_for_task(
        state=state,
        task_type="general",
        top_k=4,
    )

    retrieved_docs = rag_result["retrieved_docs"]

    llm_result = generate_report_with_llm(
        query=state["query"],
        retrieved_docs=retrieved_docs,
        tool_result_text=state.get("tool_result_text", ""),
    )

    if llm_result.get("ok"):
        report_text = llm_result["report_text"]
        report_method = f"LLM-assisted Report Writer ({llm_result.get('report_style')})"
    else:
        report_text = build_report_from_context(
            query=state["query"],
            retrieved_docs=retrieved_docs,
        )
        report_method = (
            "Template fallback Report Writer "
            f"(LLM unavailable: {llm_result.get('error')})"
        )

    result = f"""
这是汇报生成任务。系统已根据本地科研资料库生成结构化汇报草稿。

生成方式：
{report_method}

{report_text}
""".strip()

    return {
        "retrieved_docs": rag_result["retrieved_docs"],
        "sources": rag_result["sources"],
        "tool_used": "none",
        "tool_result": {},
        "tool_result_text": "",
        "result": result,
    }