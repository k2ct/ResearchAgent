from .state import AgentState

from .llm_classifier import classify_with_llm, get_llm_classifier_enabled

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



def paper_node(state: AgentState) -> dict:
    return {
        "result": "这是论文问答任务，后续会接入论文 RAG。"
    }


def experiment_node(state: AgentState) -> dict:
    return {
        "result": "这是实验分析任务，后续会接入 CSV / JSONL 分析工具。"
    }


def dataset_node(state: AgentState) -> dict:
    return {
        "result": "这是数据集推荐任务，后续会接入数据集资料库。"
    }


def code_node(state: AgentState) -> dict:
    return {
        "result": "这是代码辅助任务，后续会接入代码解释与修改工具。"
    }


def general_node(state: AgentState) -> dict:
    return {
        "result": "这是通用科研助手任务。"
    }

'''
def final_answer_node(state: AgentState) -> dict:
    final_answer = f"""
任务类型：{state["task_type"]}
处理结果：{state["result"]}
""".strip()

    return {
        "final_answer": final_answer
    }
'''


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
    

def report_node(state: AgentState) -> dict:
    return {
        "result": "这是报告生成任务，后续会接入自动 PPT 生成工具。"
    }
