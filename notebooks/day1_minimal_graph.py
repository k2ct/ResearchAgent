from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    query: str
    task_type: str
    result: str
    final_answer: str


def classify_task(state: AgentState) -> dict:
    query = state["query"]

    if "论文" in query or "paper" in query.lower():
        task_type = "paper_question"
    elif "实验" in query or "coco" in query.lower() or "幻觉" in query:
        task_type = "experiment_analysis"
    elif "数据集" in query or "dataset" in query.lower():
        task_type = "dataset_recommendation"
    elif "代码" in query or "脚本" in query or "bug" in query.lower():
        task_type = "code_help"
    else:
        task_type = "general"

    return {
        "task_type": task_type
    }


def route_task(state: AgentState) -> Literal[
    "paper_node",
    "experiment_node",
    "dataset_node",
    "code_node",
    "general_node",
]:
    task_type = state["task_type"]

    if task_type == "paper_question":
        return "paper_node"
    elif task_type == "experiment_analysis":
        return "experiment_node"
    elif task_type == "dataset_recommendation":
        return "dataset_node"
    elif task_type == "code_help":
        return "code_node"
    else:
        return "general_node"


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


def final_answer_node(state: AgentState) -> dict:
    final_answer = f"""
任务类型：{state["task_type"]}
处理结果：{state["result"]}
""".strip()

    return {
        "final_answer": final_answer
    }


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_task", classify_task)
    workflow.add_node("paper_node", paper_node)
    workflow.add_node("experiment_node", experiment_node)
    workflow.add_node("dataset_node", dataset_node)
    workflow.add_node("code_node", code_node)
    workflow.add_node("general_node", general_node)
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
        },
    )

    workflow.add_edge("paper_node", "final_answer")
    workflow.add_edge("experiment_node", "final_answer")
    workflow.add_edge("dataset_node", "final_answer")
    workflow.add_edge("code_node", "final_answer")
    workflow.add_edge("general_node", "final_answer")

    workflow.add_edge("final_answer", END)

    return workflow.compile()


def run_agent(query: str):
    graph = build_graph()

    result = graph.invoke({
        "query": query,
        "task_type": "",
        "result": "",
        "final_answer": "",
    })

    return result


if __name__ == "__main__":
    while True:
        query = input("\n请输入你的科研问题，输入 q 退出：\n> ")

        if query.lower() in ["q", "quit", "exit"]:
            print("已退出。")
            break

        result = run_agent(query)

        print("\n===== Agent 输出 =====")
        print(result["final_answer"])