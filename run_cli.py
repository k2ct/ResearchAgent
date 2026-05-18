'''
from src.research_agent.graph.workflow import build_graph


def run_agent(query: str) -> dict:
    graph = build_graph()

    result = graph.invoke({
        "query": query,
        "task_type": "",
        "result": "",
        "final_answer": "",
        "classifier_source": "",
        "route_reason": "",
    })

    return result


def main():
    print("ResearchAgent v0.1 CLI")
    print("输入 q / quit / exit 退出")

    while True:
        query = input("\n请输入你的科研问题：\n> ")

        if query.lower() in ["q", "quit", "exit"]:
            print("已退出。")
            break

        result = run_agent(query)

        print("\n===== Agent 输出 =====")
        print(result["final_answer"])


if __name__ == "__main__":
    main()

'''


from src.research_agent.graph.workflow import build_graph


def create_initial_state(query: str) -> dict:
    return {
        "query": query,
        "task_type": "",
        "result": "",
        "final_answer": "",
        "classifier_source": "",
        "route_reason": "",
    }


def run_agent_with_graph(graph, query: str) -> dict:
    """
    使用已经编译好的 graph 执行一次查询。
    避免每次调用都重复 build_graph。
    """
    return graph.invoke(create_initial_state(query))


def run_agent(query: str) -> dict:
    graph = build_graph()
    result = graph.invoke(create_initial_state(query))
    return result


def print_welcome():
    print("=" * 60)
    print("ResearchAgent v0.1")
    print("A minimal LangGraph-based research assistant demo.")
    print("=" * 60)
    print("支持任务类型：")
    print("1. paper_question          论文问答 / 文献总结")
    print("2. experiment_analysis     实验结果 / 幻觉指标分析")
    print("3. dataset_recommendation  数据集推荐")
    print("4. report_generation       组会汇报 / PPT 文案")
    print("5. code_help               代码 / 环境 / 报错问题")
    print("6. general                 通用科研助手")
    print()
    print("输入 q / quit / exit 退出")


def main():
    graph = build_graph()

    print_welcome()

    while True:
        query = input("\n请输入你的科研问题：\n> ").strip()

        if query.lower() in ["q", "quit", "exit"]:
            print("已退出。")
            break

        if not query:
            print("输入不能为空，请重新输入。")
            continue

        #result = graph.invoke(create_initial_state(query))
        result = run_agent_with_graph(graph, query)

        print("\n===== Agent 输出 =====")
        print(result["final_answer"])


if __name__ == "__main__":
    main()