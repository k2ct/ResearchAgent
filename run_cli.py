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
    print("ResearchAgent v0.5 Memory + Multi-Agent Preview")
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

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.graph.workflow import build_graph



'''
def create_initial_state(query: str) -> dict:
    return {
        "query": query,
        "task_type": "",
        "result": "",
        "final_answer": "",
        "classifier_source": "",
        "route_reason": "",
    }
'''
def create_initial_state(query: str) -> dict:
    return {
        "query": query,
        "task_type": "",
        "result": "",
        "final_answer": "",
        "classifier_source": "",
        "route_reason": "",
        "retrieved_docs": [],
        "sources": [],

        # Day 12 新增
        "tool_used": "none",
        "tool_result": {},
        "tool_result_text": "",

        # Day 13 新增
        "evidence_status": "",
        "evidence_reason": "",
        "evidence_warnings": [],

        # Phase 3: Memory-aware
        "memory_context": "",
        "retrieved_memories": [],
        "memory_count": 0,
        "memory_used": False,
        "memory_error": "",

        # Phase 3: Multi-Agent
        "multi_agent_enabled": False,
        "primary_agent": "",
        "handoff_plan": {},
        "handoff_results": [],
        "handoff_summary": "",
        "handoff_sources": [],
        "handoff_memory_ids": [],
        "handoff_count": 0,
        "memory_written": False,
        "memory_write_error": "",
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
    print("ResearchAgent v0.5 Memory + Multi-Agent Preview")
    print("LangGraph + Agentic RAG + CSV/JSONL Tools + Evidence Checker + LLM-assisted Report Writer + Memory System + Multi-Agent")
    print("=" * 60)
    print("支持任务类型：")
    print("1. paper_question          论文问答 / 文献总结")
    print("2. experiment_analysis     实验结果 / CSV/JSONL 文件分析")
    print("3. dataset_recommendation  数据集推荐 / 数据集说明")
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

        # Multi-agent metrics
        if result.get("multi_agent_enabled"):
            print(f"\n--- Multi-Agent ---")
            print(f"  Primary Agent: {result.get('primary_agent', 'N/A')}")
            print(f"  Handoffs: {result.get('handoff_count', 0)}")
            print(f"  Summary: {result.get('handoff_summary', '')}")
            if result.get("memory_written"):
                print(f"  Memory: written")
            if result.get("memory_write_error"):
                print(f"  Memory Write Error: {result['memory_write_error']}")

        # Memory metrics (always show if available)
        if result.get("memory_used"):
            print(f"\n--- Memory ---")
            print(f"  Records: {result.get('memory_count', 0)}")
        if result.get("memory_error"):
            print(f"  Error: {result['memory_error']}")


if __name__ == "__main__":
    main()