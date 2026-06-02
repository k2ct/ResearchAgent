from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


from run_cli import create_initial_state
from research_agent.graph.workflow import build_graph


TEST_QUERIES = [
    # 应该有 RAG sources
    "OpenImages-MIAP 的性别标注是图像级还是 bbox 级？",

    # 应该有工具分析 + RAG
    "请分析 data/experiments/sample_metrics.csv",

    # 应该有 JSONL 工具分析 + RAG
    "请分析 data/experiments/sample_generations.jsonl",

    # 可能没有 RAG / Tool，应该 weak
    "我今天应该怎么安排科研任务",

    # code 暂时没有工具，应该 weak
    "ModuleNotFoundError: No module named langgraph 怎么解决",
]


def main():
    graph = build_graph()

    for query in TEST_QUERIES:
        print("=" * 100)
        print("用户输入：", query)
        print("=" * 100)

        result = graph.invoke(create_initial_state(query))

        print(result["final_answer"])

        print("\nDebug:")
        print("- task_type:", result.get("task_type"))
        print("- tool_used:", result.get("tool_used"))
        print("- retrieved_docs count:", len(result.get("retrieved_docs", [])))
        print("- sources count:", len(result.get("sources", [])))
        print("- evidence_status:", result.get("evidence_status"))
        print("- evidence_reason:", result.get("evidence_reason"))
        print("- evidence_warnings:", result.get("evidence_warnings"))
        print()


if __name__ == "__main__":
    main()