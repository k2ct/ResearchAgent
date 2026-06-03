from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


from run_cli import create_initial_state
from research_agent.graph.workflow import build_graph


TEST_QUERIES = [
    "帮我生成 coco_val_n300_g1 实验的组会汇报文本",
    "请给我一份 OpenImages-MIAP 数据集相关的 PPT 汇报草稿",
    "请总结 Guardrail-Agnostic 这篇论文，生成组会汇报内容",
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
        print()


if __name__ == "__main__":
    main()