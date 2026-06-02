from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


from run_cli import create_initial_state
from research_agent.graph.workflow import build_graph


TEST_QUERIES = [
    "请分析 data/experiments/sample_metrics.csv",
    "请分析 data/experiments/sample_generations.jsonl",
    "请帮我解释 coco_val_n300_g1 这个实验的目的",
    "OpenImages-MIAP 的性别标注是图像级还是 bbox 级？",
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
        print("- tool_used:", result.get("tool_used"))
        print("- tool_result ok:", result.get("tool_result", {}).get("ok"))
        print("- retrieved_docs count:", len(result.get("retrieved_docs", [])))
        print("- sources count:", len(result.get("sources", [])))
        print()


if __name__ == "__main__":
    main()