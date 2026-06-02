from pathlib import Path
import sys

# 将 src 和项目根加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# 导入 Agent 构建与初始 State
from run_cli import create_initial_state
from research_agent.graph.workflow import build_graph

# 测试问题列表
TEST_QUERIES = [
    # CSV 工具 + RAG
    "请分析 data/experiments/sample_metrics.csv",

    # JSONL 工具 + RAG
    "请分析 data/experiments/sample_generations.jsonl",

    # 仅 RAG
    "OpenImages-MIAP 的性别标注是图像级还是 bbox 级？",

    # 无工具、弱证据
    "我今天应该怎么安排科研任务",

    # 代码问题
    "ModuleNotFoundError: No module named langgraph 怎么解决",
]

def main():
    graph = build_graph()

    for query in TEST_QUERIES:
        print("=" * 100)
        print("用户输入：", query)
        print("=" * 100)

        # 执行 Agent
        result = graph.invoke(create_initial_state(query))

        # 打印最终回答
        print(result["final_answer"])

        # 打印 debug 信息
        print("\nDebug 信息：")
        print("- task_type:", result.get("task_type"))
        print("- tool_used:", result.get("tool_used"))
        print("- tool_result ok:", result.get("tool_result", {}).get("ok"))
        print("- retrieved_docs 数量:", len(result.get("retrieved_docs", [])))
        print("- sources 数量:", len(result.get("sources", [])))
        print("- evidence_status:", result.get("evidence_status"))
        print("- evidence_reason:", result.get("evidence_reason"))
        print("- evidence_warnings:", result.get("evidence_warnings"))
        print()

if __name__ == "__main__":
    main()