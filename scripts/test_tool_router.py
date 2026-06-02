from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from research_agent.tools.tool_router import (
    extract_file_paths_from_query,
    run_tool_from_query,
)


TEST_QUERIES = [
    "请分析 data/experiments/sample_metrics.csv",
    "请分析 data/experiments/sample_generations.jsonl",
    "请帮我看看 README.md",
    "这个问题没有文件路径，只是普通实验问题",
]


def main():
    for query in TEST_QUERIES:
        print("=" * 80)
        print("Query:", query)

        paths = extract_file_paths_from_query(query)
        print("Detected paths:", paths)

        result = run_tool_from_query(query)
        print("Tool used:", result["tool_used"])
        print("OK:", result["ok"])
        print("Formatted text:")
        print(result["formatted_text"][:1000])
        print()


if __name__ == "__main__":
    main()