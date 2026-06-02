from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from research_agent.tools.jsonl_analyzer import (
    analyze_jsonl_file,
    format_jsonl_analysis,
)


def main():
    test_files = [
        "data/experiments/sample_generations.jsonl",
        "data/experiments/non_existing.jsonl",
        "README.md",
    ]

    print("=" * 80)
    print("Test JSONL Analyzer")
    print("=" * 80)

    for file_path in test_files:
        print("\n" + "=" * 80)
        print(f"File: {file_path}")
        print("=" * 80)

        analysis = analyze_jsonl_file(file_path)
        print(format_jsonl_analysis(analysis))


if __name__ == "__main__":
    main()