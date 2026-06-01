from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from research_agent.tools.csv_analyzer import (
    analyze_csv_file,
    format_csv_analysis,
)


def main():
    test_files = [
        "data/experiments/sample_metrics.csv",
        "data/experiments/non_existing.csv",
        "README.md",
    ]

    print("=" * 80)
    print("Test CSV Analyzer")
    print("=" * 80)

    for file_path in test_files:
        print("\n" + "=" * 80)
        print(f"File: {file_path}")
        print("=" * 80)

        analysis = analyze_csv_file(file_path)
        print(format_csv_analysis(analysis))


if __name__ == "__main__":
    main()