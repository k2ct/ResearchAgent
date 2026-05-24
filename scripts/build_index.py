from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from research_agent.rag.indexer import build_vector_index


def main():
    print("=" * 60)
    print("Build ResearchAgent Chroma Index")
    print("=" * 60)

    build_vector_index(reset=True)

    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()