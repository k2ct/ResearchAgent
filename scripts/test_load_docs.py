from pathlib import Path
import sys


# 把项目根目录 F:\ResearchAgent 加入 Python 模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.research_agent.rag.loaders import (
    load_all_documents,
    summarize_documents,
    format_document_preview,
)


def main():
    docs = load_all_documents()

    print("=" * 60)
    print(f"Loaded {len(docs)} documents.")
    print("=" * 60)

    summary = summarize_documents(docs)

    print("\nDocument summary by source_type:")
    for source_type, count in summary.items():
        print(f"- {source_type}: {count}")

    print("\n" + "=" * 60)
    print("Document previews")
    print("=" * 60)

    for i, doc in enumerate(docs):
        print(f"\n[{i}]")
        print(format_document_preview(doc))
        print("-" * 60)


if __name__ == "__main__":
    main()