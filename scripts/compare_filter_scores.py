from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from research_agent.rag.retriever import retrieve_documents_with_scores


def print_results(title, results):
    print("\n" + title)
    print("-" * 80)

    for i, (doc, score) in enumerate(results, start=1):
        print(f"[{i}] score={score}")
        print("source_type:", doc.metadata.get("source_type"))
        print("path:", doc.metadata.get("path"))
        print("preview:", doc.page_content[:150].replace("\n", " "))
        print()


def main():
    query = "OpenImages-MIAP 的性别标注是图像级还是 bbox 级？"
    task_type = "dataset_recommendation"

    print("=" * 80)
    print("Compare Retriever Scores With / Without Metadata Filter")
    print("=" * 80)
    print("Query:", query)
    print("Task Type:", task_type)

    filtered_results = retrieve_documents_with_scores(
        query=query,
        task_type=task_type,
        top_k=3,
        use_filter=True,
    )

    unfiltered_results = retrieve_documents_with_scores(
        query=query,
        task_type=task_type,
        top_k=3,
        use_filter=False,
    )

    print_results("With metadata filter", filtered_results)
    print_results("Without metadata filter", unfiltered_results)


if __name__ == "__main__":
    main()