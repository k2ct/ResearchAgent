from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from research_agent.rag.retriever import (
    build_metadata_filter,
    retrieve_documents_with_scores,
)


TEST_CASES = [
    {
        "query": "OpenImages-MIAP 的性别标注是图像级还是 bbox 级？",
        "task_type": "dataset_recommendation",
    },
    {
        "query": "coco_val_n300_g1 这个实验的目的是什么？",
        "task_type": "experiment_analysis",
    },
    {
        "query": "Guardrail-Agnostic 这篇论文关注什么问题？",
        "task_type": "paper_question",
    },
]


def main():
    print("=" * 80)
    print("Test Retriever Scores")
    print("=" * 80)

    for case in TEST_CASES:
        query = case["query"]
        task_type = case["task_type"]

        metadata_filter = build_metadata_filter(task_type)

        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print(f"Task Type: {task_type}")
        print(f"Metadata Filter: {metadata_filter}")
        print("=" * 80)

        results = retrieve_documents_with_scores(
            query=query,
            task_type=task_type,
            top_k=3,
            use_filter=True,
        )

        if not results:
            print("No results.")
            continue

        for i, (doc, score) in enumerate(results, start=1):
            print(f"\n[{i}] score = {score}")
            print("Source:", doc.metadata.get("path"))
            print("Source Type:", doc.metadata.get("source_type"))
            print("Title:", doc.metadata.get("title"))
            print("Dataset:", doc.metadata.get("dataset"))
            print("Run Tag:", doc.metadata.get("run_tag"))
            print("Chunk ID:", doc.metadata.get("chunk_id"))
            print("\nContent Preview:")
            print(doc.page_content[:300].replace("\n", " "))
            print("-" * 80)


if __name__ == "__main__":
    main()