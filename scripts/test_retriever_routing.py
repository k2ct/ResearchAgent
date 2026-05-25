from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from research_agent.rag.retriever import (
    build_metadata_filter,
    retrieve_documents,
    format_retrieved_docs,
    extract_sources_from_docs,
)


TEST_CASES = [
    {
        "query": "Guardrail-Agnostic 这篇论文关注什么问题？",
        "task_type": "paper_question",
        "expected_source_type": "paper_note",
    },
    {
        "query": "coco_val_n300_g1 这个实验的目的是什么？",
        "task_type": "experiment_analysis",
        "expected_source_type": "experiment_doc",
    },
    {
        "query": "OpenImages-MIAP 的性别标注是图像级还是 bbox 级？",
        "task_type": "dataset_recommendation",
        "expected_source_type": "dataset_doc",
    },
    {
        "query": "GQA 对场景图和关系分析有什么价值？",
        "task_type": "dataset_recommendation",
        "expected_source_type": "dataset_doc",
    },
]


def main():
    print("=" * 80)
    print("Test Retriever Routing")
    print("=" * 80)

    for case in TEST_CASES:
        query = case["query"]
        task_type = case["task_type"]
        expected_source_type = case["expected_source_type"]

        metadata_filter = build_metadata_filter(task_type)

        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print(f"Task Type: {task_type}")
        print(f"Metadata Filter: {metadata_filter}")
        print(f"Expected Source Type: {expected_source_type}")
        print("=" * 80)

        docs = retrieve_documents(
            query=query,
            task_type=task_type,
            top_k=3,
            use_filter=True,
        )

        if not docs:
            print("No documents retrieved.")
            continue

        all_match = all(
            doc.metadata.get("source_type") == expected_source_type
            for doc in docs
        )

        print(f"Retrieved {len(docs)} documents.")
        print(f"All source_type match expected: {all_match}")

        print("\nRetrieved Docs:")
        print(format_retrieved_docs(docs, max_chars_per_doc=300))

        print("\nSources:")
        sources = extract_sources_from_docs(docs)
        for source in sources:
            print(f"- {source}")


if __name__ == "__main__":
    main()