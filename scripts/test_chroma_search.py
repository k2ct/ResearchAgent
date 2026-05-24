from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from research_agent.rag.indexer import similarity_search


TEST_QUERIES = [
    "OpenImages-MIAP 的性别标注是图像级还是 bbox 级？",
    "coco_val_n300_g1 这个实验的目的是什么？",
    "Guardrail-Agnostic 这篇论文关注什么问题？",
    "GQA 对场景图和关系分析有什么价值？",
]


def main():
    for query in TEST_QUERIES:
        print("=" * 80)
        print(f"Query: {query}")
        print("=" * 80)

        docs = similarity_search(query, k=3)

        for i, doc in enumerate(docs, start=1):
            print(f"\n[{i}]")
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