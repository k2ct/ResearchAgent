"""
Test RAG retrieval mode switching (vector vs hybrid).

Verifies:
- Both modes return results
- Metadata filters work correctly (dataset_recommendation -> dataset_doc, etc.)
- hybrid mode does not break vector mode
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.rag.retriever import retrieve_documents
from research_agent.rag.hybrid_retriever import hybrid_retrieve_documents

TEST_CASES = [
    {
        "query": "OpenImages-MIAP 的性别标注是图像级还是 bbox 级？",
        "task_type": "dataset_recommendation",
        "expected_source_type": "dataset_doc",
    },
    {
        "query": "coco_val_n300_g1 这个实验的目的是什么？",
        "task_type": "experiment_analysis",
        "expected_source_type": "experiment_doc",
    },
    {
        "query": "Guardrail-Agnostic 这篇论文适合放在哪类 related work？",
        "task_type": "paper_question",
        "expected_source_type": "paper_note",
    },
]

PASS = 0
FAIL = 0


def check(condition: bool, label: str):
    global PASS, FAIL
    if condition:
        print(f"    PASS  {label}")
        PASS += 1
    else:
        print(f"    FAIL  {label}")
        FAIL += 1


def print_doc(doc, idx: int, max_chars: int = 200):
    """Print a single retrieved document."""
    meta = doc.metadata
    preview = doc.page_content[:max_chars].replace("\n", " ")
    print(f"    [{idx}] source_type={meta.get('source_type','?')} "
          f"path={meta.get('path','?')} "
          f"title={meta.get('title','?')}")
    if meta.get("dataset"):
        print(f"        dataset={meta['dataset']}  run_tag={meta.get('run_tag','')}")
    print(f"        preview: {preview}...")


def test_retrieval_mode(mode: str, query: str, task_type: str, expected_source_type: str):
    """
    Test a single retrieval mode for a given query/task_type.

    Checks:
    - At least 1 result returned
    - All results have the correct source_type (metadata filter)
    """
    print(f"\n{'─' * 60}")
    print(f"  Mode: {mode:6s}  |  Query: {query[:50]}...")
    print(f"{'─' * 60}")

    if mode == "vector":
        # Force vector mode via explicit parameter
        docs = retrieve_documents(
            query=query,
            task_type=task_type,
            top_k=3,
            retrieval_mode="vector",
        )
    else:
        # Force hybrid mode via explicit parameter
        docs = retrieve_documents(
            query=query,
            task_type=task_type,
            top_k=3,
            retrieval_mode="hybrid",
        )

    check(len(docs) > 0, f"returned {len(docs)} docs (expected >= 1)")

    for i, doc in enumerate(docs, start=1):
        print_doc(doc, i)

        actual_source_type = doc.metadata.get("source_type", "")
        check(
            actual_source_type == expected_source_type,
            f"doc[{i}] source_type is '{actual_source_type}' (expected '{expected_source_type}')",
        )

    return True


def main():
    global PASS, FAIL

    print("=" * 60)
    print("RAG Retrieval Mode Test Suite")
    print("=" * 60)
    print(f"Env RAG_RETRIEVAL_MODE: vector (default)")

    for tc in TEST_CASES:
        test_retrieval_mode(
            mode="vector",
            query=tc["query"],
            task_type=tc["task_type"],
            expected_source_type=tc["expected_source_type"],
        )
        test_retrieval_mode(
            mode="hybrid",
            query=tc["query"],
            task_type=tc["task_type"],
            expected_source_type=tc["expected_source_type"],
        )

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    if FAIL > 0:
        print("\nSome tests FAILED. Check the output above.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
