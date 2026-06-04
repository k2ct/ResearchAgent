"""
Test script for Hybrid RAG v1 -- standalone test, does not modify any existing files.

Usage:
    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_hybrid_retriever.py
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from research_agent.rag.hybrid_retriever import (
    hybrid_retrieve_documents_with_scores,
    vector_search_documents,
    keyword_search_documents,
    get_source_type_for_task_local,
)


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
    {
        "query": "帮我生成 coco_val_n300_g1 实验的组会汇报文本",
        "task_type": "general",
        "expected_source_type": None,  # general 允许混合 source_type
    },
]


def _preview(text: str, max_chars: int = 200) -> str:
    """Truncate text preview."""
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def print_results(title: str, results, check_source_type=None):
    """Print retrieval results."""
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)

    if not results:
        print("  (No results)")
        return

    for i, (doc, score) in enumerate(results, start=1):
        metadata = doc.metadata
        source_type = metadata.get("source_type", "?")
        path = metadata.get("path", "?")
        doc_title = metadata.get("title", "")
        dataset = metadata.get("dataset", "")
        run_tag = metadata.get("run_tag", "")

        # Check source_type
        flag = ""
        if check_source_type and source_type != check_source_type:
            flag = " !! MISMATCH"

        print(f"\n  [{i}] score={score:.4f}{flag}")
        print(f"      source_type: {source_type}")
        print(f"      path:        {path}")
        if doc_title:
            print(f"      title:       {doc_title}")
        if dataset:
            print(f"      dataset:     {dataset}")
        if run_tag:
            print(f"      run_tag:     {run_tag}")
        print(f"      preview:     {_preview(doc.page_content)}")

    # Summary of source types
    if check_source_type:
        all_match = all(
            doc.metadata.get("source_type") == check_source_type
            for doc, _ in results
        )
        if all_match:
            print(f"\n  [OK] All results have source_type = {check_source_type}")
        else:
            types_found = set(doc.metadata.get("source_type") for doc, _ in results)
            print(f"\n  [WARN] Expected {check_source_type}, found: {types_found}")


def main():
    print("=" * 90)
    print("  Hybrid RAG v1 — Test Suite")
    print("=" * 90)

    passed = 0
    failed = 0

    for idx, case in enumerate(TEST_CASES, start=1):
        query = case["query"]
        task_type = case["task_type"]
        expected_st = case["expected_source_type"]

        print("\n" + "#" * 90)
        print(f"#  Test {idx}: {query}")
        print(f"#  Task Type: {task_type}")
        expected_filter = get_source_type_for_task_local(task_type)
        print(f"#  Source filter: {expected_filter}")
        print("#" * 90)

        try:
            # --- Vector Results ---
            vec_results = vector_search_documents(
                query=query,
                task_type=task_type,
                top_k=5,
            )
            print_results("Vector Search Results", vec_results, expected_st)

            # --- Keyword Results ---
            kw_results = keyword_search_documents(
                query=query,
                task_type=task_type,
                top_k=5,
            )
            print_results("Keyword Search Results", kw_results, expected_st)

            # --- Hybrid Fused Results ---
            fused_results = hybrid_retrieve_documents_with_scores(
                query=query,
                task_type=task_type,
                top_k=5,
            )
            print_results("Hybrid Fused Results", fused_results, expected_st)

            # --- Validation ---
            if expected_st is not None:
                if fused_results:
                    all_match = all(
                        doc.metadata.get("source_type") == expected_st
                        for doc, _ in fused_results
                    )
                    if all_match:
                        print(f"\n  [OK] PASS: All fused results have source_type = {expected_st}")
                        passed += 1
                    else:
                        types_found = set(
                            doc.metadata.get("source_type") for doc, _ in fused_results
                        )
                        print(f"\n  [FAIL] Expected {expected_st}, found {types_found}")
                        failed += 1
                else:
                    print(f"\n  [WARN] No hybrid results for {task_type}")
            else:
                # general: no filter expected
                print(f"\n  [OK] PASS: general task_type allows mixed source_types")
                passed += 1

        except Exception as e:
            print(f"\n  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Final summary
    print("\n" + "=" * 90)
    print(f"  Summary: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 90)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
