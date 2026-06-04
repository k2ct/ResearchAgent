"""
Test Heuristic Reranker v1 for Hybrid RAG.

Compares hybrid retrieval with and without reranker for 3 queries.
Displays top-3 results side-by-side with scores, metadata, and previews.
"""

import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from langchain_core.documents import Document
from research_agent.rag.hybrid_retriever import hybrid_retrieve_documents_with_scores
from research_agent.rag.reranker import (
    extract_query_terms,
    compute_metadata_match_score,
    compute_content_match_score,
    compute_section_title_score,
    compute_rerank_score,
    rerank_candidates,
)

TEST_CASES = [
    {
        "query": "coco_val_n300_g1 这个实验的目的是什么？",
        "task_type": "experiment_analysis",
        "expected_source_type": "experiment_doc",
        "expected_run_tag": "coco_val_n300_g1",
    },
    {
        "query": "OpenImages-MIAP 的性别标注是图像级还是 bbox 级？",
        "task_type": "dataset_recommendation",
        "expected_source_type": "dataset_doc",
        "expected_dataset": "OpenImages-MIAP",
    },
    {
        "query": "Guardrail-Agnostic 这篇论文适合放在哪类 related work？",
        "task_type": "paper_question",
        "expected_source_type": "paper_note",
        "expected_title": "Guardrail-Agnostic",
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


def print_results(label: str, results: List[Tuple[Document, float]]):
    """Pretty-print retrieval results."""
    if not results:
        print(f"  {label}: (no results)")
        return

    for i, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        preview = doc.page_content[:150].replace("\n", " ")
        sec_title = meta.get("section_title", "")
        print(f"  [{i}] {label}  score={score:.4f}")
        print(f"       path={meta.get('path','?')}")
        print(f"       source_type={meta.get('source_type','?')}")
        print(f"       section_title={sec_title[:80] if sec_title else '(none)'}")
        print(f"       title={meta.get('title','?')}")
        if meta.get("dataset"):
            print(f"       dataset={meta['dataset']}  run_tag={meta.get('run_tag','')}")
        print(f"       preview: {preview}...")
        print()


def test_query(tc: dict):
    """Test one query with and without reranker."""
    query = tc["query"]
    task_type = tc["task_type"]
    expected_source = tc["expected_source_type"]

    print(f"\n{'=' * 70}")
    print(f"  Query: {query}")
    print(f"  Task:  {task_type}")
    print(f"{'=' * 70}")

    # ── Without reranker ──
    results_no_rerank = hybrid_retrieve_documents_with_scores(
        query=query,
        task_type=task_type,
        top_k=3,
        use_reranker=False,
    )

    # ── With reranker ──
    results_with_rerank = hybrid_retrieve_documents_with_scores(
        query=query,
        task_type=task_type,
        top_k=3,
        use_reranker=True,
    )

    # ── Basic checks ──
    check(len(results_no_rerank) > 0, f"Without reranker: got {len(results_no_rerank)} results")
    check(len(results_with_rerank) > 0, f"With reranker:    got {len(results_with_rerank)} results")

    # ── Metadata filter check ──
    for i, (doc, _) in enumerate(results_no_rerank, 1):
        check(
            doc.metadata.get("source_type") == expected_source,
            f"no-rerank[{i}] source_type is '{doc.metadata.get('source_type')}'",
        )
    for i, (doc, _) in enumerate(results_with_rerank, 1):
        check(
            doc.metadata.get("source_type") == expected_source,
            f"with-rerank[{i}] source_type is '{doc.metadata.get('source_type')}'",
        )

    # ── Top-1 expected content check ──
    if "expected_run_tag" in tc:
        top1_no = results_no_rerank[0][0].metadata.get("run_tag", "")
        top1_with = results_with_rerank[0][0].metadata.get("run_tag", "")
        check(
            tc["expected_run_tag"] in top1_no,
            f"no-rerank top-1 run_tag contains '{tc['expected_run_tag']}' (got '{top1_no}')",
        )
        check(
            tc["expected_run_tag"] in top1_with,
            f"with-rerank top-1 run_tag contains '{tc['expected_run_tag']}' (got '{top1_with}')",
        )

    if "expected_dataset" in tc:
        top1_no = results_no_rerank[0][0].metadata.get("dataset", "")
        top1_with = results_with_rerank[0][0].metadata.get("dataset", "")
        check(
            tc["expected_dataset"] in top1_no,
            f"no-rerank top-1 dataset contains '{tc['expected_dataset']}' (got '{top1_no}')",
        )
        check(
            tc["expected_dataset"] in top1_with,
            f"with-rerank top-1 dataset contains '{tc['expected_dataset']}' (got '{top1_with}')",
        )

    if "expected_title" in tc:
        top1_no = results_no_rerank[0][0].metadata.get("title", "")
        top1_with = results_with_rerank[0][0].metadata.get("title", "")
        check(
            tc["expected_title"].lower() in top1_no.lower(),
            f"no-rerank top-1 title contains '{tc['expected_title']}' (got '{top1_no[:60]}')",
        )
        check(
            tc["expected_title"].lower() in top1_with.lower(),
            f"with-rerank top-1 title contains '{tc['expected_title']}' (got '{top1_with[:60]}')",
        )

    # ── Print results ──
    print()
    print("  --- Without reranker ---")
    print_results("no_rerank", results_no_rerank)
    print("  --- With reranker ---")
    print_results("with_rerank", results_with_rerank)

    # ── Rerank score plausibility check ──
    if results_with_rerank and len(results_with_rerank) >= 2:
        top_score = results_with_rerank[0][1]
        second_score = results_with_rerank[1][1]
        check(
            top_score >= second_score,
            f"Rerank top-1 score ({top_score:.4f}) >= top-2 ({second_score:.4f})",
        )


def test_unit_functions():
    """Test individual reranker unit functions."""
    print(f"\n{'=' * 70}")
    print("  Unit Tests: extract_query_terms, compute_*_score")
    print(f"{'=' * 70}")

    # Test extract_query_terms
    terms = extract_query_terms("coco_val_n300_g1 experiment analysis")
    check("coco_val_n300_g1" in terms["phrases"], "extract: run_tag identified as phrase")
    check(len(terms["tokens"]) > 0, "extract: tokens extracted")
    check("experiment" in terms["token_set"], "extract: 'experiment' in token_set")

    # Test extract_query_terms with Chinese
    terms_cn = extract_query_terms("OpenImages-MIAP 的性别标注是图像级还是 bbox 级？")
    check("openimages-miap" in terms_cn["phrases"], "extract: dataset name lowercased")
    # CJK phrases capture contiguous blocks; individual chars go to tokens
    has_cjk_phrase = any("性别" in p for p in terms_cn["phrases"])
    has_cjk_token = "性" in terms_cn["tokens"] or "别" in terms_cn["tokens"]
    check(has_cjk_phrase or has_cjk_token, "extract: CJK tokens present")

    print("\n  Unit tests complete.\n")


def main():
    global PASS, FAIL

    print("=" * 70)
    print("  Heuristic Reranker v1 Test Suite")
    print("=" * 70)

    test_unit_functions()

    for tc in TEST_CASES:
        test_query(tc)

    print(f"\n{'=' * 70}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 70}")

    if FAIL > 0:
        print("\nSome tests FAILED. Check the output above for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
