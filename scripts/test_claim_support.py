"""
Test Claim Support Retrieval v1.

Tests 5 scientific claims for evidence retrieval, classification,
query decomposition, evidence grouping, and report generation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.claim.claim_support import (
    classify_claim_intent,
    build_claim_queries,
    retrieve_claim_evidence,
    group_evidence_by_purpose,
    build_claim_support_report,
    generate_claim_support,
)

TEST_CLAIMS = [
    "共现关系可能诱发 LVLM 对缺失对象的幻觉。",
    "只依赖视觉 token 可能不足以缓解多模态幻觉。",
    "反事实样本可以帮助评估视觉语言模型中的偏差。",
    "实验 run_tag 可以作为追踪实验设置和结果的重要 metadata。",
    "组会 PPT 可以作为长期科研记忆的一部分。",
]

PASS = 0
FAIL = 0


def check(condition: bool, label: str):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {label}")
        PASS += 1
    else:
        print(f"  FAIL  {label}")
        FAIL += 1


def test_claim(claim: str, idx: int):
    """Test a single claim end-to-end."""
    print(f"\n{'=' * 70}")
    print(f"  Test {idx}: {claim[:60]}...")
    print(f"{'=' * 70}")

    # 1. Classify
    intent = classify_claim_intent(claim)
    print(f"\n  Claim Type: {intent['claim_type']}")
    print(f"  Keywords:   {intent['keywords'][:8]}")

    # 2. Build queries
    queries = build_claim_queries(claim)
    print(f"  Queries generated: {len(queries)}")
    for q in queries:
        print(f"    [{q['purpose']:14s}] task={q['task_type']:22s} {q['query'][:60]}...")

    check(len(queries) >= 3, f"at least 3 queries generated (got {len(queries)})")
    purposes = {q["purpose"] for q in queries}
    check("theory" in purposes, "theory query present")
    check("experiment" in purposes, "experiment query present")
    check(len(purposes) >= 4, f"at least 4 distinct purposes (got {len(purposes)})")

    # 3. Retrieve evidence
    evidence = retrieve_claim_evidence(claim, top_k_per_query=3)
    print(f"\n  Evidence retrieved: {len(evidence)} items")

    # 4. Group
    grouped = group_evidence_by_purpose(evidence)
    print(f"  Grouped counts:")
    for purpose, items in grouped.items():
        print(f"    {purpose:14s}: {len(items)} items")
    check(sum(len(v) for v in grouped.values()) == len(evidence), "grouped total matches evidence total")

    # 5. Build report
    report = build_claim_support_report(claim, grouped)
    print(f"\n  {'─' * 60}")
    print(f"  Report (first 1500 chars):")
    print(f"  {'─' * 60}")
    for line in report[:1500].splitlines():
        print(f"  {line}")

    # 6. Full pipeline
    result = generate_claim_support(claim, top_k_per_query=3)
    check(result["claim"] == claim, "result claim matches")
    check(result["claim_type"] == intent["claim_type"], "claim_type consistent")
    check(len(result["queries"]) >= 3, "queries in result")
    if len(result["sources"]) > 0:
        check(True, f"sources present ({len(result['sources'])})")
    else:
        # KB may not cover this claim — report should honestly state lack of evidence
        has_honest_report = (
            "未检索到" in result["report"]
            or "未检索" in result["report"]
            or "no evidence" in result["report"].lower()
        )
        check(has_honest_report, "no sources but report honestly states lack of evidence")
    check(len(result["report"]) > 100, f"report non-trivial ({len(result['report'])} chars)")

    return True


def main():
    global PASS, FAIL

    print("=" * 70)
    print("  Claim Support Retrieval v1 Test Suite")
    print("=" * 70)

    for i, claim in enumerate(TEST_CLAIMS, start=1):
        test_claim(claim, i)

    print(f"\n{'=' * 70}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 70}")

    if FAIL > 0:
        print("\nSome tests FAILED. Check above for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
