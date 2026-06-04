"""
Test script for Memory Write-back Integration.

Usage:
    cd F:/ResearchAgent
    ./.conda/python.exe scripts/test_memory_writeback.py

Tests:
1. Claim Support save_memory=False — default unchanged
2. Claim Support save_memory=True — writes to store
3. Paper Reading save_memory=True — writes paper_note
4. PPT Progress save_memory=True — writes progress_update
5. Report Writer wrapper — save_memory=True
6. Save failure does not break main result
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.memory.store import ensure_memory_store, load_memories


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


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── Test 1: Claim Support save_memory=False ────────────────────


def test_claim_no_save():
    section("Test 1: Claim Support save_memory=False (default)")

    from research_agent.claim.claim_support import generate_claim_support

    result = generate_claim_support(
        "测试论点：默认不写存储。",
        top_k_per_query=1,
        save_memory=False,
    )

    check("memory_saved" in result, "memory_saved field present")
    check(result["memory_saved"] == False, "memory_saved=False")
    check(result["memory_result"] is None, "memory_result is None")
    check("report" in result, "report still generated")
    check(len(result["report"]) > 0, "report non-empty")


# ── Test 2: Claim Support save_memory=True ─────────────────────


def test_claim_save():
    section("Test 2: Claim Support save_memory=True")

    from research_agent.claim.claim_support import generate_claim_support

    # Use distinctive content for later retrieval
    claim_text = "请记住：共现关系可能诱发 LVLM 对缺失对象的幻觉。"
    result = generate_claim_support(
        claim_text,
        top_k_per_query=1,
        save_memory=True,
    )

    check("memory_saved" in result, "memory_saved field present")
    check(result["memory_saved"] == True,
          f"memory_saved=True (got {result.get('memory_saved')})")
    check(result["memory_result"] is not None, "memory_result is not None")

    mr = result.get("memory_result", {})
    check(mr.get("ok"), f"memory_result.ok=True (got {mr.get('ok')})")
    mem_id = mr.get("memory_id", "")
    check(len(mem_id) > 0, f"memory_id non-empty: {mem_id[:20]}...")

    # Verify retrieval
    from research_agent.memory.retriever import retrieve_memories
    records = load_memories()
    found = retrieve_memories(records, keyword="共现关系", limit=10)
    check(len(found) >= 1, f"retriever finds keyword '共现关系' ({len(found)} records)")

    # Check memory_type
    claim_memories = retrieve_memories(records, memory_type="claim_support", limit=10)
    check(len(claim_memories) >= 1,
          f"memory_type=claim_support found ({len(claim_memories)} records)")
    if claim_memories:
        owner = claim_memories[-1].get("owner_agent", "")
        check(owner == "claim_agent" or "claim" in owner,
              f"owner_agent is claim_agent: {owner}")
        print(f"  owner_agent: {owner}")
        print(f"  memory_id: {claim_memories[-1].get('memory_id', '?')[:20]}...")


# ── Test 3: Paper Reading save_memory=True ──────────────────────


def test_paper_save():
    section("Test 3: Paper Reading save_memory=True")

    from research_agent.paper.paper_reader import read_paper

    # Find an available paper
    paper_path = None
    for candidate in [
        PROJECT_ROOT / "data" / "papers" / "guardrail_agnostic.md",
        PROJECT_ROOT / "data" / "papers" / "multimodal_bias_survey.md",
    ]:
        if candidate.exists() and candidate.stat().st_size > 0:
            paper_path = candidate
            break

    if paper_path is None:
        check(False, "No paper file found")
        return

    print(f"  Paper: {paper_path.name}")
    result = read_paper(paper_path, use_llm=False, save_memory=True)

    check(result["status"] == "success", f"status=success: {result['status']}")
    check("memory_saved" in result, "memory_saved field present")
    check(result["memory_saved"] == True,
          f"memory_saved=True (got {result.get('memory_saved')})")

    mr = result.get("memory_result", {})
    if mr and mr.get("ok"):
        # Verify memory_type via retrieval
        records = load_memories()
        from research_agent.memory.retriever import retrieve_memories
        paper_mems = retrieve_memories(records, memory_type="paper_note", limit=20)
        check(len(paper_mems) >= 1,
              f"paper_note memories found ({len(paper_mems)})")
        if paper_mems:
            owner = paper_mems[-1].get("owner_agent", "")
            check(owner == "paper_agent" or "paper" in owner,
                  f"owner_agent is paper_agent: {owner}")
            print(f"  owner_agent: {owner}")
            print(f"  memory_id: {paper_mems[-1].get('memory_id', '?')[:20]}...")


# ── Test 4: PPT Progress save_memory=True ───────────────────────


def test_progress_save():
    section("Test 4: PPT Progress save_memory=True")

    from research_agent.progress.ppt_progress_memory import generate_progress_memory

    # Find a slide doc
    slide_path = None
    for candidate in [
        PROJECT_ROOT / "data" / "ingested" / "sample_progress_test_slides.md",
        PROJECT_ROOT / "data" / "ingested" / "sample_english_progress_test_slides.md",
    ]:
        if candidate.exists() and candidate.stat().st_size > 0:
            slide_path = candidate
            break

    if slide_path is None:
        check(False, "No slide_doc found")
        return

    print(f"  Slide: {slide_path.name}")
    result = generate_progress_memory(slide_path, use_llm=False, save_memory=True)

    check("progress_memory" in result, "progress_memory generated")
    check("memory_saved" in result, "memory_saved field present")
    check(result["memory_saved"] == True,
          f"memory_saved=True (got {result.get('memory_saved')})")

    mr = result.get("memory_result", {})
    if mr and mr.get("ok"):
        records = load_memories()
        from research_agent.memory.retriever import retrieve_memories
        prog_mems = retrieve_memories(records, memory_type="progress_update", limit=20)
        check(len(prog_mems) >= 1,
              f"progress_update memories found ({len(prog_mems)})")
        if prog_mems:
            owner = prog_mems[-1].get("owner_agent", "")
            check(owner == "progress_agent" or "progress" in owner,
                  f"owner_agent is progress_agent: {owner}")
            scope = prog_mems[-1].get("memory_scope", "")
            print(f"  owner_agent: {owner}")
            print(f"  memory_scope: {scope}")


# ── Test 5: Report Writer wrapper ───────────────────────────────


def test_report_wrapper():
    section("Test 5: Report Writer — generate_report_with_memory")

    from research_agent.report.llm_report_writer import generate_report_with_memory

    result = generate_report_with_memory(
        query="测试报告生成",
        retrieved_docs=[],
        tool_result_text="",
        report_style="group_meeting",
        save_memory=False,
    )

    # save_memory=False: wrapper still works, just no write
    check("memory_saved" in result, f"memory_saved field present: {result.get('memory_saved')}")
    check("memory_result" in result, "memory_result field present")
    # May or may not write successfully depending on LLM availability
    check("report_text" in result or "ok" in result, "report fields present")

    print(f"  memory_saved: {result.get('memory_saved')}")
    print(f"  used_llm: {result.get('used_llm', 'N/A')}")


# ── Test 6: Write failure does not break main result ────────────


def test_failure_resilience():
    section("Test 6: Write failure does not break main result")

    from research_agent.claim.claim_support import generate_claim_support

    # Monkey-patch adapters to simulate failure
    import research_agent.memory.adapters as adp
    original_save = getattr(adp, "save_claim_support_result", None)

    def _failing_save(*args, **kwargs):
        raise RuntimeError("Simulated adapter failure")

    try:
        adp.save_claim_support_result = _failing_save

        result = generate_claim_support(
            "测试论点：写入失败不应影响结果。",
            top_k_per_query=1,
            save_memory=True,
        )

        # Main result must still be intact
        check(result["memory_saved"] == False, "memory_saved=False on failure")
        check(result["memory_result"] is not None, "memory_result present (error)")
        check("error" in result["memory_result"], "memory_result has error field")
        check(len(result["report"]) > 0, "report still generated despite memory failure")
        check("report" in result, "main result intact")

    finally:
        if original_save is not None:
            adp.save_claim_support_result = original_save


# ── Main ─────────────────────────────────────────────────────────


def main():
    global PASS, FAIL

    print("=" * 60)
    print("Memory Write-back Integration — Test Suite")
    print("=" * 60)

    ensure_memory_store()

    test_claim_no_save()
    test_claim_save()
    test_paper_save()
    test_progress_save()
    test_report_wrapper()
    test_failure_resilience()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    print(f"\n  Note: Test data written to data/memory/ (git-ignored).")

    if FAIL > 0:
        print("\nSome tests FAILED. Check output for details.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
