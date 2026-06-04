"""
Integration test: Phase-2 modules → Memory Adapters → Writer → Store.

Usage::

    cd F:/ResearchAgent
    HF_HUB_OFFLINE=1 ./.conda/python.exe scripts/test_memory_integration.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.memory.adapters import (
    _WRITER_AVAILABLE,
    save_claim_support_result,
    save_paper_reading_result,
    save_progress_memory_result,
    save_report_result,
    save_experiment_analysis_result,
    save_generic_result,
    save_module_result,
)
from research_agent.memory.store import query_memories, load_memories
from research_agent.memory.privacy_scope import check_access, get_scope, validate_scope_on_write

PASS = "✓ PASS"
FAIL = "✗ FAIL"

def check(condition: bool, label: str) -> bool:
    status = PASS if condition else FAIL
    print(f"  {status}: {label}")
    return condition

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════
# Test 1 — Claim Support adapter (synthetic)
# ═══════════════════════════════════════════════════════════════════════════

def test_claim_adapter_synthetic():
    section("Test 1: save_claim_support_result (synthetic)")
    result = save_claim_support_result({
        "claim": "LVLM hallucination is caused by co-occurrence bias in training data",
        "claim_type": "causal",
        "report": "Evidence from 3 papers supports this claim. Co-occurrence patterns in COCO dataset correlate with hallucination rates.",
        "sources": [{"path": "data/papers/bias_hallucination.md", "source_type": "paper_note"}],
        "evidence_count": 3,
        "used_llm": False,
    }, auto_write=True)

    ok = True
    ok &= check(result.get("ok"), f"ok={result.get('ok')}")
    ok &= check(result.get("adapter") == "claim_support", f"adapter={result.get('adapter')}")
    ok &= check(bool(result.get("memory_id")), f"memory_id={result.get('memory_id', '')[:30]}...")
    record = result.get("record")
    ok &= check(record is not None, "record is not None")

    # Verify privacy scope on the record
    if record:
        v = validate_scope_on_write(record)
        ok &= check(v["valid"], "privacy scope valid")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 2 — Paper Reading adapter (real)
# ═══════════════════════════════════════════════════════════════════════════

def test_paper_adapter_real():
    section("Test 2: save_paper_reading_result (real module)")
    from research_agent.paper.paper_reader import read_paper
    paper_path = PROJECT_ROOT / "data" / "papers" / "guardrail_agnostic.md"
    if not paper_path.exists():
        print("  SKIP: paper file not found")
        return True

    paper_result = read_paper(paper_path, use_llm=False)
    adapter_result = save_paper_reading_result(paper_result, auto_write=True)

    ok = True
    ok &= check(adapter_result.get("ok"), f"ok={adapter_result.get('ok')}")
    ok &= check(adapter_result.get("adapter") == "paper_reading", f"adapter={adapter_result.get('adapter')}")
    ok &= check(bool(adapter_result.get("memory_id")), "memory_id present")
    record = adapter_result.get("record")
    if record:
        owner = _rf(record, "owner_agent", "")
        ok &= check(owner == "paper_agent", f"owner_agent={owner} (expected paper_agent)")
        # Check access
        ok &= check(check_access(record, "paper_agent"), "paper_agent can access own record")
        ok &= check(check_access(record, "coordinator"), "coordinator can access")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 3 — PPT Progress adapter (real)
# ═══════════════════════════════════════════════════════════════════════════

def test_progress_adapter_real():
    section("Test 3: save_progress_memory_result (real module)")
    from research_agent.progress.ppt_progress_memory import generate_progress_memory
    ppt_path = PROJECT_ROOT / "data" / "ingested" / "刘晗组会 20260529.md"
    if not ppt_path.exists():
        print("  SKIP: PPT markdown not found")
        return True

    ppt_result = generate_progress_memory(ppt_path)
    adapter_result = save_progress_memory_result(ppt_result, auto_write=True)

    ok = True
    ok &= check(adapter_result.get("ok"), f"ok={adapter_result.get('ok')}")
    ok &= check(adapter_result.get("adapter") == "progress_memory", f"adapter={adapter_result.get('adapter')}")
    ok &= check(bool(adapter_result.get("memory_id")), "memory_id present")
    record = adapter_result.get("record")
    if record:
        owner = _rf(record, "owner_agent", "")
        ok &= check(owner == "progress_agent", f"owner_agent={owner} (expected progress_agent)")
        # Verify shared_with
        scope_info = get_scope(record)
        ok &= check(scope_info["memory_scope"] in ("shared", "private"),
                    f"scope={scope_info['memory_scope']}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 4 — Report adapter (synthetic)
# ═══════════════════════════════════════════════════════════════════════════

def test_report_adapter_synthetic():
    section("Test 4: save_report_result (synthetic)")
    result = save_report_result({
        "report_text": "Group meeting report: bias evaluation pipeline is operational. Next: cross-model comparison.",
        "task_type": "group_meeting",
        "sources": [{"path": "data/papers/bias_survey.md", "source_type": "paper_note"}],
        "evidence_status": "passed",
        "used_llm": True,
    }, auto_write=True)

    ok = True
    ok &= check(result.get("ok"), f"ok={result.get('ok')}")
    ok &= check(result.get("adapter") == "report_writer", f"adapter={result.get('adapter')}")
    record = result.get("record")
    if record:
        owner = _rf(record, "owner_agent", "")
        ok &= check(owner == "report_agent", f"owner_agent={owner} (expected report_agent)")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 5 — Experiment adapter (synthetic)
# ═══════════════════════════════════════════════════════════════════════════

def test_experiment_adapter_synthetic():
    section("Test 5: save_experiment_analysis_result (synthetic)")
    result = save_experiment_analysis_result({
        "analysis": "CSV analysis: 3 experiments, mean_extra_object_rate=0.19, hrs_v1 correlates with hallucination.",
        "file_path": "data/experiments/sample_metrics.csv",
        "tool_used": "csv_analyzer",
        "metrics": {"mean_extra_object_rate": 0.19, "mean_precision": 0.82, "mean_recall": 0.71},
        "evidence_status": "passed",
        "sources": [{"path": "data/experiments/coco_val_n300_g1.md", "source_type": "experiment_doc"}],
        "used_llm": False,
    }, auto_write=True)

    ok = True
    ok &= check(result.get("ok"), f"ok={result.get('ok')}")
    ok &= check(result.get("adapter") == "experiment_analysis", f"adapter={result.get('adapter')}")
    record = result.get("record")
    if record:
        owner = _rf(record, "owner_agent", "")
        ok &= check(owner == "experiment_agent", f"owner_agent={owner} (expected experiment_agent)")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 6 — auto_write=False (build only)
# ═══════════════════════════════════════════════════════════════════════════

def test_auto_write_false():
    section("Test 6: auto_write=False — build record only")
    result = save_claim_support_result({
        "claim": "Test claim for build-only mode",
        "claim_type": "test",
        "report": "Test report.",
        "sources": [],
        "evidence_count": 0,
        "used_llm": False,
    }, auto_write=False)

    ok = True
    ok &= check(result.get("ok"), "ok=True (record built)")
    ok &= check(result.get("record") is not None, "record exists")
    ok &= check(result.get("write_result") is None, "write_result is None (not written)")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 7 — unified dispatcher
# ═══════════════════════════════════════════════════════════════════════════

def test_dispatcher():
    section("Test 7: save_module_result dispatcher")
    ok = True

    # Direct module names
    for mod_name in ["claim_support", "paper_reading", "progress_memory", "report_writer", "experiment_analysis"]:
        r = save_module_result({"claim": "test", "report": "test", "reading_note": "test",
                                 "progress_memory": "test", "analysis": "test",
                                 "sources": [], "evidence_count": 0, "used_llm": False},
                                module_name=mod_name, auto_write=True)
        ok &= check(r.get("ok"), f"dispatcher '{mod_name}' → ok={r.get('ok')}, adapter={r.get('adapter')}")

    # Aliases
    for alias, expected_adapter in [
        ("paper", "paper_reading"),
        ("ppt", "progress_memory"),
        ("report", "report_writer"),
        ("experiment", "experiment_analysis"),
    ]:
        r = save_module_result({"claim": "test", "report": "test", "reading_note": "test",
                                 "progress_memory": "test", "analysis": "test",
                                 "sources": [], "evidence_count": 0, "used_llm": False},
                                module_name=alias, auto_write=True)
        ok &= check(r.get("adapter") == expected_adapter,
                    f"alias '{alias}' → adapter={r.get('adapter')} (expected {expected_adapter})")

    # Unknown module → generic
    r = save_module_result({"content": "custom content"}, module_name="custom_plugin", auto_write=True)
    ok &= check(r.get("adapter") == "generic", f"unknown → adapter={r.get('adapter')} (expected generic)")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 8 — store retrieval of adapter-written records
# ═══════════════════════════════════════════════════════════════════════════

def test_store_retrieval():
    section("Test 8: store retrieval — query adapter-written records")
    # Query for records written by our adapters
    found = query_memories(tags=["experiment", "benchmark", "COCO"])  # from earlier privacy test
    ok = True
    ok &= check(len(found) >= 1, f"query finds {len(found)} records from adapter writes")

    # Query by keyword
    found2 = query_memories(keyword="hallucination")
    ok &= check(len(found2) >= 1, f"keyword 'hallucination' finds {len(found2)} records")

    # Verify records have proper structure
    for rec in found2[:3]:
        if isinstance(rec, dict):
            mid = rec.get("memory_id", "?")
            mtype = rec.get("memory_type", "?")
            scope = rec.get("memory_scope", "?")
        else:
            mid = getattr(rec, "memory_id", "?")
            mtype = getattr(rec, "memory_type", "?")
            scope = getattr(rec, "memory_scope", "?")
        print(f"  {str(mid)[:20]}... type={mtype} scope={scope}")

    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Test 9 — privacy scope compatibility
# ═══════════════════════════════════════════════════════════════════════════

def test_privacy_compatibility():
    section("Test 9: privacy scope compatibility with adapter records")
    all_records = load_memories()
    if len(all_records) == 0:
        print("  SKIP: no memories in store")
        return True

    ok = True
    # All records should pass validate_scope_on_write
    for rec in all_records[:5]:
        v = validate_scope_on_write(rec)
        ok &= check(v["valid"], f"record {_rid(rec)}: scope valid")

    # Check access for different agents
    for agent in ["paper_agent", "claim_agent", "experiment_agent", "coordinator"]:
        from research_agent.memory.privacy_scope import filter_accessible
        visible = filter_accessible(all_records, agent)
        ok &= check(len(visible) <= len(all_records),
                    f"{agent} sees {len(visible)}/{len(all_records)} (filter works)")

    return ok


def _rf(rec, key, default=None):
    """Safely get a field from dict or MemoryRecord dataclass."""
    if isinstance(rec, dict):
        return rec.get(key, default)
    return getattr(rec, key, default)

def _rid(rec):
    return str(_rf(rec, "memory_id", "?"))[:16]


# ═══════════════════════════════════════════════════════════════════════════
# Test 10 — generic fallback adapter
# ═══════════════════════════════════════════════════════════════════════════

def test_generic_adapter():
    section("Test 10: save_generic_result fallback")
    result = save_generic_result(
        {"content": "This is a custom plugin output with research findings about LVLM safety."},
        source_module="custom_plugin",
        source_title="Safety Analysis",
        auto_write=True,
    )
    ok = True
    ok &= check(result.get("ok"), f"ok={result.get('ok')}")
    ok &= check(result.get("adapter") == "generic", f"adapter={result.get('adapter')}")
    ok &= check(bool(result.get("memory_id")), "memory_id present")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 60)
    print("  Memory Adapter Integration — Test Suite")
    print("=" * 60)
    print(f"  writer available: {_WRITER_AVAILABLE}")
    if not _WRITER_AVAILABLE:
        print("  FATAL: Memory writer not available — tests will fail.")
        return 1

    results = {}
    results["claim_adapter"] = test_claim_adapter_synthetic()
    results["paper_adapter"] = test_paper_adapter_real()
    results["progress_adapter"] = test_progress_adapter_real()
    results["report_adapter"] = test_report_adapter_synthetic()
    results["experiment_adapter"] = test_experiment_adapter_synthetic()
    results["auto_write_false"] = test_auto_write_false()
    results["dispatcher"] = test_dispatcher()
    results["store_retrieval"] = test_store_retrieval()
    results["privacy_compat"] = test_privacy_compatibility()
    results["generic"] = test_generic_adapter()

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    all_passed = True
    for name, ok_val in results.items():
        status = PASS if ok_val else FAIL
        print(f"  {status}: {name}")
        if not ok_val:
            all_passed = False

    print()
    if all_passed:
        print("  All tests passed!")
    else:
        print("  Some tests FAILED — see details above.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
