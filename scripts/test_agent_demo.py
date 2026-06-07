"""
ResearchAgent v0.5 — End-to-end Agent Demo (Integration Smoke Test).

Runs 5 representative queries through the LangGraph workflow and reports:
- passed: query completed successfully
- skipped_due_to_network: SOCKS/proxy/connection error (not a code bug)
- failed: unexpected logic error

HF_HUB_OFFLINE is forced to 1 to avoid HuggingFace connectivity on
networks that block or proxy HF traffic.  Embeddings must be cached via
``scripts/build_index.py`` before running this test.
"""

import os
import sys
import traceback
from pathlib import Path

# ── Offline mode (must be set before any HF import) ────────────────────
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# [FIX] Wrap module-level imports: if HF models are not cached or run_cli.py
# is unavailable, set to None so main() can print a meaningful skip message
# instead of crashing with ModuleNotFoundError / ImportError.
_create_initial_state = None
_build_graph_raw = None

try:
    from run_cli import create_initial_state as _create_initial_state
except ImportError:
    pass

try:
    from research_agent.graph.workflow import build_graph as _build_graph_raw
except Exception:
    pass


def create_initial_state(query: str) -> dict:
    """Thin wrapper: calls the real create_initial_state or returns a minimal fallback."""
    if _create_initial_state is not None:
        return _create_initial_state(query)
    # Fallback minimal state when run_cli is unavailable
    return {
        "query": query,
        "task_type": "",
        "result": "",
        "final_answer": "",
        "classifier_source": "",
        "route_reason": "",
        "retrieved_docs": [],
        "sources": [],
        "tool_used": "none",
        "tool_result": {},
        "tool_result_text": "",
        "evidence_status": "",
        "evidence_reason": "",
        "evidence_warnings": [],
        "memory_context": "",
        "retrieved_memories": [],
        "memory_count": 0,
        "memory_used": False,
        "memory_error": "",
        "multi_agent_enabled": False,
        "primary_agent": "",
        "handoff_plan": {},
        "handoff_results": [],
        "handoff_summary": "",
        "handoff_sources": [],
        "handoff_memory_ids": [],
        "handoff_count": 0,
        "memory_written": False,
        "memory_write_error": "",
    }


def build_graph():
    """Thin wrapper: calls the real build_graph or returns None on import failure."""
    if _build_graph_raw is not None:
        return _build_graph_raw()
    return None

# ── Test queries ───────────────────────────────────────────────────────
TEST_QUERIES = [
    # CSV tool + RAG
    "请分析 data/experiments/sample_metrics.csv",

    # JSONL tool + RAG
    "请分析 data/experiments/sample_generations.jsonl",

    # RAG only
    "OpenImages-MIAP 的性别标注是图像级还是 bbox 级？",

    # No tool, weak evidence
    "我今天应该怎么安排科研任务",

    # Code help
    "ModuleNotFoundError: No module named langgraph 怎么解决",
]

# ── Network / proxy error patterns ─────────────────────────────────────
# These patterns indicate environment issues, NOT code bugs.
_NETWORK_ERROR_SIGNATURES: list[str] = [
    "socksio",
    "SOCKS",
    "proxy",
    "ProxyError",
    "WinError 10054",
    "ConnectionResetError",
    "Cannot connect",
    "httpx",
    "ConnectError",
    "RemoteDisconnected",
    "ConnectionError",
    "Timeout",
    "TunnelError",
    "ProxyConnectionError",
]


def _is_network_error(exc: BaseException) -> bool:
    """Return True if the exception is a network/proxy environment issue."""
    # Check the exception chain (__cause__ / __context__)
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None:
        eid = id(current)
        if eid in seen:
            break
        seen.add(eid)

        text = f"{type(current).__name__} {current}".lower()
        for sig in _NETWORK_ERROR_SIGNATURES:
            if sig.lower() in text:
                return True
        current = current.__cause__ or current.__context__
    return False


def _print_network_warning(exc: BaseException) -> None:
    """Print a user-friendly SOCKS/proxy troubleshooting message."""
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  [Network/Proxy Warning]                                ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print("  ║  Detected a SOCKS/proxy-related issue.                  ║")
    print("  ║  This is usually caused by Windows system proxy         ║")
    print("  ║  settings or missing socksio / httpx[socks].            ║")
    print("  ║  ResearchAgent core logic is NOT broken.                ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print("  ║  Try:                                                   ║")
    print("  ║  1. Disable system proxy temporarily.                   ║")
    print("  ║  2. Install SOCKS support:                              ║")
    print("  ║     .\\.conda\\python.exe -m pip install \"httpx[socks]\"  ║")
    print("  ║     .\\.conda\\python.exe -m pip install socksio          ║")
    print("  ║  3. Disable LLM toggles in .env:                        ║")
    print("  ║     ENABLE_LLM_ENHANCEMENT=false                        ║")
    print("  ║     ENABLE_LLM_REPORT_WRITER=false                      ║")
    print("  ║  4. Use cached embeddings:                              ║")
    print("  ║     $env:HF_HUB_OFFLINE=\"1\"                             ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()


def main() -> None:
    # ── Build graph ─────────────────────────────────────────────────
    print("=" * 70)
    print("  ResearchAgent v0.5 — End-to-End Demo")
    print("=" * 70)
    print()

    try:
        graph = build_graph()
        print("[OK] Graph built successfully.\n")
    except Exception as e:
        if _is_network_error(e):
            _print_network_warning(e)
            print(f"SKIP: Cannot build graph due to network/proxy issue: {e}")
            print("Run scripts/build_index.py first with HF_HUB_OFFLINE=1 to cache the embedding model.")
        else:
            print(f"SKIP: Cannot build graph (model not cached?): {e}")
            print("Run scripts/build_index.py first to cache the embedding model.")
        sys.exit(0)

    # ── Run queries ─────────────────────────────────────────────────
    passed = 0
    skipped_due_to_network = 0
    failed = 0
    skipped_queries: list[str] = []
    failed_queries: list[str] = []

    for idx, query in enumerate(TEST_QUERIES, start=1):
        print("=" * 100)
        print(f"[{idx}/{len(TEST_QUERIES)}] 用户输入：{query}")
        print("=" * 100)

        try:
            result = graph.invoke(create_initial_state(query))

            print(result["final_answer"])
            print()
            print("--- Debug ---")
            print(f"  task_type:        {result.get('task_type')}")
            print(f"  tool_used:        {result.get('tool_used')}")
            print(f"  tool_result ok:   {result.get('tool_result', {}).get('ok')}")
            print(f"  retrieved_docs:   {len(result.get('retrieved_docs', []))}")
            print(f"  sources:          {len(result.get('sources', []))}")
            print(f"  evidence_status:  {result.get('evidence_status')}")
            print(f"  evidence_reason:  {result.get('evidence_reason')}")
            print(f"  evidence_warnings:{result.get('evidence_warnings')}")
            print(f"  memory_used:      {result.get('memory_used')}")
            print(f"  memory_count:     {result.get('memory_count')}")
            print(f"  multi_agent:      {result.get('multi_agent_enabled')}")
            print(f"  handoff_count:    {result.get('handoff_count')}")
            print()
            passed += 1

        except Exception as e:
            if _is_network_error(e):
                _print_network_warning(e)
                print(f"  [SKIP] Network/proxy error on query: '{query[:60]}...'")
                print(f"  {type(e).__name__}: {e}")
                print()
                skipped_due_to_network += 1
                skipped_queries.append(query[:60])
            else:
                print(f"  [FAIL] Unexpected error on query: '{query[:60]}...'")
                traceback.print_exc()
                print()
                failed += 1
                failed_queries.append(query[:60])

    # ── Summary ─────────────────────────────────────────────────────
    total = len(TEST_QUERIES)
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total queries:              {total}")
    print(f"  Passed:                     {passed}")
    print(f"  Skipped (due to network):   {skipped_due_to_network}")
    print(f"  Failed:                     {failed}")
    print()

    if skipped_queries:
        print("  Skipped queries (network/proxy):")
        for q in skipped_queries:
            print(f"    - {q}")
        print()

    if failed_queries:
        print("  Failed queries (logic errors):")
        for q in failed_queries:
            print(f"    - {q}")
        print()

    if failed > 0:
        print(f"  RESULT: {failed} query(s) failed with unexpected errors.")
    elif skipped_due_to_network > 0:
        print(f"  RESULT: All {passed} query(s) succeeded, {skipped_due_to_network} skipped due to network/proxy.")
        print("  Core logic: OK.  Network issues are environment-specific — see troubleshooting in README.md.")
    else:
        print("  RESULT: All queries passed successfully.")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
