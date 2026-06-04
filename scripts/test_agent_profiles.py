"""
Test Multi-Agent Profiles.

Verifies profile completeness, agent selection logic, and convenience accessors.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.agents.profiles import (
    AgentProfile,
    build_default_agent_profiles,
    get_agent_profile,
    list_agent_profiles,
    select_agent_for_task,
    get_accessible_memory_types,
    get_allowed_tools,
    build_agent_system_prompt,
)

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


# ── Test 1: Profile count ─────────────────────────────────────────

def test_profile_count():
    print("\n── Test 1: 9 agents defined ──")
    profiles = build_default_agent_profiles()
    expected = [
        "coordinator_agent", "paper_agent", "experiment_agent",
        "claim_agent", "progress_agent", "report_agent",
        "code_agent", "memory_agent", "general_agent",
    ]
    check(len(profiles) == 9, f"9 profiles (got {len(profiles)})")
    for aid in expected:
        check(aid in profiles, f"  {aid} present")
    return profiles


# ── Test 2: Mandatory fields ──────────────────────────────────────

def test_profile_fields(profiles: dict):
    print("\n── Test 2: Each profile has required fields ──")
    for aid, p in profiles.items():
        check(len(p.agent_id) > 0, f"{aid}: agent_id")
        check(len(p.display_name) > 0, f"{aid}: display_name='{p.display_name}'")
        check(len(p.description) > 0, f"{aid}: description")
        check(len(p.system_prompt) > 200, f"{aid}: system_prompt ({len(p.system_prompt)} chars)")
        check(len(p.allowed_tools) >= 1, f"{aid}: allowed_tools={len(p.allowed_tools)}")
        check(len(p.memory_types) >= 1, f"{aid}: memory_types={len(p.memory_types)}")
        check(len(p.task_types) >= 1, f"{aid}: task_types={len(p.task_types)}")
        # No-fabrication rule in every prompt
        check("不要编造" in p.system_prompt or "do not fabricate" in p.system_prompt.lower(),
              f"{aid}: no-fabrication in system_prompt")


# ── Test 3: Agent-specific expectations ───────────────────────────

def test_agent_specifics(profiles: dict):
    print("\n── Test 3: Agent-specific responsibilities ──")

    paper = profiles["paper_agent"]
    check("rag_retriever" in paper.allowed_tools, "paper_agent: rag_retriever")
    check("paper_reading" in paper.task_types, "paper_agent: paper_reading task")

    experiment = profiles["experiment_agent"]
    check("csv_analyzer" in experiment.allowed_tools, "experiment_agent: csv_analyzer")
    check("jsonl_analyzer" in experiment.allowed_tools, "experiment_agent: jsonl_analyzer")

    claim = profiles["claim_agent"]
    check("claim_support" in claim.allowed_tools, "claim_agent: claim_support tool")
    check("claim_support" in claim.memory_types, "claim_agent: claim_support memory")

    report = profiles["report_agent"]
    check("report_writer" in report.allowed_tools, "report_agent: report_writer")

    memory = profiles["memory_agent"]
    check(len(memory.memory_types) >= 10, f"memory_agent: >=10 memory types ({len(memory.memory_types)})")
    check("memory_consolidation" in memory.allowed_tools, "memory_agent: memory_consolidation")
    check("private" in memory.memory_scopes, "memory_agent: private scope")
    check("global" in memory.memory_scopes, "memory_agent: global scope")

    coordinator = profiles["coordinator_agent"]
    check(len(coordinator.handoff_targets) >= 6, f"coordinator: >=6 handoff targets ({len(coordinator.handoff_targets)})")

    general = profiles["general_agent"]
    check("coordinator_agent" in general.handoff_targets, "general_agent: handoff to coordinator")


# ── Test 4: select_agent_for_task ──────────────────────────────────

def test_agent_selection():
    print("\n── Test 4: select_agent_for_task ──")

    tests = [
        ("paper_question", "请总结这篇论文的方法部分", "paper_agent"),
        ("experiment_analysis", "分析 sample_metrics.csv 的指标", "experiment_agent"),
        ("claim_support", "帮我找证据支持这个论点", "claim_agent"),
        ("report_generation", "生成组会汇报的 PPT 文案", "report_agent"),
        ("progress_memory", "总结上次组会的进展", "progress_agent"),
        ("code_question", "这个 ModuleNotFoundError 怎么修", "code_agent"),
        ("memory_query", "我之前做了哪些实验", "memory_agent"),
        ("general", "今天天气怎么样", "general_agent"),
    ]

    for task_type, query, expected in tests:
        result = select_agent_for_task(task_type, query=query)
        agent_id = result["agent_id"]
        confidence = result["confidence"]
        check(agent_id == expected,
              f"task='{task_type}' → {expected} (got {agent_id}, confidence={confidence:.1f})")

    # Preferred agent overrides
    result = select_agent_for_task(
        "experiment_analysis",
        query="分析 CSV",
        preferred_agent="paper_agent",
    )
    check(result["agent_id"] == "paper_agent", "preferred_agent overrides task_type")
    check(result["confidence"] == 1.0, "preferred_agent confidence=1.0")

    # Unknown task_type falls back to keyword
    result2 = select_agent_for_task("unknown_type", query="帮我看看这个论文")
    check(result2["agent_id"] == "paper_agent" or result2["agent_id"] == "general_agent",
          f"unknown task → keyword fallback (got {result2['agent_id']})")

    # Empty everything falls back to general
    result3 = select_agent_for_task("unknown_type", query="")
    check(result3["agent_id"] == "general_agent",
          f"empty → general_agent (got {result3['agent_id']})")


# ── Test 5: Convenience accessors ──────────────────────────────────

def test_convenience_accessors():
    print("\n── Test 5: Convenience accessors ──")

    # get_agent_profile
    p = get_agent_profile("paper_agent")
    check(p is not None, "get_agent_profile returns profile")
    check(p.agent_id == "paper_agent", "  correct agent_id")

    p_none = get_agent_profile("nonexistent")
    check(p_none is None, "get_agent_profile returns None for invalid id")

    # list_agent_profiles
    all_p = list_agent_profiles()
    check(len(all_p) == 9, f"list_agent_profiles returns 9 (got {len(all_p)})")

    # get_accessible_memory_types
    mem_types = get_accessible_memory_types("experiment_agent")
    check(len(mem_types) >= 3, f"experiment_agent: >=3 memory types ({len(mem_types)})")
    check("experiment_result" in mem_types, "experiment_agent: experiment_result memory")

    mem_types_bad = get_accessible_memory_types("nonexistent")
    check(mem_types_bad == [], "invalid agent returns empty memory types")

    # get_allowed_tools
    tools = get_allowed_tools("report_agent")
    check(len(tools) >= 3, f"report_agent: >=3 tools ({len(tools)})")
    check("report_writer" in tools, "report_agent: report_writer tool")

    tools_bad = get_allowed_tools("nonexistent")
    check(tools_bad == [], "invalid agent returns empty tools")


# ── Test 6: build_agent_system_prompt ──────────────────────────────

def test_system_prompt_builder():
    print("\n── Test 6: build_agent_system_prompt ──")

    prompt = build_agent_system_prompt("claim_agent", task_context="用户论点：LVLM 存在对象幻觉")
    check(len(prompt) > 300, f"prompt non-trivial ({len(prompt)} chars)")
    check("不要编造" in prompt, "prompt contains no-fabrication rule")
    check("Claim Supporter" in prompt or "论点" in prompt, "prompt contains agent identity")
    check("用户论点" in prompt, "prompt contains task_context")

    # Invalid agent
    prompt_bad = build_agent_system_prompt("nonexistent")
    check("不要编造" in prompt_bad, "invalid agent returns no-fabrication rules only")

    # Without task context
    prompt_noctx = build_agent_system_prompt("code_agent")
    check("Code Assistant" in prompt_noctx or "代码" in prompt_noctx,
          "prompt without context works")


# ── Test 7: Profile immutability check ─────────────────────────────

def test_profile_to_dict():
    print("\n── Test 7: AgentProfile.to_dict() ──")

    p = get_agent_profile("coordinator_agent")
    d = p.to_dict()
    check(isinstance(d, dict), "to_dict returns dict")
    check(d["agent_id"] == "coordinator_agent", "  agent_id preserved")
    check(isinstance(d["responsibilities"], list), "  responsibilities is list")
    check(isinstance(d["allowed_tools"], list), "  allowed_tools is list")
    check(isinstance(d["memory_types"], list), "  memory_types is list")


# ── Main ──────────────────────────────────────────────────────────

def main():
    global PASS, FAIL

    print("=" * 60)
    print("  Multi-Agent Profiles Test Suite")
    print("=" * 60)

    profiles = test_profile_count()
    test_profile_fields(profiles)
    test_agent_specifics(profiles)
    test_agent_selection()
    test_convenience_accessors()
    test_system_prompt_builder()
    test_profile_to_dict()

    print(f"\n{'=' * 60}")
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")

    if FAIL > 0:
        print("\nSome tests FAILED.")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
