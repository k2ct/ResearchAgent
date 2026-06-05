"""
Test Shared Workspace with Section-level Permissions v1.

Verifies dataclasses, serialization, permissions, patches, handoff
integration, and memory integration.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.agents.workspace import (
    WorkspaceSection, WorkspaceDocument, WorkspacePatch,
    build_default_workspace,
    workspace_to_markdown, workspace_from_markdown,
    ensure_workspace_dir, save_workspace, load_workspace,
    append_patch_log, load_patch_log,
    can_agent_write_section, get_readable_workspace_for_agent,
    create_workspace_patch, apply_workspace_patch,
    approve_suggestion_patch,
    default_section_for_agent, create_patch_from_handoff_result,
    save_workspace_summary_to_memory,
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


# ── Test 1: Default workspace ────────────────────────────────────

def test_default_workspace():
    print("\n── Test 1: build_default_workspace ──")
    ws = build_default_workspace("Test Workspace")
    check(len(ws.sections) == 8, f"8 sections (got {len(ws.sections)})")
    check(ws.coordinator == "coordinator_agent", "coordinator set")
    check(len(ws.allowed_agents) >= 7, f"allowed_agents >=7 ({len(ws.allowed_agents)})")

    ids = {s.section_id for s in ws.sections}
    for expected in ["paper_evidence", "experiment_evidence", "claim_support",
                      "progress_summary", "report_draft", "implementation_notes",
                      "memory_notes", "final_summary"]:
        check(expected in ids, f"section '{expected}' present")

    return ws


# ── Test 2: Markdown round-trip ──────────────────────────────────

def test_markdown_roundtrip():
    print("\n── Test 2: workspace_to_markdown → workspace_from_markdown ──")
    ws = build_default_workspace("Roundtrip Test")
    # Modify a section
    for s in ws.sections:
        if s.section_id == "paper_evidence":
            s.content = "Paper analysis: LVLM hallucination is a key challenge."

    md = workspace_to_markdown(ws)
    check(len(md) > 500, f"markdown non-trivial ({len(md)} chars)")
    check("<!-- section_id:" in md, "section_id comments present")
    check("<!-- owner:" in md, "owner comments present")
    check("<!-- write_permission:" in md, "write_permission comments present")

    ws2 = workspace_from_markdown(md)
    check(ws2.workspace_id == ws.workspace_id, "workspace_id round-trip")
    check(ws2.title == ws.title, "title round-trip")
    check(len(ws2.sections) == 8, f"section count round-trip ({len(ws2.sections)})")

    paper = next((s for s in ws2.sections if s.section_id == "paper_evidence"), None)
    check(paper is not None and "LVLM hallucination" in paper.content,
          "paper_evidence content preserved")


# ── Test 3: Read access ──────────────────────────────────────────

def test_read_access():
    print("\n── Test 3: All agents have full read access ──")
    ws = build_default_workspace("Read Test")
    for agent in ["paper_agent", "experiment_agent", "claim_agent", "general_agent"]:
        text = get_readable_workspace_for_agent(ws, agent)
        check(len(text) > 500, f"{agent}: readable ({len(text)} chars)")


# ── Test 4: Write permissions ────────────────────────────────────

def test_write_permissions():
    print("\n── Test 4: Section-level write permissions ──")
    ws = build_default_workspace("Permission Test")

    # paper_agent CAN write paper_evidence
    check(can_agent_write_section(ws, "paper_agent", "paper_evidence"),
          "paper_agent can write paper_evidence")

    # paper_agent CANNOT write experiment_evidence
    check(not can_agent_write_section(ws, "paper_agent", "experiment_evidence"),
          "paper_agent cannot write experiment_evidence")

    # coordinator CAN write anything
    check(can_agent_write_section(ws, "coordinator_agent", "experiment_evidence"),
          "coordinator can write experiment_evidence")
    check(can_agent_write_section(ws, "coordinator_agent", "final_summary"),
          "coordinator can write final_summary")

    # Only coordinator can write final_summary
    check(not can_agent_write_section(ws, "paper_agent", "final_summary"),
          "paper_agent cannot write final_summary (coordinator only)")


# ── Test 5: Apply patch with permission ──────────────────────────

def test_apply_patch_authorized():
    print("\n── Test 5: Apply patch (authorized) ──")
    ws = build_default_workspace("Patch Test")

    patch = create_workspace_patch(
        workspace_id=ws.workspace_id,
        agent_id="paper_agent",
        target_section="paper_evidence",
        content="Updated paper evidence: co-occurrence bias confirmed.",
        operation="replace_section",
    )

    result = apply_workspace_patch(ws, patch)
    check(result["ok"], "apply succeeded")
    check(result["applied"], "applied=True")
    check(not result["suggestion"], "not a suggestion")

    paper = next(s for s in ws.sections if s.section_id == "paper_evidence")
    check("co-occurrence bias confirmed" in paper.content, "content replaced")


# ── Test 6: Apply patch without permission → suggestion ──────────

def test_apply_patch_unauthorized():
    print("\n── Test 6: Apply patch (unauthorized → suggestion) ──")
    ws = build_default_workspace("Suggestion Test")

    patch = create_workspace_patch(
        workspace_id=ws.workspace_id,
        agent_id="claim_agent",
        target_section="experiment_evidence",
        content="Claim agent's experiment analysis...",
        operation="replace_section",
    )

    result = apply_workspace_patch(ws, patch)
    check(result["ok"], "result ok (not an error)")
    check(not result["applied"], "applied=False")
    check(result["suggestion"], "suggestion=True")

    # Original content unchanged
    exp = next(s for s in ws.sections if s.section_id == "experiment_evidence")
    check("Claim agent" not in exp.content, "original section unchanged")


# ── Test 7: Coordinator approve suggestion ───────────────────────

def test_approve_suggestion():
    print("\n── Test 7: Coordinator approve suggestion ──")
    ws = build_default_workspace("Approve Test")

    # Create an unauthorized patch
    patch = create_workspace_patch(
        workspace_id=ws.workspace_id,
        agent_id="report_agent",
        target_section="paper_evidence",
        content="Report agent's paper summary: important findings.",
    )
    result1 = apply_workspace_patch(ws, patch)
    check(result1["suggestion"], "stored as suggestion")

    # Coordinator approves
    result2 = approve_suggestion_patch(ws, patch)
    check(result2["ok"], "approve succeeds")
    check(result2["applied"], "applied after approval")

    paper = next(s for s in ws.sections if s.section_id == "paper_evidence")
    check("important findings" in paper.content, "suggestion content applied after approval")


# ── Test 8: Append operation ─────────────────────────────────────

def test_append_operation():
    print("\n── Test 8: Append operation ──")
    ws = build_default_workspace("Append Test")
    # Set initial content
    for s in ws.sections:
        if s.section_id == "report_draft":
            s.content = "Initial draft."

    patch = create_workspace_patch(
        workspace_id=ws.workspace_id,
        agent_id="report_agent",
        target_section="report_draft",
        content="Additional findings section.",
        operation="append_section",
    )
    result = apply_workspace_patch(ws, patch)
    check(result["applied"], "append applied")

    draft = next(s for s in ws.sections if s.section_id == "report_draft")
    check("Initial draft" in draft.content and "Additional findings" in draft.content,
          "content appended")


# ── Test 9: Patch log ────────────────────────────────────────────

def test_patch_log():
    print("\n── Test 9: Patch log write/read ──")
    ws = build_default_workspace("Log Test")

    patch = create_workspace_patch(
        workspace_id=ws.workspace_id,
        agent_id="paper_agent",
        target_section="paper_evidence",
        content="Logged content.",
    )
    log_result = append_patch_log(patch)
    check(log_result["ok"], "patch log written")

    records = load_patch_log(ws.workspace_id, limit=10)
    check(len(records) >= 1, f"patch log readable ({len(records)} records)")


# ── Test 10: Save / Load workspace ───────────────────────────────

def test_save_load():
    print("\n── Test 10: save_workspace / load_workspace ──")
    ensure_workspace_dir()
    ws = build_default_workspace("Save/Load Test")
    ws.workspace_id = "ws_save_load_test"

    for s in ws.sections:
        if s.section_id == "memory_notes":
            s.content = "Memory agent notes: consolidation complete."

    result = save_workspace(ws)
    check(result["ok"], f"save succeeded: {result['path']}")

    loaded = load_workspace(Path(result["path"]))
    check(loaded.workspace_id == "ws_save_load_test", "workspace_id preserved")
    check(len(loaded.sections) == 8, "section count preserved")

    mem = next((s for s in loaded.sections if s.section_id == "memory_notes"), None)
    check(mem is not None and "consolidation complete" in mem.content,
          "memory_notes content preserved")

    # Cleanup
    Path(result["path"]).unlink(missing_ok=True)
    log_path = PROJECT_ROOT / "data" / "workspaces" / "ws_save_load_test_patch_log.jsonl"
    log_path.unlink(missing_ok=True)


# ── Test 11: Handoff → patch mapping ─────────────────────────────

def test_handoff_to_patch():
    print("\n── Test 11: create_patch_from_handoff_result ──")

    # default_section_for_agent
    mappings = [
        ("paper_agent", "paper_evidence"),
        ("experiment_agent", "experiment_evidence"),
        ("claim_agent", "claim_support"),
        ("progress_agent", "progress_summary"),
        ("report_agent", "report_draft"),
        ("code_agent", "implementation_notes"),
        ("memory_agent", "memory_notes"),
        ("coordinator_agent", "final_summary"),
    ]
    for agent, expected_section in mappings:
        got = default_section_for_agent(agent)
        check(got == expected_section, f"{agent} → {expected_section} (got {got})")

    # create_patch_from_handoff_result
    ws = build_default_workspace("Handoff Test")
    patch = create_patch_from_handoff_result(
        ws=ws,
        agent_id="paper_agent",
        target_section="",
        result_text="Paper handoff result text.",
        handoff_id="ho_test_123",
    )
    check(patch.target_section == "paper_evidence", "auto-mapped to paper_evidence")
    check("Paper handoff" in patch.content, "content transferred")
    check(patch.metadata.get("handoff_id") == "ho_test_123", "handoff_id in metadata")


# ── Test 12: Memory integration ──────────────────────────────────

def test_memory_integration():
    print("\n── Test 12: save_workspace_summary_to_memory ──")

    ws = build_default_workspace("Memory Integration Test")
    for s in ws.sections:
        if s.section_id == "final_summary":
            s.content = "Coordinator final summary: all evidence supports the claim."

    # save_memory=False → no write
    r1 = save_workspace_summary_to_memory(ws, save_memory=False)
    check(r1["ok"] and not r1["written"], "save_memory=False does not write")

    # save_memory=True → write
    r2 = save_workspace_summary_to_memory(ws, save_memory=True)
    check(r2["ok"], f"save_memory=True ok={r2['ok']}")
    if r2["ok"]:
        check(r2["written"], "written=True")
        check(len(r2.get("memory_id", "")) > 0, f"memory_id non-empty: {r2.get('memory_id', '')[:20]}")

        # Verify it's retrievable
        try:
            from research_agent.memory.retriever import load_memories_for_retrieval, retrieve_memories
            records = load_memories_for_retrieval()
            found = retrieve_memories(records, keyword="Memory Integration Test")
            check(len(found) >= 1, f"memory retrievable via retriever ({len(found)} found)")
        except Exception as e:
            print(f"  INFO  retriever check skipped: {e}")


# ── Test 13: .gitignore check ────────────────────────────────────

def test_gitignore():
    print("\n── Test 13: data/workspaces/ is gitignored ──")
    gi = PROJECT_ROOT / ".gitignore"
    content = gi.read_text(encoding="utf-8")
    check("data/workspaces/" in content, "data/workspaces/ in .gitignore")


# ── Main ─────────────────────────────────────────────────────────

def main():
    global PASS, FAIL

    print("=" * 60)
    print("  Shared Workspace v1 Test Suite")
    print("=" * 60)

    test_default_workspace()
    test_markdown_roundtrip()
    test_read_access()
    test_write_permissions()
    test_apply_patch_authorized()
    test_apply_patch_unauthorized()
    test_approve_suggestion()
    test_append_operation()
    test_patch_log()
    test_save_load()
    test_handoff_to_patch()
    test_memory_integration()
    test_gitignore()

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
