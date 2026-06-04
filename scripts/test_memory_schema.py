"""
Test Memory Schema — creation, validation, serialisation round-trips.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.memory.schema import (
    # Types
    MemoryRecord,
    MemoryLevel,
    MemoryScope,
    AgentRole,
    MemoryType,
    SourceModule,
    # Factory
    create_memory_record,
    # Validation
    validate_memory_record,
    # Serialisation
    memory_record_to_dict,
    memory_record_from_dict,
    memory_record_to_jsonl,
    memory_record_from_jsonl,
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


# ── Test 1: long_term shared research_direction ───────────────────

def test_long_term_shared():
    print("\n── Test 1: long_term shared research_direction ──")

    rec = create_memory_record(
        content="多模态幻觉研究是本项目的核心方向，重点关注 LVLM 在图像描述中的对象幻觉问题。",
        memory_level="long_term",
        memory_scope="shared",
        memory_type="research_direction",
        owner_agent="coordinator",
        source_module="manual",
        tags=["研究方向", "hallucination", "LVLM"],
        importance=5,
    )

    check(rec.memory_id.startswith("mem_"), "memory_id starts with mem_")
    check(rec.memory_level == "long_term", "memory_level = long_term")
    check(rec.memory_scope == "shared", "memory_scope = shared")
    check(rec.memory_type == "research_direction", "memory_type = research_direction")
    check(rec.owner_agent == "coordinator", "owner_agent = coordinator")
    check(rec.importance == 5, "importance = 5")
    check(len(rec.content) > 0, "content non-empty")
    check(len(rec.summary) > 0, "summary auto-generated")
    check(len(rec.created_at) > 0, "created_at populated")
    check(len(rec.memory_id) > 20, f"memory_id looks valid ({rec.memory_id})")

    errors = validate_memory_record(rec)
    check(len(errors) == 0, f"no validation errors (got {errors})")

    return rec


# ── Test 2: mid_term private todo ─────────────────────────────────

def test_mid_term_private():
    print("\n── Test 2: mid_term private todo ──")

    rec = create_memory_record(
        content="下周组会前完成 stereotype library 第一版构建，并跑通 coco_val_n300_g1 的对比实验。",
        memory_level="mid_term",
        memory_scope="private",
        memory_type="todo",
        owner_agent="experiment_agent",
        source_module="chat",
        tags=["待办", "stereotype", "coco_val_n300_g1"],
        importance=4,
        status="active",
    )

    check(rec.memory_level == "mid_term", "memory_level = mid_term")
    check(rec.memory_scope == "private", "memory_scope = private")
    check(rec.memory_type == "todo", "memory_type = todo")
    check(rec.status == "active", "status = active")
    check(rec.shared_with == [], "shared_with empty for private")

    errors = validate_memory_record(rec)
    check(len(errors) == 0, f"no validation errors (got {errors})")

    return rec


# ── Test 3: short_term private general_note ───────────────────────

def test_short_term_private():
    print("\n── Test 3: short_term private general_note ──")

    rec = create_memory_record(
        content="刚才用 MinerU 解析了一篇新论文，摘要提到 guardrail-agnostic evaluation。先记下来，后续处理。",
        memory_level="short_term",
        memory_scope="private",
        memory_type="general_note",
        owner_agent="paper_agent",
        source_module="paper_reading",
        tags=["临时", "MinerU", "guardrail-agnostic"],
        importance=2,
    )

    check(rec.memory_level == "short_term", "memory_level = short_term")
    check(rec.memory_type == "general_note", "memory_type = general_note")

    errors = validate_memory_record(rec)
    check(len(errors) == 0, f"no validation errors (got {errors})")

    return rec


# ── Test 4: invalid memory_level ──────────────────────────────────

def test_invalid_fields():
    print("\n── Test 4: invalid fields detected ──")

    # Invalid memory_level
    rec = create_memory_record(
        content="test",
        memory_level="ultra_term",  # type: ignore — deliberately wrong
        memory_scope="private",
        memory_type="general_note",
        owner_agent="general_agent",
    )
    errors = validate_memory_record(rec)
    check(len(errors) >= 1, f"invalid memory_level detected ({errors[0][:60]}...)")

    # Invalid importance
    rec2 = create_memory_record(
        content="test",
        memory_level="short_term",
        memory_scope="private",
        memory_type="general_note",
        owner_agent="general_agent",
        importance=99,
    )
    errors2 = validate_memory_record(rec2)
    check(any("importance" in e for e in errors2), f"invalid importance detected")

    # Empty content
    rec3 = create_memory_record(
        content="",
        memory_level="short_term",
        memory_scope="private",
        memory_type="general_note",
        owner_agent="general_agent",
    )
    errors3 = validate_memory_record(rec3)
    check(any("content" in e.lower() for e in errors3), f"empty content detected")


# ── Test 5: dict round-trip ───────────────────────────────────────

def test_dict_roundtrip():
    print("\n── Test 5: dict round-trip ──")

    rec = create_memory_record(
        content="Guardrail-Agnostic 论文提出了不依赖安全护栏的偏见评估方法。",
        memory_level="long_term",
        memory_scope="shared",
        memory_type="paper_note",
        owner_agent="paper_agent",
        source_module="paper_reading",
        source_path="data/papers/guardrail_agnostic.md",
        source_title="Guardrail-Agnostic Societal Bias Evaluation",
        tags=["bias", "guardrail", "LVLM"],
        importance=4,
        shared_with=["claim_agent", "report_agent"],
        metadata={"arxiv_id": "2401.xxxxx", "venue": "CVPR 2026"},
    )

    d = memory_record_to_dict(rec)
    check(isinstance(d, dict), "to_dict returns dict")
    check(d["memory_id"] == rec.memory_id, "dict memory_id matches")
    check(d["metadata"]["arxiv_id"] == "2401.xxxxx", "nested metadata preserved")
    check("claim_agent" in d["shared_with"], "shared_with preserved in dict")

    rec2 = memory_record_from_dict(d)
    check(isinstance(rec2, MemoryRecord), "from_dict returns MemoryRecord")
    check(rec2.memory_id == rec.memory_id, "round-trip: memory_id")
    check(rec2.content == rec.content, "round-trip: content")
    check(rec2.memory_type == rec.memory_type, "round-trip: memory_type")
    check(rec2.metadata == rec.metadata, "round-trip: metadata")
    check(rec2.shared_with == rec.shared_with, "round-trip: shared_with")


# ── Test 6: jsonl round-trip ──────────────────────────────────────

def test_jsonl_roundtrip():
    print("\n── Test 6: jsonl round-trip ──")

    rec = create_memory_record(
        content="反事实样本可有效检测 LVLM 中的性别偏差，实验结果见 coco_val_n300_g1。",
        memory_level="mid_term",
        memory_scope="shared",
        memory_type="experiment_result",
        owner_agent="experiment_agent",
        source_module="experiment_tool",
        tags=["反事实", "偏差", "COCO"],
        importance=3,
    )

    line = memory_record_to_jsonl(rec)
    check(isinstance(line, str), "to_jsonl returns str")
    check("\n" not in line, "jsonl is single line")
    check("反事实" in line, "CJK characters preserved in jsonl")

    rec2 = memory_record_from_jsonl(line)
    check(isinstance(rec2, MemoryRecord), "from_jsonl returns MemoryRecord")
    check(rec2.memory_id == rec.memory_id, "jsonl round-trip: memory_id")
    check(rec2.content == rec.content, "jsonl round-trip: content")
    check(rec2.tags == rec.tags, "jsonl round-trip: tags")


# ── Test 7: summary auto-generation ───────────────────────────────

def test_summary_autogen():
    print("\n── Test 7: summary auto-generation ──")

    long_content = (
        "CHAIR 指标是评估图像描述中幻觉现象的常用方法，"
        "它通过对比模型生成的描述与 ground truth 对象标注来度量对象幻觉率。"
        "然而 CHAIR 仅限于对象幻觉，无法捕捉属性幻觉和关系幻觉。"
    )

    rec = create_memory_record(
        content=long_content,
        memory_level="mid_term",
        memory_scope="private",
        memory_type="general_note",
        owner_agent="general_agent",
        summary="",  # should auto-generate
    )

    check(len(rec.summary) <= 120, f"summary <= 120 chars (got {len(rec.summary)})")
    check(rec.summary == long_content[:120].replace("\n", " ").strip(),
          "summary = first 120 chars of content")

    # Explicit summary
    rec2 = create_memory_record(
        content=long_content,
        memory_level="mid_term",
        memory_scope="private",
        memory_type="general_note",
        owner_agent="general_agent",
        summary="CHAIR 指标概述及其局限性",
    )
    check(rec2.summary == "CHAIR 指标概述及其局限性", "explicit summary preserved")


# ── Test 8: shared_with list ──────────────────────────────────────

def test_shared_with():
    print("\n── Test 8: shared_with list ──")

    rec = create_memory_record(
        content="本周实验进度：coco_val_n300_g1 已跑完，hrs_v1 均值 0.33。",
        memory_level="mid_term",
        memory_scope="shared",
        memory_type="progress_update",
        owner_agent="experiment_agent",
        source_module="experiment_tool",
        shared_with=["coordinator", "report_agent", "progress_agent"],
    )

    check(len(rec.shared_with) == 3, "shared_with has 3 entries")
    check("coordinator" in rec.shared_with, "coordinator in shared_with")
    check("report_agent" in rec.shared_with, "report_agent in shared_with")

    # Default empty
    rec2 = create_memory_record(
        content="private note",
        memory_level="short_term",
        memory_scope="private",
        memory_type="general_note",
        owner_agent="general_agent",
    )
    check(rec2.shared_with == [], "private record has empty shared_with")


# ── Main ──────────────────────────────────────────────────────────

def main():
    global PASS, FAIL

    print("=" * 60)
    print("  Memory Schema v1 Test Suite")
    print("=" * 60)

    test_long_term_shared()
    test_mid_term_private()
    test_short_term_private()
    test_invalid_fields()
    test_dict_roundtrip()
    test_jsonl_roundtrip()
    test_summary_autogen()
    test_shared_with()

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
