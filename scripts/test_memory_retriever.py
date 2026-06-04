"""
Test Memory Retriever v1.

Creates a temporary in-memory dataset with 12 diverse memory records,
then runs retrieval queries for each filter dimension.
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from research_agent.memory.retriever import (
    load_memories,
    retrieve_memories,
    retrieve_from_store,
)

# ── Build test dataset ────────────────────────────────────────────

_now = datetime(2026, 6, 5, 12, 0, 0, tzinfo=timezone.utc)


def _ts(offset_hours: int = 0) -> str:
    return (_now + timedelta(hours=offset_hours)).isoformat()


SAMPLE_RECORDS = [
    {
        "memory_id": "mem_20260605_a001",
        "memory_level": "long_term",
        "memory_scope": "shared",
        "memory_type": "research_direction",
        "owner_agent": "coordinator",
        "shared_with": ["paper_agent", "experiment_agent", "report_agent"],
        "content": "多模态幻觉研究是本项目的核心方向，重点关注 LVLM 在图像描述中的对象幻觉问题。",
        "summary": "核心研究方向：LVLM 对象幻觉",
        "tags": ["研究方向", "hallucination", "LVLM"],
        "source_module": "manual",
        "source_path": "",
        "source_title": "项目研究计划",
        "created_at": _ts(-720),
        "updated_at": _ts(-24),
        "importance": 5,
        "status": "active",
        "visibility": "internal",
        "metadata": {},
    },
    {
        "memory_id": "mem_20260605_a002",
        "memory_level": "long_term",
        "memory_scope": "shared",
        "memory_type": "paper_note",
        "owner_agent": "paper_agent",
        "shared_with": ["claim_agent", "report_agent"],
        "content": "Guardrail-Agnostic 论文提出了不依赖安全护栏的偏见评估方法，使用间接任务检测 LVLM 中的社会偏见。",
        "summary": "Guardrail-Agnostic 偏见评估方法",
        "tags": ["bias", "guardrail", "LVLM", "论文"],
        "source_module": "paper_reading",
        "source_path": "data/papers/guardrail_agnostic.md",
        "source_title": "Guardrail-Agnostic Societal Bias Evaluation",
        "created_at": _ts(-500),
        "updated_at": _ts(-100),
        "importance": 4,
        "status": "active",
        "visibility": "internal",
        "metadata": {"arxiv_id": "2401.xxxxx"},
    },
    {
        "memory_id": "mem_20260605_a003",
        "memory_level": "mid_term",
        "memory_scope": "private",
        "memory_type": "experiment_result",
        "owner_agent": "experiment_agent",
        "shared_with": [],
        "content": "coco_val_n300_g1 实验完成。hrs_v1 均值 0.33，top20 高风险图像已筛选。",
        "summary": "coco_val_n300_g1 实验结果",
        "tags": ["实验", "COCO", "hrs_v1", "coco_val_n300_g1"],
        "source_module": "experiment_tool",
        "source_path": "data/experiments/coco_val_n300_g1.md",
        "source_title": "COCO Validation Hallucination Screening n300 g1",
        "created_at": _ts(-200),
        "updated_at": _ts(-50),
        "importance": 4,
        "status": "active",
        "visibility": "internal",
        "metadata": {"run_tag": "coco_val_n300_g1", "hrs_v1_mean": 0.33},
    },
    {
        "memory_id": "mem_20260605_a004",
        "memory_level": "long_term",
        "memory_scope": "shared",
        "memory_type": "research_direction",
        "owner_agent": "coordinator",
        "shared_with": ["paper_agent", "progress_agent"],
        "content": "组会 PPT 积累机制：每次组会后将 PPT 内容拆成 Slide Memory 录入知识库，实现长期科研记忆。",
        "summary": "PPT 作为长期科研记忆",
        "tags": ["PPT", "组会", "知识管理", "科研记忆"],
        "source_module": "ppt_progress",
        "source_path": "",
        "source_title": "组会 PPT 积累方案",
        "created_at": _ts(-600),
        "updated_at": _ts(-10),
        "importance": 3,
        "status": "active",
        "visibility": "internal",
        "metadata": {},
    },
    {
        "memory_id": "mem_20260605_a005",
        "memory_level": "mid_term",
        "memory_scope": "private",
        "memory_type": "todo",
        "owner_agent": "experiment_agent",
        "shared_with": [],
        "content": "下周组会前完成 stereotype library 第一版构建，并跑通 sd35_gender_swap 对比实验。",
        "summary": "待办：stereotype library + sd35 实验",
        "tags": ["待办", "stereotype", "sd35"],
        "source_module": "chat",
        "source_path": "",
        "source_title": "",
        "created_at": _ts(-48),
        "updated_at": _ts(-48),
        "importance": 4,
        "status": "active",
        "visibility": "private",
        "metadata": {},
    },
    {
        "memory_id": "mem_20260605_a006",
        "memory_level": "short_term",
        "memory_scope": "private",
        "memory_type": "general_note",
        "owner_agent": "general_agent",
        "shared_with": [],
        "content": "刚才用 MinerU 解析了一篇关于 VLM safety 的新论文，摘要提到 guardrail-agnostic evaluation。",
        "summary": "临时笔记：VLM safety 论文",
        "tags": ["临时", "MinerU", "VLM safety"],
        "source_module": "paper_reading",
        "source_path": "",
        "source_title": "",
        "created_at": _ts(-2),
        "updated_at": _ts(-2),
        "importance": 2,
        "status": "active",
        "visibility": "private",
        "metadata": {},
    },
    {
        "memory_id": "mem_20260605_a007",
        "memory_level": "long_term",
        "memory_scope": "global",
        "memory_type": "project_decision",
        "owner_agent": "coordinator",
        "shared_with": [],
        "content": "决定采用 RAG + Hybrid Search + Reranker 作为默认检索架构，替代纯 vector search。",
        "summary": "架构决策：Hybrid RAG",
        "tags": ["架构", "RAG", "hybrid", "决策"],
        "source_module": "manual",
        "source_path": "",
        "source_title": "RAG 架构决策",
        "created_at": _ts(-400),
        "updated_at": _ts(-400),
        "importance": 5,
        "status": "active",
        "visibility": "internal",
        "metadata": {},
    },
    {
        "memory_id": "mem_20260605_a008",
        "memory_level": "long_term",
        "memory_scope": "global",
        "memory_type": "user_preference",
        "owner_agent": "user",
        "shared_with": [],
        "content": "用户偏好中文回答，报告格式偏好 Markdown，组会 PPT 偏好简洁要点式。",
        "summary": "语言和格式偏好",
        "tags": ["偏好", "中文", "Markdown", "PPT"],
        "source_module": "chat",
        "source_path": "",
        "source_title": "",
        "created_at": _ts(-800),
        "updated_at": _ts(-30),
        "importance": 3,
        "status": "active",
        "visibility": "private",
        "metadata": {},
    },
    {
        "memory_id": "mem_20260605_a009",
        "memory_level": "long_term",
        "memory_scope": "shared",
        "memory_type": "paper_note",
        "owner_agent": "paper_agent",
        "shared_with": ["experiment_agent", "claim_agent"],
        "content": "CHAIR 指标是评估图像描述幻觉的标准方法，但仅限于对象幻觉。需要结合属性幻觉和关系幻觉指标。",
        "summary": "CHAIR 指标局限",
        "tags": ["CHAIR", "hallucination", "指标", "论文"],
        "source_module": "paper_reading",
        "source_path": "data/papers/chair_metric.md",
        "source_title": "CHAIR: Evaluating Hallucination in Image Captioning",
        "created_at": _ts(-300),
        "updated_at": _ts(-50),
        "importance": 4,
        "status": "active",
        "visibility": "internal",
        "metadata": {},
    },
    {
        "memory_id": "mem_20260605_a010",
        "memory_level": "mid_term",
        "memory_scope": "private",
        "memory_type": "code_note",
        "owner_agent": "code_agent",
        "shared_with": [],
        "content": "chunker.py 的 _should_avoid_merge 函数针对 slide 文档做了特殊处理，防止合并破坏 slide 边界。",
        "summary": "chunker.py slide 边界处理",
        "tags": ["代码", "chunker", "slide"],
        "source_module": "code_assistant",
        "source_path": "src/research_agent/rag/chunker.py",
        "source_title": "chunker.py — Markdown-aware Chunker",
        "created_at": _ts(-100),
        "updated_at": _ts(-10),
        "importance": 2,
        "status": "active",
        "visibility": "private",
        "metadata": {"file": "chunker.py", "function": "_should_avoid_merge"},
    },
    {
        "memory_id": "mem_20260605_a011",
        "memory_level": "short_term",
        "memory_scope": "private",
        "memory_type": "issue",
        "owner_agent": "experiment_agent",
        "shared_with": [],
        "content": "sd35_gender_swap 实验的 mean_extra_object_rate 异常偏高（0.15），需要排查是否解析脚本有误。",
        "summary": "sd35_gender_swap 指标异常",
        "tags": ["bug", "sd35", "实验指标"],
        "source_module": "experiment_tool",
        "source_path": "data/experiments/sd35_gender_swap.md",
        "source_title": "SD3.5 Gender Swap Experiment",
        "created_at": _ts(-5),
        "updated_at": _ts(-5),
        "importance": 4,
        "status": "active",
        "visibility": "private",
        "metadata": {},
    },
    {
        "memory_id": "mem_20260605_a012",
        "memory_level": "long_term",
        "memory_scope": "shared",
        "memory_type": "meeting_note",
        "owner_agent": "progress_agent",
        "shared_with": ["coordinator", "report_agent"],
        "content": "2026-06-04 组会纪要：讨论了 RAG v2 的 hybrid mode 切换方案，决定默认使用 vector，hybrid 作为实验性 feature。",
        "summary": "0604 组会：RAG v2 切换方案",
        "tags": ["组会", "RAG", "决策"],
        "source_module": "ppt_progress",
        "source_path": "",
        "source_title": "2026-06-04 组会纪要",
        "created_at": _ts(-24),
        "updated_at": _ts(-24),
        "importance": 3,
        "status": "active",
        "visibility": "internal",
        "metadata": {},
    },
    # Archived/expired record — should be retrievable by status filter
    {
        "memory_id": "mem_20260605_a013",
        "memory_level": "short_term",
        "memory_scope": "private",
        "memory_type": "general_note",
        "owner_agent": "general_agent",
        "shared_with": [],
        "content": "已过期的旧笔记：最初使用纯 BM25 做 keyword search，后改用 token overlap。",
        "summary": "旧方案：BM25 尝试",
        "tags": ["旧笔记", "BM25", "已废弃"],
        "source_module": "chat",
        "source_path": "",
        "source_title": "",
        "created_at": _ts(-900),
        "updated_at": _ts(-900),
        "importance": 1,
        "status": "expired",
        "visibility": "private",
        "metadata": {},
    },
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


# ── Helper: write sample data to temp file ────────────────────────

def _write_temp_jsonl() -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for rec in SAMPLE_RECORDS:
        tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.close()
    return Path(tmp.name)


# ── Tests ─────────────────────────────────────────────────────────


def test_query_by_memory_type():
    print("\n── Test 1: query by memory_type ──")
    results = retrieve_memories(SAMPLE_RECORDS, memory_type="research_direction")
    check(len(results) == 2, f"2 research_direction records (got {len(results)})")
    for r in results:
        check(r["memory_type"] == "research_direction", f"  {r['memory_id']} is research_direction")

    results2 = retrieve_memories(SAMPLE_RECORDS, memory_type="paper_note")
    check(len(results2) == 2, f"2 paper_note records (got {len(results2)})")

    results3 = retrieve_memories(SAMPLE_RECORDS, memory_type="nonexistent")
    check(len(results3) == 0, "0 results for nonexistent type")


def test_query_by_tag():
    print("\n── Test 2: query by tag (OR logic) ──")
    results = retrieve_memories(SAMPLE_RECORDS, tags=["hallucination"])
    # Should match a001 (tags: hallucination), a009 (tags: hallucination)
    check(len(results) >= 2, f">= 2 records tagged 'hallucination' (got {len(results)})")
    for r in results:
        has_tag = any("hallucination" in t.lower() for t in r.get("tags", []))
        check(has_tag, f"  {r['memory_id']} has hallucination tag")

    # Multiple tags (OR)
    results2 = retrieve_memories(SAMPLE_RECORDS, tags=["CHAIR", "BM25"])
    # Should match a009 (CHAIR tag) and a013 (BM25 tag)
    check(len(results2) >= 2, f">= 2 records for tags CHAIR/BM25 (got {len(results2)})")

    # Tag not present
    results3 = retrieve_memories(SAMPLE_RECORDS, tags=["nonexistent_tag_xyz"])
    check(len(results3) == 0, "0 results for nonexistent tag")


def test_query_by_source_module():
    print("\n── Test 3: query by source_module ──")
    results = retrieve_memories(SAMPLE_RECORDS, source_module="paper_reading")
    # a002, a006, a009
    check(len(results) == 3, f"3 paper_reading records (got {len(results)})")
    for r in results:
        check(r["source_module"] == "paper_reading", f"  {r['memory_id']}")

    results2 = retrieve_memories(SAMPLE_RECORDS, source_module="experiment_tool")
    check(len(results2) >= 2, f">= 2 experiment_tool records (got {len(results2)})")


def test_query_by_keyword():
    print("\n── Test 4: query by keyword (case-insensitive) ──")
    # "hallucination" appears in source_title of a003, and tags/content of a009
    results = retrieve_memories(SAMPLE_RECORDS, keyword="hallucination")
    check(len(results) >= 2, f">= 2 records mentioning 'hallucination' (got {len(results)})")

    # Case-insensitive
    results2 = retrieve_memories(SAMPLE_RECORDS, keyword="CHAIR")
    check(len(results2) >= 1, f">= 1 record mentioning 'CHAIR' (got {len(results2)})")
    results2b = retrieve_memories(SAMPLE_RECORDS, keyword="chair")
    check(len(results2b) == len(results2), "case-insensitive match")

    # Chinese keyword
    results3 = retrieve_memories(SAMPLE_RECORDS, keyword="组会")
    check(len(results3) >= 2, f">= 2 records mentioning '组会' (got {len(results3)})")

    # No match
    results4 = retrieve_memories(SAMPLE_RECORDS, keyword="xyzzy_not_present")
    check(len(results4) == 0, "0 results for unknown keyword")


def test_query_by_time_range():
    print("\n── Test 5: query by time range ──")

    # Records created in the last 100 hours
    recent = retrieve_memories(
        SAMPLE_RECORDS,
        created_after=(_now - timedelta(hours=100)).isoformat(),
    )
    check(len(recent) >= 5, f">= 5 records in last 100h (got {len(recent)})")

    # Records created before 400h ago
    old = retrieve_memories(
        SAMPLE_RECORDS,
        created_before=(_now - timedelta(hours=400)).isoformat(),
    )
    check(len(old) >= 4, f">= 4 records older than 400h (got {len(old)})")

    # Records updated in the last 30h
    recent_update = retrieve_memories(
        SAMPLE_RECORDS,
        updated_after=(_now - timedelta(hours=30)).isoformat(),
    )
    check(len(recent_update) >= 3, f">= 3 records updated in last 30h (got {len(recent_update)})")


def test_query_by_importance():
    print("\n── Test 6: query by importance range ──")

    high = retrieve_memories(SAMPLE_RECORDS, importance_min=4)
    check(len(high) >= 6, f">= 6 records with importance >= 4 (got {len(high)})")

    low = retrieve_memories(SAMPLE_RECORDS, importance_max=2)
    check(len(low) >= 2, f">= 2 records with importance <= 2 (got {len(low)})")

    exact = retrieve_memories(SAMPLE_RECORDS, importance_min=5, importance_max=5)
    check(len(exact) == 2, f"2 records with importance == 5 (got {len(exact)})")


def test_query_by_status():
    print("\n── Test 7: query by status ──")

    active = retrieve_memories(SAMPLE_RECORDS, status="active")
    check(len(active) == 12, f"12 active records (got {len(active)})")

    expired = retrieve_memories(SAMPLE_RECORDS, status="expired")
    check(len(expired) == 1, f"1 expired record (got {len(expired)})")


def test_query_by_owner_agent():
    print("\n── Test 8: query by owner_agent ──")

    results = retrieve_memories(SAMPLE_RECORDS, owner_agent="coordinator")
    check(len(results) == 3, f"3 coordinator records (got {len(results)})")

    results2 = retrieve_memories(SAMPLE_RECORDS, owner_agent="experiment_agent")
    check(len(results2) == 3, f"3 experiment_agent records (got {len(results2)})")


def test_query_combined():
    print("\n── Test 9: combined filters ──")

    # long_term + paper_note + tagged bias
    results = retrieve_memories(
        SAMPLE_RECORDS,
        memory_level="long_term",
        memory_type="paper_note",
        tags=["bias"],
    )
    check(len(results) == 1, f"1 long_term paper_note with bias tag (got {len(results)})")
    if results:
        check(results[0]["memory_id"] == "mem_20260605_a002", "  correct record")

    # high importance + keyword
    results2 = retrieve_memories(
        SAMPLE_RECORDS,
        importance_min=4,
        keyword="coco",
    )
    check(len(results2) >= 1, f">= 1 high-importance record mentioning coco")

    # source_module + time + status
    results3 = retrieve_memories(
        SAMPLE_RECORDS,
        source_module="experiment_tool",
        created_after=(_now - timedelta(hours=300)).isoformat(),
        status="active",
    )
    check(len(results3) >= 1, f">= 1 active experiment_tool record in time range")


def test_limit():
    print("\n── Test 10: limit ──")

    all_results = retrieve_memories(SAMPLE_RECORDS, memory_level="long_term")
    check(len(all_results) >= 6, f">= 6 long_term records total")

    limited = retrieve_memories(SAMPLE_RECORDS, memory_level="long_term", limit=3)
    check(len(limited) == 3, "limit=3 returns exactly 3")


def test_jsonl_file_roundtrip():
    print("\n── Test 11: JSONL file round-trip ──")

    tmp_path = _write_temp_jsonl()
    try:
        loaded = load_memories(tmp_path)
        check(len(loaded) == len(SAMPLE_RECORDS),
              f"load_memories reads {len(loaded)} records (expected {len(SAMPLE_RECORDS)})")

        # retrieve_from_store convenience
        results = retrieve_from_store(
            tmp_path,
            memory_type="paper_note",
        )
        check(len(results) == 2, f"retrieve_from_store finds 2 paper_notes (got {len(results)})")
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Main ──────────────────────────────────────────────────────────

def main():
    global PASS, FAIL

    print("=" * 60)
    print("  Memory Retriever v1 Test Suite")
    print(f"  Dataset: {len(SAMPLE_RECORDS)} records")
    print("=" * 60)

    test_query_by_memory_type()
    test_query_by_tag()
    test_query_by_source_module()
    test_query_by_keyword()
    test_query_by_time_range()
    test_query_by_importance()
    test_query_by_status()
    test_query_by_owner_agent()
    test_query_combined()
    test_limit()
    test_jsonl_file_roundtrip()

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
