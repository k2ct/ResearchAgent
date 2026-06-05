from pathlib import Path
import sys

import streamlit as st


# =========================
# Path setup
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from research_agent.graph.workflow import build_graph


# =========================
# Initial State
# =========================

def create_initial_state(query: str) -> dict:
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

        # Phase 3: Memory-aware
        "memory_context": "",
        "retrieved_memories": [],
        "memory_count": 0,
        "memory_used": False,
        "memory_error": "",

        # Phase 3: Multi-Agent
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


# =========================
# Cache graph
# =========================

@st.cache_resource
def get_graph():
    """
    Build LangGraph only once.

    This avoids repeatedly compiling the graph every time
    Streamlit reruns the script.
    """
    return build_graph()


def run_agent(query: str) -> dict:
    graph = get_graph()
    return graph.invoke(create_initial_state(query))


# =========================
# Page config
# =========================

st.set_page_config(
    page_title="ResearchAgent v0.4 Preview",
    page_icon="🧠",
    layout="wide",
)


# =========================
# Sidebar
# =========================

st.sidebar.title("ResearchAgent v0.4 Preview")

st.sidebar.markdown(
    """
**Current Capabilities**

- LangGraph workflow
- Agentic RAG
- CSV / JSONL tool calling
- Evidence Checker
- Report Writer (Template / LLM-assisted)
- Sources display
"""
)

st.sidebar.markdown("---")

# ── Config status ────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ 运行模式")
import os
col_a, col_b = st.sidebar.columns(2)
with col_a:
    multi_agent = os.getenv("ENABLE_MULTI_AGENT", "false").strip().lower() == "true"
    st.metric("Multi-Agent", "✅ ON" if multi_agent else "⬜ OFF")
with col_b:
    memory_aware = os.getenv("ENABLE_MEMORY_AWARE_AGENT", "false").strip().lower() == "true"
    st.metric("Memory-Aware", "✅ ON" if memory_aware else "⬜ OFF")

st.sidebar.markdown("---")

show_debug = st.sidebar.checkbox("显示 Debug 信息", value=True)

st.sidebar.markdown("### 示例问题")

example_queries = [
    "请分析 data/experiments/sample_metrics.csv",
    "请分析 data/experiments/sample_generations.jsonl",
    "OpenImages-MIAP 的性别标注是图像级还是 bbox 级？",
    "请帮我解释 coco_val_n300_g1 这个实验的目的",
    "我今天应该怎么安排科研任务",
    "ModuleNotFoundError: No module named langgraph 怎么解决",
]

selected_example = st.sidebar.selectbox(
    "选择一个示例问题",
    example_queries,
)


# =========================
# Main UI
# =========================

st.title("🧠 ResearchAgent v0.4 Preview")
st.caption("LangGraph + Agentic RAG + CSV/JSONL Tools + Evidence Checker + LLM-assisted Report Writer")

st.markdown(
    """
这是一个面向科研场景的 Agent Demo。  
它可以根据用户问题进行任务分类，并结合本地 RAG 知识库、CSV/JSONL 工具和证据检查模块生成回答。
"""
)

query = st.text_area(
    "请输入你的科研问题",
    value=selected_example,
    height=120,
)

run_button = st.button("运行 Agent", type="primary")


# =========================
# Run
# =========================

if run_button:
    if not query.strip():
        st.warning("请输入问题。")
    else:
        with st.spinner("Agent 正在运行..."):
            result = run_agent(query.strip())

        st.success("运行完成")

        # -------------------------
        # Summary cards
        # -------------------------

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("任务类型", result.get("task_type", "unknown"))

        with col2:
            st.metric("工具调用", result.get("tool_used", "none"))

        with col3:
            st.metric("证据状态", result.get("evidence_status", "unknown"))

        with col4:
            st.metric("Sources", len(result.get("sources", [])))

        with col5:
            mem_label = f"{result.get('memory_count', 0)}"
            if result.get("memory_written"):
                mem_label += " ✏️"
            st.metric("Memory", mem_label)

        with col6:
            ho_count = result.get("handoff_count", 0)
            ho_label = str(ho_count)
            if ho_count > 0 and result.get("handoff_summary"):
                ho_label += " 🔄"
            st.metric("Handoffs", ho_label)

        st.markdown("---")

        # -------------------------
        # Final answer
        # -------------------------

        st.subheader("最终回答")
        st.markdown(result.get("final_answer", ""))

        # -------------------------
        # Sources
        # -------------------------

        st.subheader("Sources")

        sources = result.get("sources", [])

        if not sources:
            st.info("无 Sources。")
        else:
            for i, source in enumerate(sources, start=1):
                path = source.get("path", "unknown")
                source_type = source.get("source_type", "unknown")
                title = source.get("title", "")
                dataset = source.get("dataset", "")
                run_tag = source.get("run_tag", "")

                with st.expander(f"{i}. [{source_type}] {path}"):
                    if title:
                        st.write("title:", title)
                    if dataset:
                        st.write("dataset:", dataset)
                    if run_tag:
                        st.write("run_tag:", run_tag)

        # -------------------------
        # Tool result
        # -------------------------

        st.subheader("工具分析结果")

        tool_used = result.get("tool_used", "none")
        tool_result_text = result.get("tool_result_text", "")

        if tool_used == "none":
            st.info("本次没有调用 CSV / JSONL 工具。")
        else:
            st.write(f"工具：`{tool_used}`")
            st.text(tool_result_text)

        # -------------------------
        # Evidence
        # -------------------------

        st.subheader("证据检查")

        st.write("状态：", result.get("evidence_status", "unknown"))
        st.write("说明：", result.get("evidence_reason", ""))

        warnings = result.get("evidence_warnings", [])

        if warnings:
            st.warning("\n".join(f"- {w}" for w in warnings))
        else:
            st.success("无证据警告。")

        # -------------------------
        # Memory
        # -------------------------

        st.subheader("研究记忆")

        memory_used = result.get("memory_used", False)
        memory_count = result.get("memory_count", 0)
        memory_error = result.get("memory_error", "")

        if memory_error and "disabled" not in memory_error.lower():
            st.warning(f"Memory retrieval error: {memory_error}")
        elif memory_used:
            st.success(f"Memory Used: True ({memory_count} records)")
        else:
            st.info(f"Memory Used: False ({'disabled' if memory_error else 'no relevant memories'})")

        memories = result.get("retrieved_memories", [])
        if memories:
            for i, m in enumerate(memories[:5], start=1):
                mid = m.get("memory_id", "?") if isinstance(m, dict) else "?"
                mtype = m.get("memory_type", "?") if isinstance(m, dict) else "?"
                summary = (m.get("summary", "") or m.get("content", "")[:100]) if isinstance(m, dict) else ""
                tags = m.get("tags", []) if isinstance(m, dict) else []
                source_title = m.get("source_title", "") if isinstance(m, dict) else ""
                tag_str = ", ".join(tags[:5]) if tags else ""

                label = f"{i}. [{mtype}] {mid[:12]}..."
                if source_title:
                    label += f" — {source_title[:60]}"

                with st.expander(label):
                    st.write("**Summary:**", summary[:300])
                    if tag_str:
                        st.write("**Tags:**", tag_str)

        # -------------------------
        # Multi-Agent / Handoff
        # -------------------------

        handoff_count = result.get("handoff_count", 0)
        handoff_summary = result.get("handoff_summary", "")
        primary_agent = result.get("primary_agent", "")

        if handoff_count > 0 or primary_agent:
            st.markdown("---")
            st.subheader("🤝 Multi-Agent / Handoff")

            if primary_agent:
                st.write(f"**Primary Agent**: {primary_agent}")
            if handoff_summary:
                st.info(handoff_summary)

            handoff_results = result.get("handoff_results", [])
            if handoff_results:
                for i, hr in enumerate(handoff_results[:8], start=1):
                    if isinstance(hr, dict):
                        label = f"{i}. {hr.get('to_agent', '?')} — {hr.get('status', '?')} (conf={hr.get('confidence', 0):.2f})"
                        with st.expander(label):
                            st.write("**Result**:", str(hr.get("result_text", ""))[:500])
                            st.write("**Sources count**:", hr.get("sources_count", 0))
                    else:
                        st.write(f"{i}. {str(hr)[:200]}")

            # Coordinator summary
            coord_summary = result.get("coordinator_summary", "")
            if coord_summary:
                with st.expander("📋 Coordinator Final Arbitration"):
                    st.markdown(coord_summary)

            # Arbitration
            arbitration = result.get("arbitration")
            if arbitration and isinstance(arbitration, dict):
                conflicts = arbitration.get("conflicts", {})
                if conflicts.get("has_conflict"):
                    st.warning(f"⚠️ Conflicts detected: {', '.join(conflicts.get('conflict_types', []))}")

        # -------------------------
        # Memory Write-back
        # -------------------------

        if result.get("memory_written"):
            st.success("📝 Memory write-back: OK")
        if result.get("memory_write_error"):
            st.warning(f"Memory write-back error: {result['memory_write_error']}")

        # -------------------------
        # Debug
        # -------------------------

        if show_debug:
            st.markdown("---")
            st.subheader("🔍 Debug State")

            with st.expander("查看完整 AgentState"):
                st.json(result)

            with st.expander("查看 retrieved_docs"):
                st.json(result.get("retrieved_docs", []))

            with st.expander("查看 tool_result"):
                st.json(result.get("tool_result", {}))

            with st.expander("查看 retrieved_memories"):
                st.json(result.get("retrieved_memories", []))

            if handoff_count > 0:
                with st.expander("查看 handoff_results (raw)"):
                    st.json(result.get("handoff_results", []))

                with st.expander("查看 handoff_plan"):
                    st.json(result.get("handoff_plan", {}))

            if result.get("arbitration"):
                with st.expander("查看 arbitration"):
                    st.json(result.get("arbitration", {}))