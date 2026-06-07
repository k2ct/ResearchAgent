from pathlib import Path
import sys
import os

import streamlit as st


# =========================
# Path setup
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


# ── Graceful imports ────────────────────────────────────────────────
_GRAPH_OK = False
_WORKFLOW_OK = False
build_graph = None

try:
    from research_agent.graph.workflow import build_graph as _bg
    build_graph = _bg
    _GRAPH_OK = True
    _WORKFLOW_OK = True
except ImportError as e:
    _GRAPH_IMPORT_ERROR = str(e)
    _GRAPH_OK = False


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
    if not _GRAPH_OK:
        return None
    try:
        return build_graph()
    except Exception:
        return None


def run_agent(query: str) -> dict:
    graph = get_graph()
    if graph is None:
        return {
            "final_answer": "Graph compilation failed. Please check logs.",
            "task_type": "error",
            "evidence_status": "error",
            "memory_error": "Graph unavailable",
            "memory_used": False,
            "memory_count": 0,
            "sources": [],
            "retrieved_memories": [],
            "handoff_count": 0,
            "handoff_results": [],
            "tool_used": "none",
            "tool_result_text": "",
            "evidence_reason": "Graph build error",
            "evidence_warnings": ["Graph compilation failed"],
            "primary_agent": "",
            "handoff_summary": "",
            "handoff_plan": {},
            "memory_written": False,
            "memory_write_error": "",
            "multi_agent_enabled": False,
        }
    return graph.invoke(create_initial_state(query))


# =========================
# Page config
# =========================

st.set_page_config(
    page_title="ResearchAgent v0.5 Memory + Multi-Agent Preview",
    page_icon="🧠",
    layout="wide",
)


# =========================
# Sidebar
# =========================

st.sidebar.title("ResearchAgent v0.5")
st.sidebar.caption("Memory + Multi-Agent Preview")

# ── Capabilities ────────────────────────────────────────────────────
st.sidebar.markdown("### 🧩 Capabilities")
cap_items = [
    "LangGraph workflow",
    "Agentic RAG + Hybrid RAG",
    "CSV / JSONL tool calling",
    "Evidence Checker",
    "Report Writer (Template / LLM)",
    "Memory System (Store / Retrieve / Privacy / Consolidation)",
    "Memory-aware Agent retrieval",
    "Multi-Agent Orchestrator (8 agents)",
    "Agent Handoff + Arbitration",
    "Shared Workspace (section-level permissions)",
    "Multi-Agent Tracing & Quality Eval",
]
for item in cap_items:
    st.sidebar.markdown(f"- {item}")

st.sidebar.markdown("---")

# ── Config status ───────────────────────────────────────────────────
st.sidebar.markdown("### ⚙️ Runtime Status")

# LLM Enhancement
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# LLM state
llm_enabled = os.getenv("ENABLE_LLM_ENHANCEMENT", "false").strip().lower() in ("1", "true", "yes", "y")
# Try to detect from research_agent if available
try:
    from research_agent.llm.client import is_llm_enhancement_enabled
    llm_enabled = is_llm_enhancement_enabled()
except Exception:
    pass

# Memory-aware state
memory_aware = False
try:
    from research_agent.graph.nodes import _is_memory_aware_enabled
    memory_aware = _is_memory_aware_enabled()
except Exception:
    memory_aware = os.getenv("ENABLE_MEMORY_AWARE_AGENT", "false").strip().lower() in ("1", "true", "yes", "y")

# Multi-agent state
multi_agent = os.getenv("ENABLE_MULTI_AGENT", "false").strip().lower() == "true"

# MinerU backend
mineru_backend = os.getenv("MINERU_BACKEND", "local").strip().lower()
mineru_label = f"{mineru_backend}" if mineru_backend else "local"

# Retrieval mode
retrieval_mode = os.getenv("RETRIEVAL_MODE", "vector").strip().lower()
retrieval_label = retrieval_mode if retrieval_mode in ("vector", "hybrid") else "vector"

col_a, col_b = st.sidebar.columns(2)
with col_a:
    st.metric("LLM Enhancement", "✅ ON" if llm_enabled else "⬜ OFF")
    st.metric("Memory-aware", "✅ ON" if memory_aware else "⬜ OFF")
with col_b:
    st.metric("Multi-Agent", "✅ ON" if multi_agent else "⬜ OFF")
    st.metric("Retrieval", retrieval_label)

st.sidebar.metric("MinerU Backend", mineru_label)

# ── Warnings ────────────────────────────────────────────────────────
if not _GRAPH_OK:
    st.sidebar.error("⚠️ Graph module failed to import. Check dependencies.")

st.sidebar.markdown("---")

show_debug = st.sidebar.checkbox("Show Debug Info", value=True)

st.sidebar.markdown("### 📝 示例问题")

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

st.title("🧠 ResearchAgent v0.5 Memory + Multi-Agent Preview")
st.caption(
    "LangGraph + Agentic RAG + CSV/JSONL Tools + Evidence Checker + "
    "LLM-assisted Report Writer + Memory System + Multi-Agent Orchestrator"
)

st.markdown(
    """
这是一个面向科研场景的 Agent Demo。
它可以根据用户问题进行任务分类，并结合本地 RAG 知识库、CSV/JSONL 工具、研究记忆系统、
和多 Agent 协作生成回答。
"""
)

# Show a warning if graph is unavailable
if not _GRAPH_OK:
    st.error("⚠️ LangGraph workflow failed to load. Some features may be unavailable.")
    st.info("Please check that all dependencies are installed: `pip install -r requirements.txt`")

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
            try:
                result = run_agent(query.strip())
            except Exception as e:
                st.error(f"Agent execution error: {type(e).__name__}: {e}")
                result = {
                    "final_answer": f"Execution failed: {e}",
                    "task_type": "error",
                    "evidence_status": "error",
                    "memory_error": str(e),
                    "memory_used": False,
                    "memory_count": 0,
                    "sources": [],
                    "retrieved_memories": [],
                    "handoff_count": 0,
                    "handoff_results": [],
                    "tool_used": "none",
                    "tool_result_text": "",
                    "evidence_reason": "",
                    "evidence_warnings": [str(e)],
                    "primary_agent": "",
                    "handoff_summary": "",
                    "handoff_plan": {},
                    "memory_written": False,
                    "memory_write_error": "",
                    "multi_agent_enabled": False,
                }

        st.success("运行完成")

        # ─────────────────────────────────────────────────────────────
        # Summary Metrics (8 cards)
        # ─────────────────────────────────────────────────────────────

        col1, col2, col3, col4 = st.columns(4)
        col5, col6, col7, col8 = st.columns(4)

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
            if result.get("memory_used"):
                mem_label += " ✓"
            if result.get("memory_written"):
                mem_label += " 💾"
            st.metric("Memory Used", mem_label)

        with col6:
            ho_count = result.get("handoff_count", 0)
            ho_label = str(ho_count)
            if ho_count > 0:
                ho_label += " 🔄"
            st.metric("Handoffs", ho_label)

        with col7:
            primary = result.get("primary_agent", "")
            primary_label = primary if primary else "—"
            st.metric("Primary Agent", primary_label)

        with col8:
            tq = result.get("trace_quality")
            if tq and isinstance(tq, dict):
                tq_label = tq.get("quality_label", "—")
            else:
                tq_label = "—"
            st.metric("Trace Quality", tq_label)

        st.markdown("---")

        # ─────────────────────────────────────────────────────────────
        # Final answer
        # ─────────────────────────────────────────────────────────────

        st.subheader("📝 最终回答")
        st.markdown(result.get("final_answer", ""))

        # ─────────────────────────────────────────────────────────────
        # Sources
        # ─────────────────────────────────────────────────────────────

        st.subheader("📚 Sources")

        sources = result.get("sources", [])

        if not sources:
            st.info("无 Sources。")
        else:
            for i, source in enumerate(sources, start=1):
                if not isinstance(source, dict):
                    st.write(f"{i}. {str(source)[:200]}")
                    continue
                path = source.get("path", "unknown")
                source_type = source.get("source_type", "unknown")
                title = source.get("title", "")
                dataset = source.get("dataset", "")
                run_tag = source.get("run_tag", "")
                from_memory = source.get("from_memory", False)

                label_parts = [f"{i}. [{source_type}]"]
                if from_memory:
                    label_parts.append("🧠")
                label_parts.append(path)
                label = " ".join(label_parts)

                with st.expander(label):
                    if title:
                        st.write("title:", title)
                    if dataset:
                        st.write("dataset:", dataset)
                    if run_tag:
                        st.write("run_tag:", run_tag)
                    if from_memory:
                        st.write("from_memory: True")

        # ─────────────────────────────────────────────────────────────
        # Tool result
        # ─────────────────────────────────────────────────────────────

        st.subheader("🔧 工具分析结果")

        tool_used = result.get("tool_used", "none")
        tool_result_text = result.get("tool_result_text", "")

        if tool_used == "none":
            st.info("本次没有调用 CSV / JSONL 工具。")
        else:
            st.write(f"工具：`{tool_used}`")
            st.text(tool_result_text)

        # ─────────────────────────────────────────────────────────────
        # Evidence
        # ─────────────────────────────────────────────────────────────

        st.subheader("🔍 证据检查")

        st.write("状态：", result.get("evidence_status", "unknown"))
        st.write("说明：", result.get("evidence_reason", ""))

        warnings_list = result.get("evidence_warnings", [])

        if warnings_list:
            st.warning("\n".join(f"- {w}" for w in warnings_list))
        else:
            st.success("无证据警告。")

        # ─────────────────────────────────────────────────────────────
        # Memory Section
        # ─────────────────────────────────────────────────────────────

        st.subheader("🧠 研究记忆 (Memory)")

        memory_used = result.get("memory_used", False)
        memory_count = result.get("memory_count", 0)
        memory_error = result.get("memory_error", "")

        mem_col1, mem_col2 = st.columns([1, 3])
        with mem_col1:
            if memory_used:
                st.success(f"✅ Memory Used ({memory_count} records)")
            elif memory_error and "disabled" in memory_error.lower():
                st.info("⬜ Memory-aware agent disabled")
            elif memory_error:
                st.warning(f"⚠️ Memory error: {memory_error}")
            else:
                st.info("⬜ No relevant memories")

            # Memory saved indicator
            if result.get("memory_written"):
                st.success("💾 Memory saved to store")
            if result.get("memory_write_error"):
                st.warning(f"Write-back error: {result['memory_write_error']}")

        with mem_col2:
            memories = result.get("retrieved_memories", [])
            if memories:
                for i, m in enumerate(memories[:10], start=1):
                    if not isinstance(m, dict):
                        st.write(f"{i}. {str(m)[:200]}")
                        continue

                    mid = m.get("memory_id", "?")
                    mtype = m.get("memory_type", "?")
                    mscope = m.get("memory_scope", "private")
                    owner = m.get("owner_agent", "")
                    summary = m.get("summary", "")
                    if not summary:
                        summary = (m.get("content", "") or "")[:100]
                    tags = m.get("tags", []) if isinstance(m.get("tags"), list) else []
                    source_title = m.get("source_title", "")

                    # Build label
                    label_parts = [f"{i}. [{mtype}]"]
                    if mscope:
                        label_parts.append(f"({mscope})")
                    label_parts.append(mid[:16])
                    if not mid.startswith("mem_"):
                        label_parts.append("...")
                    if owner:
                        label_parts.append(f"— {owner}")
                    label = " ".join(label_parts)

                    with st.expander(label):
                        # Key metadata line
                        meta_line = f"**ID**: `{mid}` | **Type**: `{mtype}` | **Scope**: `{mscope}`"
                        if owner:
                            meta_line += f" | **Owner**: `{owner}`"
                        st.markdown(meta_line)

                        if summary:
                            st.write("**Summary:**", summary[:300])
                        if tags:
                            st.write("**Tags:**", ", ".join(tags[:8]))
                        if source_title:
                            st.write("**Source:**", source_title[:120])
                        # Do NOT display full content
            elif not memory_used:
                st.caption("(no memories retrieved)")

        # ─────────────────────────────────────────────────────────────
        # Multi-Agent / Handoff Section
        # ─────────────────────────────────────────────────────────────

        handoff_count = result.get("handoff_count", 0)
        handoff_summary = result.get("handoff_summary", "")
        primary_agent = result.get("primary_agent", "")
        multi_agent_enabled = result.get("multi_agent_enabled", False)

        if handoff_count > 0 or primary_agent or multi_agent_enabled:
            st.markdown("---")
            st.subheader("🤝 Multi-Agent / Handoff")

            # Primary agent + summary
            ma_col1, ma_col2 = st.columns([1, 2])
            with ma_col1:
                if primary_agent:
                    st.info(f"**Primary Agent**: {primary_agent}")
                if handoff_count > 0:
                    st.metric("Handoff Count", handoff_count)
                tq = result.get("trace_quality")
                if tq and isinstance(tq, dict):
                    st.metric("Trace Quality", tq.get("quality_label", "—"))

            with ma_col2:
                if handoff_summary:
                    st.info(handoff_summary)

            # Handoff plan
            handoff_plan = result.get("handoff_plan", {}) or {}
            if handoff_plan and isinstance(handoff_plan, dict):
                targets = handoff_plan.get("targets", [])
                if targets:
                    st.write("**Handoff Plan Targets**:", ", ".join(targets))

            # Individual handoff results
            handoff_results = result.get("handoff_results", [])
            if handoff_results:
                st.markdown("---")
                st.write("**Handoff Results:**")
                for i, hr in enumerate(handoff_results[:8], start=1):
                    if isinstance(hr, dict):
                        to_agent = hr.get("to_agent", "?")
                        status = hr.get("status", "?")
                        conf = hr.get("confidence", 0)
                        if isinstance(conf, (int, float)):
                            conf_str = f"{conf:.2f}"
                        else:
                            conf_str = str(conf)
                        label = f"{i}. {to_agent} — {status} (conf={conf_str})"

                        with st.expander(label):
                            result_text = str(hr.get("result_text", ""))[:500]
                            if result_text:
                                st.markdown(result_text)
                            else:
                                st.caption("(no result text)")
                            st.write("**Sources count**:", hr.get("sources_count", len(hr.get("sources", [])) or 0))
                            memory_ids = hr.get("memory_ids", [])
                            if memory_ids:
                                st.write("**Memory IDs**:", len(memory_ids))
                    else:
                        st.write(f"{i}. {str(hr)[:200]}")

            # Coordinator summary / arbitration
            coord_summary = result.get("coordinator_summary", "")
            if coord_summary:
                with st.expander("📋 Coordinator Final Arbitration"):
                    st.markdown(coord_summary)

            arbitration = result.get("arbitration")
            if arbitration and isinstance(arbitration, dict):
                conflicts = arbitration.get("conflicts", {}) or {}
                if conflicts.get("has_conflict"):
                    st.warning(
                        f"⚠️ Conflicts detected: "
                        f"{', '.join(conflicts.get('conflict_types', []))}"
                    )

        # ─────────────────────────────────────────────────────────────
        # Workspace Section
        # ─────────────────────────────────────────────────────────────

        ws_info = result.get("workspace")
        if ws_info and isinstance(ws_info, dict):
            st.markdown("---")
            st.subheader("📋 Shared Workspace")

            ws_id = ws_info.get("workspace_id", "?")
            ws_title = ws_info.get("title", "")
            section_count = ws_info.get("section_count", 0)
            patch_count = ws_info.get("patch_count", 0)
            suggestions_count = ws_info.get("suggestions_count", 0)

            ws_col1, ws_col2, ws_col3, ws_col4 = st.columns(4)
            with ws_col1:
                st.metric("Workspace ID", ws_id[:20] if ws_id != "?" else "?")
            with ws_col2:
                st.metric("Sections", section_count)
            with ws_col3:
                st.metric("Patches", patch_count)
            with ws_col4:
                st.metric("Suggestions", suggestions_count)

            if ws_title:
                st.write(f"**Title**: {ws_title}")

            # Show patches if available
            ws_patches = ws_info.get("patches", [])
            if ws_patches:
                for i, p in enumerate(ws_patches[:10], start=1):
                    if not isinstance(p, dict):
                        continue
                    p_label = (
                        f"Patch {i}: {p.get('agent_id','?')} → "
                        f"{p.get('target_section','?')} "
                        f"({p.get('status','?')})"
                    )
                    with st.expander(p_label):
                        st.write("**Operation**:", p.get("operation", "?"))
                        st.write("**Reason**:", p.get("reason", ""))
                        st.write("**Content preview**:",
                                 str(p.get("content", ""))[:300])

        # ─────────────────────────────────────────────────────────────
        # Memory Write-back Summary
        # ─────────────────────────────────────────────────────────────

        if result.get("memory_written"):
            st.success("📝 Memory write-back: OK")
        if result.get("memory_write_error"):
            st.warning(f"Memory write-back error: {result['memory_write_error']}")

        # ─────────────────────────────────────────────────────────────
        # Debug Section
        # ─────────────────────────────────────────────────────────────

        if show_debug:
            st.markdown("---")
            st.subheader("🔍 Debug State")

            # ── Full raw state ──────────────────────────────────
            with st.expander("查看完整 AgentState (raw)"):
                st.json(result)

            # ── RAG ──────────────────────────────────────────────
            with st.expander("查看 retrieved_docs"):
                st.json(result.get("retrieved_docs", []))

            with st.expander("查看 tool_result"):
                st.json(result.get("tool_result", {}))

            # ── Memory ───────────────────────────────────────────
            with st.expander("查看 retrieved_memories"):
                st.json(result.get("retrieved_memories", []))

            # ── Multi-Agent ──────────────────────────────────────
            if result.get("handoff_plan"):
                with st.expander("查看 handoff_plan"):
                    st.json(result.get("handoff_plan", {}))

            if result.get("handoff_results"):
                with st.expander("查看 handoff_results (raw)"):
                    st.json(result.get("handoff_results", []))

            if result.get("arbitration"):
                with st.expander("查看 arbitration"):
                    st.json(result.get("arbitration", {}))

            if result.get("trace_quality"):
                with st.expander("查看 trace_quality"):
                    st.json(result.get("trace_quality", {}))

            # ── Workspace ────────────────────────────────────────
            ws_info_debug = result.get("workspace")
            if ws_info_debug:
                with st.expander("查看 workspace patches"):
                    st.json(ws_info_debug)
