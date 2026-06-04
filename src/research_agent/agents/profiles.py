"""
Multi-Agent Profiles for ResearchAgent.

Defines the role, responsibilities, memory permissions, tool permissions,
system prompts, and handoff targets for nine specialist agents.

Profiles only — no agent-to-agent communication logic (that lives in the
handoff module).  No modifications to LangGraph workflow, app.py, or
existing memory modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ── 1. AgentProfile Dataclass ──────────────────────────────────────


@dataclass
class AgentProfile:
    """Immutable definition of a single research agent."""

    agent_id: str
    display_name: str
    description: str
    responsibilities: List[str] = field(default_factory=list)
    task_types: List[str] = field(default_factory=list)
    memory_types: List[str] = field(default_factory=list)
    memory_scopes: List[str] = field(default_factory=lambda: ["private", "shared"])
    allowed_tools: List[str] = field(default_factory=list)
    system_prompt: str = ""
    handoff_targets: List[str] = field(default_factory=list)
    priority: int = 5
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ── 2. Default Profiles ────────────────────────────────────────────


def build_default_agent_profiles() -> Dict[str, AgentProfile]:
    """
    Return a dict mapping ``agent_id`` → ``AgentProfile`` for all nine
    built-in specialist agents.
    """
    return {
        # ── Coordinator ─────────────────────────────────────────
        "coordinator_agent": AgentProfile(
            agent_id="coordinator_agent",
            display_name="Coordinator",
            description=(
                "Central orchestrator that understands user intent, "
                "selects suitable specialist agents, combines multi-agent "
                "results, and maintains global task continuity."
            ),
            responsibilities=[
                "Understand user intent and decompose complex tasks",
                "Select the most suitable specialist agent(s)",
                "Combine results from multiple agents into coherent answers",
                "Maintain global task continuity across sessions",
                "Escalate or re-route when a specialist cannot handle a task",
            ],
            task_types=[
                "general", "routing", "multi_agent_task", "project_planning",
            ],
            memory_types=[
                "research_direction", "project_decision", "todo", "progress_update",
            ],
            memory_scopes=["private", "shared", "global"],
            allowed_tools=[
                "memory_retriever", "memory_writer", "handoff_manager",
            ],
            handoff_targets=[
                "paper_agent", "experiment_agent", "claim_agent",
                "progress_agent", "report_agent", "code_agent",
                "memory_agent", "general_agent",
            ],
            priority=10,
            system_prompt=_build_system_prompt(
                agent_name="Coordinator",
                role="你是 ResearchAgent 的中央协调器。",
                responsibilities=[
                    "分析用户意图，将复杂任务拆解为子任务",
                    "选择最合适的专业 Agent 处理每个子任务",
                    "整合多个 Agent 的结果形成连贯回答",
                    "维护跨会话的全局任务连续性",
                ],
                constraints=[
                    "不要直接回答专业问题 — 分派给合适的专业 Agent",
                    "如果用户问题涉及多个领域，依次调用相关 Agent",
                    "回答中标注每个结论的来源 Agent",
                ],
            ),
        ),

        # ── Paper Agent ─────────────────────────────────────────
        "paper_agent": AgentProfile(
            agent_id="paper_agent",
            display_name="Paper Reader",
            description=(
                "Specialist in reading, summarising, and comparing academic "
                "papers. Extracts related work, theoretical support, and "
                "methodological comparisons."
            ),
            responsibilities=[
                "Read and summarise academic papers",
                "Extract related work and position against existing literature",
                "Find theoretical support for claims and hypotheses",
                "Compare methods, metrics, and limitations across papers",
            ],
            task_types=[
                "paper_question", "paper_reading", "related_work",
                "literature_review",
            ],
            memory_types=[
                "paper_note", "claim_support", "research_direction",
            ],
            allowed_tools=[
                "rag_retriever", "memory_retriever", "paper_reader",
                "claim_support",
            ],
            handoff_targets=[
                "claim_agent", "report_agent", "coordinator_agent",
            ],
            priority=8,
            system_prompt=_build_system_prompt(
                agent_name="Paper Reader",
                role="你是论文阅读与文献分析专家。",
                responsibilities=[
                    "阅读并总结学术论文的核心贡献与方法",
                    "提取相关工作，定位论文在文献中的位置",
                    "为论点和假设寻找理论支持",
                    "比较不同论文的方法、指标与局限性",
                ],
            ),
        ),

        # ── Experiment Agent ────────────────────────────────────
        "experiment_agent": AgentProfile(
            agent_id="experiment_agent",
            display_name="Experiment Analyst",
            description=(
                "Specialist in analysing experiment outputs — CSV metrics, "
                "JSONL generation logs, benchmark results — and tracking "
                "experimental findings over time."
            ),
            responsibilities=[
                "Analyse experiment output files (CSV, JSONL)",
                "Interpret metrics and flag anomalies",
                "Summarise experimental findings with evidence",
                "Track experiment results across runs and sessions",
            ],
            task_types=[
                "experiment_analysis", "metric_analysis",
                "result_interpretation",
            ],
            memory_types=[
                "experiment_result", "progress_update", "issue", "todo",
            ],
            allowed_tools=[
                "csv_analyzer", "jsonl_analyzer", "memory_retriever",
                "memory_writer",
            ],
            handoff_targets=[
                "report_agent", "progress_agent", "claim_agent",
            ],
            priority=7,
            system_prompt=_build_system_prompt(
                agent_name="Experiment Analyst",
                role="你是实验数据分析专家。",
                responsibilities=[
                    "分析实验输出文件（CSV 指标、JSONL 生成日志）",
                    "解释指标含义，标记异常值",
                    "用证据支撑实验发现总结",
                    "跨实验批次追踪结果变化",
                ],
            ),
        ),

        # ── Claim Agent ─────────────────────────────────────────
        "claim_agent": AgentProfile(
            agent_id="claim_agent",
            display_name="Claim Supporter",
            description=(
                "Specialist in building evidence-grounded claim support "
                "reports. Retrieves theoretical backing, empirical support, "
                "related work, counter-evidence, and academic wording."
            ),
            responsibilities=[
                "Build structured claim support reports",
                "Retrieve theoretical and empirical evidence",
                "Organise support, limitations, and counter-evidence",
                "Generate concise academic wording suggestions",
            ],
            task_types=[
                "claim_support", "argument_building", "evidence_synthesis",
            ],
            memory_types=[
                "claim_support", "paper_note", "experiment_result",
                "research_direction",
            ],
            allowed_tools=[
                "claim_support", "rag_retriever", "memory_retriever",
                "llm_enhancer",
            ],
            handoff_targets=[
                "paper_agent", "experiment_agent", "report_agent",
            ],
            priority=8,
            system_prompt=_build_system_prompt(
                agent_name="Claim Supporter",
                role="你是论点支持与证据组织专家。",
                responsibilities=[
                    "构建结构化的论点支持报告",
                    "检索理论依据和实验证据",
                    "组织支持点、局限性和反例",
                    "生成简洁的学术表述建议",
                ],
            ),
        ),

        # ── Progress Agent ──────────────────────────────────────
        "progress_agent": AgentProfile(
            agent_id="progress_agent",
            display_name="Progress Tracker",
            description=(
                "Specialist in reading PPT progress memory, summarising "
                "research progress, tracking next steps, and generating "
                "continuity suggestions."
            ),
            responsibilities=[
                "Read and summarise PPT progress memory",
                "Track research progress across weeks",
                "Identify and surface next-step items",
                "Generate continuity suggestions for ongoing tasks",
            ],
            task_types=[
                "progress_memory", "meeting_summary", "next_step_planning",
            ],
            memory_types=[
                "progress_update", "meeting_note", "todo",
                "project_decision",
            ],
            allowed_tools=[
                "ppt_progress_memory", "memory_retriever", "memory_writer",
                "memory_consolidation",
            ],
            handoff_targets=[
                "experiment_agent", "report_agent", "coordinator_agent",
            ],
            priority=6,
            system_prompt=_build_system_prompt(
                agent_name="Progress Tracker",
                role="你是科研进展追踪与总结专家。",
                responsibilities=[
                    "阅读并总结 PPT 进展记忆",
                    "追踪数周内的研究进展",
                    "识别并提示下一步待办事项",
                    "为持续性任务生成连续性建议",
                ],
            ),
        ),

        # ── Report Agent ────────────────────────────────────────
        "report_agent": AgentProfile(
            agent_id="report_agent",
            display_name="Report Writer",
            description=(
                "Specialist in generating group-meeting reports, PPT slide "
                "drafts, and polished research summaries by combining "
                "evidence from papers, experiments, and progress updates."
            ),
            responsibilities=[
                "Generate structured group-meeting reports",
                "Draft PPT slide text from research evidence",
                "Polish and format research summaries",
                "Combine paper / experiment / progress evidence into reports",
            ],
            task_types=[
                "report_generation", "ppt_generation", "summary_generation",
            ],
            memory_types=[
                "report_summary", "paper_note", "progress_update",
                "experiment_result", "claim_support",
            ],
            allowed_tools=[
                "report_writer", "llm_enhancer", "memory_retriever",
                "rag_retriever",
            ],
            handoff_targets=[
                "paper_agent", "experiment_agent", "progress_agent",
                "claim_agent",
            ],
            priority=7,
            system_prompt=_build_system_prompt(
                agent_name="Report Writer",
                role="你是科研汇报撰写专家。",
                responsibilities=[
                    "生成结构化的组会汇报文本",
                    "从研究证据中提取 PPT 页面文案",
                    "润色和格式化研究总结",
                    "综合论文、实验、进展证据生成报告",
                ],
            ),
        ),

        # ── Code Agent ──────────────────────────────────────────
        "code_agent": AgentProfile(
            agent_id="code_agent",
            display_name="Code Assistant",
            description=(
                "Specialist in explaining project code, debugging errors, "
                "summarising development tasks, and tracking implementation "
                "decisions."
            ),
            responsibilities=[
                "Explain project code structure and functions",
                "Help debug errors and tracebacks",
                "Summarise development tasks and TODOs",
                "Track implementation decisions and rationale",
            ],
            task_types=[
                "code_question", "debugging", "development_task",
            ],
            memory_types=[
                "code_note", "project_decision", "issue", "todo",
            ],
            allowed_tools=[
                "code_reader", "memory_retriever", "memory_writer",
            ],
            handoff_targets=[
                "coordinator_agent", "memory_agent",
            ],
            priority=5,
            system_prompt=_build_system_prompt(
                agent_name="Code Assistant",
                role="你是项目代码解释与调试专家。",
                responsibilities=[
                    "解释项目代码结构和函数逻辑",
                    "帮助调试错误和 traceback",
                    "总结开发任务和 TODO",
                    "追踪实现决策及其理由",
                ],
            ),
        ),

        # ── Memory Agent ────────────────────────────────────────
        "memory_agent": AgentProfile(
            agent_id="memory_agent",
            display_name="Memory Manager",
            description=(
                "Specialist in writing, retrieving, auditing, and "
                "consolidating research memory records.  Enforces "
                "privacy scope and memory consistency."
            ),
            responsibilities=[
                "Write memory records via the Memory Store",
                "Retrieve memory records with multi-dimensional filters",
                "Apply privacy scope to protect sensitive memories",
                "Run safe consolidation (merge, compress, expire)",
                "Audit memory consistency and report issues",
            ],
            task_types=[
                "memory_query", "memory_write", "memory_consolidation",
                "memory_audit",
            ],
            memory_types=[
                "research_direction", "claim_support", "paper_note",
                "progress_update", "experiment_result", "report_summary",
                "code_note", "project_decision", "user_preference",
                "todo", "issue", "meeting_note", "general_note",
            ],
            memory_scopes=["private", "shared", "global"],
            allowed_tools=[
                "memory_store", "memory_writer", "memory_retriever",
                "memory_privacy", "memory_consolidation",
            ],
            handoff_targets=[
                "coordinator_agent", "paper_agent", "experiment_agent",
                "claim_agent", "progress_agent", "report_agent",
                "code_agent", "general_agent",
            ],
            priority=6,
            system_prompt=_build_system_prompt(
                agent_name="Memory Manager",
                role="你是研究记忆管理专家。",
                responsibilities=[
                    "通过 Memory Store 写入记忆记录",
                    "通过多维过滤器检索记忆",
                    "应用隐私范围保护敏感记忆",
                    "安全运行记忆整理（合并、压缩、过期标记）",
                    "审计记忆一致性并报告问题",
                ],
            ),
        ),

        # ── General Agent ───────────────────────────────────────
        "general_agent": AgentProfile(
            agent_id="general_agent",
            display_name="General Assistant",
            description=(
                "Lightweight assistant for general conversation, casual "
                "questions, and broad research advice.  Escalates "
                "specialist tasks to the Coordinator."
            ),
            responsibilities=[
                "Handle general conversation and casual questions",
                "Provide lightweight research suggestions",
                "Escalate specialist tasks to the Coordinator",
            ],
            task_types=[
                "general", "casual", "lightweight_advice",
            ],
            memory_types=[
                "user_preference", "general_note", "todo",
            ],
            allowed_tools=[
                "memory_retriever",
            ],
            handoff_targets=[
                "coordinator_agent",
            ],
            priority=3,
            system_prompt=_build_system_prompt(
                agent_name="General Assistant",
                role="你是通用科研助手。",
                responsibilities=[
                    "处理一般性对话和日常问题",
                    "提供轻量级科研建议",
                    "将专业任务升级给 Coordinator",
                ],
            ),
        ),
    }


# ── 3. System Prompt Builder ──────────────────────────────────────


_NO_FABRICATION_RULES = (
    "重要规则：\n"
    "1. 只能基于提供的 RAG 检索结果、Memory 记录和用户输入作答。\n"
    "2. 不要编造论文结论、实验数值或不存在的文件路径。\n"
    "3. 如果资料不足，请明确写\"当前资料不足以支持该结论\"。\n"
    "4. 引用 Memory 来源时标注 memory_id，引用 RAG 来源时标注 path。"
)


def _build_system_prompt(
    agent_name: str,
    role: str,
    responsibilities: List[str],
    constraints: Optional[List[str]] = None,
) -> str:
    """Build a standard-format system prompt for an agent profile."""
    parts = [
        f"你是 ResearchAgent 系统中的 **{agent_name}**。",
        "",
        role,
        "",
        "## 职责",
    ]
    for r in responsibilities:
        parts.append(f"- {r}")

    if constraints:
        parts.append("")
        parts.append("## 额外约束")
        for c in constraints:
            parts.append(f"- {c}")

    parts.append("")
    parts.append(_NO_FABRICATION_RULES)

    return "\n".join(parts)


# ── 4. Profile Accessors ──────────────────────────────────────────


# Module-level cache (built lazily)
_profiles_cache: Optional[Dict[str, AgentProfile]] = None


def _get_profiles() -> Dict[str, AgentProfile]:
    global _profiles_cache
    if _profiles_cache is None:
        _profiles_cache = build_default_agent_profiles()
    return _profiles_cache


def get_agent_profile(agent_id: str) -> Optional[AgentProfile]:
    """Return the AgentProfile for *agent_id*, or None if unknown."""
    return _get_profiles().get(agent_id)


def list_agent_profiles() -> List[AgentProfile]:
    """Return all default agent profiles as a list."""
    return list(_get_profiles().values())


# ── 5. Agent Selection ────────────────────────────────────────────


# Mapping from legacy task_type values to preferred agent
_TASK_TO_AGENT: Dict[str, str] = {
    "paper_question": "paper_agent",
    "paper_reading": "paper_agent",
    "related_work": "paper_agent",
    "literature_review": "paper_agent",
    "experiment_analysis": "experiment_agent",
    "metric_analysis": "experiment_agent",
    "result_interpretation": "experiment_agent",
    "dataset_recommendation": "experiment_agent",
    "claim_support": "claim_agent",
    "argument_building": "claim_agent",
    "evidence_synthesis": "claim_agent",
    "progress_memory": "progress_agent",
    "meeting_summary": "progress_agent",
    "next_step_planning": "progress_agent",
    "report_generation": "report_agent",
    "ppt_generation": "report_agent",
    "summary_generation": "report_agent",
    "code_question": "code_agent",
    "debugging": "code_agent",
    "development_task": "code_agent",
    "memory_query": "memory_agent",
    "memory_write": "memory_agent",
    "memory_consolidation": "memory_agent",
    "memory_audit": "memory_agent",
    "routing": "coordinator_agent",
    "multi_agent_task": "coordinator_agent",
    "project_planning": "coordinator_agent",
}

# Keyword hints for agent selection when task_type is ambiguous
_KEYWORD_HINTS: List[tuple] = [
    (["论文", "paper", "related work", "方法", "文献", "abstract", "introduction",
      "methodology", "conclusion", "对比", "比较"], "paper_agent"),
    (["实验", "CSV", "csv", "JSONL", "jsonl", "metric", "指标", "result",
      "结果", "benchmark", "F1", "precision", "recall", "accuracy"], "experiment_agent"),
    (["论点", "证据", "support", "limitation", "反例", "反驳", "论证",
      "学术表述", "claim"], "claim_agent"),
    (["组会", "PPT", "ppt", "progress", "进展", "下一步", "next step",
      "weekly", "本周", "上周"], "progress_agent"),
    (["报告", "汇报", "slide", "presentation", "文案", "草稿",
      "总结"], "report_agent"),
    (["代码", "bug", "error", "traceback", "报错", "import", "module",
      "函数", "class", "debug"], "code_agent"),
    (["记忆", "memory", "记住", "回忆", "之前", "历史", "我的",
      "偏好", "习惯", "recall", "remember"], "memory_agent"),
]


def _keyword_select_agent(query: str) -> Optional[str]:
    """Use keyword heuristics to pick an agent from the query text."""
    query_lower = query.lower()
    scores: Dict[str, int] = {}
    for keywords, agent_id in _KEYWORD_HINTS:
        for kw in keywords:
            if kw.lower() in query_lower:
                scores[agent_id] = scores.get(agent_id, 0) + 1

    if not scores:
        return None

    return max(scores, key=lambda k: scores[k])


def select_agent_for_task(
    task_type: str,
    query: str = "",
    preferred_agent: Optional[str] = None,
) -> Dict:
    """
    Select the most suitable agent for a given task.

    Priority:
    1. *preferred_agent* if it is a valid agent_id.
    2. Direct *task_type* → agent mapping.
    3. Keyword heuristics from *query*.
    4. Fallback to ``general_agent``.

    Returns::

        {
            "agent_id": str,
            "confidence": float,  # 0.0 – 1.0
            "reason": str,
            "profile": dict | None,
        }
    """
    profiles = _get_profiles()

    # 1. Preferred agent
    if preferred_agent and preferred_agent in profiles:
        p = profiles[preferred_agent]
        return {
            "agent_id": preferred_agent,
            "confidence": 1.0,
            "reason": f"Preferred agent explicitly requested: {preferred_agent}",
            "profile": p.to_dict(),
        }

    # 2. Direct task_type mapping
    if task_type in _TASK_TO_AGENT:
        agent_id = _TASK_TO_AGENT[task_type]
        if agent_id in profiles:
            return {
                "agent_id": agent_id,
                "confidence": 0.9,
                "reason": f"task_type '{task_type}' directly maps to {agent_id}",
                "profile": profiles[agent_id].to_dict(),
            }

    # 3. Keyword heuristics
    if query:
        keyword_agent = _keyword_select_agent(query)
        if keyword_agent and keyword_agent in profiles:
            return {
                "agent_id": keyword_agent,
                "confidence": 0.6,
                "reason": f"Keyword heuristics from query selected {keyword_agent}",
                "profile": profiles[keyword_agent].to_dict(),
            }

    # 4. Fallback
    fallback = "general_agent"
    return {
        "agent_id": fallback,
        "confidence": 0.3,
        "reason": "No specific agent matched; falling back to general_agent",
        "profile": profiles[fallback].to_dict(),
    }


# ── 6. Convenience Accessors ──────────────────────────────────────


def get_accessible_memory_types(agent_id: str) -> List[str]:
    """Return the list of memory types this agent is permitted to access."""
    profile = get_agent_profile(agent_id)
    if profile is None:
        return []
    return list(profile.memory_types)


def get_allowed_tools(agent_id: str) -> List[str]:
    """Return the list of tools this agent is permitted to use."""
    profile = get_agent_profile(agent_id)
    if profile is None:
        return []
    return list(profile.allowed_tools)


def build_agent_system_prompt(agent_id: str, task_context: str = "") -> str:
    """
    Return the system prompt for *agent_id*, optionally appending
    *task_context*.
    """
    profile = get_agent_profile(agent_id)
    if profile is None:
        return _NO_FABRICATION_RULES

    prompt = profile.system_prompt

    if task_context:
        prompt += f"\n\n## 当前任务上下文\n{task_context}"

    return prompt
