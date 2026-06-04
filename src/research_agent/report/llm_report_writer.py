import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()


def is_llm_report_writer_enabled() -> bool:
    """
    Check whether LLM Report Writer is enabled.

    Checks BOTH the legacy ENABLE_LLM_REPORT_WRITER flag and the new
    ENABLE_LLM_ENHANCEMENT flag. Either one can enable the writer.

    Default is disabled — avoids requiring API keys for GitHub demo.
    """
    # Legacy flag
    value = os.getenv("ENABLE_LLM_REPORT_WRITER", "false").lower()
    if value in ("1", "true", "yes", "y"):
        return True
    # New unified flag (from llm/client.py)
    value = os.getenv("ENABLE_LLM_ENHANCEMENT", "false").lower()
    return value in ("1", "true", "yes", "y")


def get_report_llm():
    """
    Create the LLM for Report Writer.

    Prefers the shared client from ``research_agent.llm.client``.
    Falls back to direct ChatOpenAI construction for backward compatibility.

    Returns None if no API key is configured.
    """
    # Try shared client first
    try:
        from research_agent.llm.client import get_chat_llm
        llm = get_chat_llm()
        if llm is not None:
            return llm
    except ImportError:
        pass

    # Fallback: direct ChatOpenAI construction (backward-compatible)
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL") or None

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
    )


def detect_report_style(query: str) -> str:
    """
    根据用户输入判断报告风格。
    """
    query_lower = query.lower()
    query_upper = query.upper()

    if (
        "ppt" in query_lower
        or "PPT" in query_upper
        or "slide" in query_lower
        or "页面" in query
        or "页" in query
    ):
        return "ppt_slide"

    if (
        "摘要" in query
        or "总结" in query
        or "summary" in query_lower
    ):
        return "summary"

    return "group_meeting"


def format_retrieved_docs_for_prompt(
    retrieved_docs: List[Dict],
    max_docs: int = 6,
    max_chars_per_doc: int = 900,
) -> str:
    """
    将 retrieved_docs 格式化为 LLM prompt 中的证据文本。
    """
    if not retrieved_docs:
        return "无检索资料。"

    parts = []

    for i, doc in enumerate(retrieved_docs[:max_docs], start=1):
        metadata = doc.get("metadata", {})
        content = doc.get("content", "")

        path = metadata.get("path", "unknown")
        source_type = metadata.get("source_type", "unknown")
        title = metadata.get("title", "")
        dataset = metadata.get("dataset", "")
        run_tag = metadata.get("run_tag", "")

        header = f"[Evidence {i}] source_type={source_type}; path={path}"

        if title:
            header += f"; title={title}"
        if dataset:
            header += f"; dataset={dataset}"
        if run_tag:
            header += f"; run_tag={run_tag}"

        preview = content[:max_chars_per_doc].replace("\n", " ")

        parts.append(f"{header}\n{preview}")

    return "\n\n".join(parts)


def build_report_prompt(
    query: str,
    retrieved_docs: List[Dict],
    tool_result_text: str = "",
    report_style: Optional[str] = None,
) -> List:
    """
    构建 Evidence-grounded Report Writer prompt。
    """
    style = report_style or detect_report_style(query)
    evidence_text = format_retrieved_docs_for_prompt(retrieved_docs)

    if not tool_result_text:
        tool_result_text = "无工具分析结果。"

    system_prompt = """
你是一个科研汇报写作助手。你必须严格基于用户提供的证据材料生成报告。

重要规则：
1. 只能使用 Evidence 和 Tool Result 中明确出现的信息。
2. 不要编造论文结论、实验数值、数据集属性或不存在的文件路径。
3. 如果资料不足，请明确写“当前资料不足以支持该结论”。
4. 保留清晰的小标题和层次结构。
5. 输出应适合科研组会或 PPT 准备。
6. 不要输出虚假的引用编号，只能引用提供的 Sources 信息。
""".strip()

    user_prompt = f"""
用户需求：
{query}

报告风格：
{style}

Evidence：
{evidence_text}

Tool Result：
{tool_result_text}

请根据报告风格生成中文报告：

如果 report_style = group_meeting：
- 输出组会汇报讲稿
- 包含：研究背景、实验/资料目标、数据与设置、方法流程、关键发现、局限与后续工作

如果 report_style = ppt_slide：
- 输出适合直接放入 PPT 的页面文案
- 用“第 1 页 / 第 2 页 / 第 3 页”组织
- 每页包含标题和 3-5 个要点
- 语言要简洁

如果 report_style = summary：
- 输出 300-500 字摘要
- 强调任务目标、依据和结论边界

请开始生成：
""".strip()

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]


def generate_report_with_llm(
    query: str,
    retrieved_docs: List[Dict],
    tool_result_text: str = "",
    report_style: Optional[str] = None,
) -> Dict:
    """
    使用 LLM 生成报告。

    返回结构：
    {
        "ok": True/False,
        "report_text": "...",
        "report_style": "...",
        "error": "",
        "used_llm": True/False
    }
    """
    style = report_style or detect_report_style(query)

    if not is_llm_report_writer_enabled():
        return {
            "ok": False,
            "used_llm": False,
            "report_style": style,
            "report_text": "",
            "error": "LLM Report Writer is disabled.",
        }

    llm = get_report_llm()

    if llm is None:
        return {
            "ok": False,
            "used_llm": False,
            "report_style": style,
            "report_text": "",
            "error": "OPENAI_API_KEY is not configured.",
        }

    try:
        messages = build_report_prompt(
            query=query,
            retrieved_docs=retrieved_docs,
            tool_result_text=tool_result_text,
            report_style=style,
        )

        response = llm.invoke(messages)

        return {
            "ok": True,
            "used_llm": True,
            "report_style": style,
            "report_text": response.content,
            "error": "",
        }

    except Exception as e:
        return {
            "ok": False,
            "used_llm": False,
            "report_style": style,
            "report_text": "",
            "error": str(e),
        }