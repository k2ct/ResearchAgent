import json
import os
from typing import Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


VALID_TASK_TYPES = {
    "paper_question",
    "experiment_analysis",
    "dataset_recommendation",
    "report_generation",
    "code_help",
    "general",
}


def get_llm_classifier_enabled() -> bool:
    """
    是否启用 LLM 分类器。
    通过 .env 里的 USE_LLM_CLASSIFIER 控制。
    """
    load_dotenv()
    value = os.getenv("USE_LLM_CLASSIFIER", "false").lower()
    return value in {"1", "true", "yes", "y"}


def build_classifier_prompt(query: str) -> str:
    """
    构造任务分类 Prompt。
    要求模型只输出 JSON，方便解析。
    """
    return f"""
你是一个科研 Agent 的任务分类器。

请根据用户输入，将任务分类为以下六类之一：

1. paper_question
   - 论文总结、论文问答、文献解释、论文对比

2. experiment_analysis
   - 实验结果分析、CSV/JSONL 输出分析、benchmark 结果解释、幻觉指标分析

3. dataset_recommendation
   - 数据集推荐、数据集选择、数据集构建建议

4. report_generation
   - 组会汇报、PPT 文案、阶段总结、实验汇报文本

5. code_help
   - 代码解释、代码报错、脚本修改、环境问题、bug 修复

6. general
   - 其他通用科研助手问题

用户输入：
{query}

请只输出 JSON，不要输出 Markdown，不要解释。

JSON 格式如下：
{{
  "task_type": "paper_question | experiment_analysis | dataset_recommendation | report_generation | code_help | general",
  "reason": "一句话说明分类原因"
}}
""".strip()


def classify_with_llm(query: str) -> Dict[str, str]:
    """
    使用 LLM 对用户问题进行分类。
    如果调用失败或解析失败，会抛出异常，由上层 fallback。
    """
    load_dotenv()

    model_name = os.getenv("OPENAI_MODEL", "deepseek-v4-pro")
    base_url = os.getenv("OPENAI_BASE_URL") or None

    llm = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        temperature=0,
    )

    prompt = build_classifier_prompt(query)
    response = llm.invoke(prompt)

    content = response.content.strip()

    data = json.loads(content)

    task_type = data.get("task_type", "").strip()
    reason = data.get("reason", "").strip()

    if task_type not in VALID_TASK_TYPES:
        raise ValueError(f"Invalid task_type from LLM: {task_type}")

    return {
        "task_type": task_type,
        "reason": reason,
    }