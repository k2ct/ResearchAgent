import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .csv_analyzer import analyze_csv_file, format_csv_analysis
from .jsonl_analyzer import analyze_jsonl_file, format_jsonl_analysis


SUPPORTED_TOOL_SUFFIXES = {
    ".csv": "csv_analyzer",
    ".jsonl": "jsonl_analyzer",
}


def extract_file_paths_from_query(query: str) -> List[str]:
    """
    从用户输入中提取可能的文件路径。

    支持：
    - data/experiments/sample_metrics.csv
    - data\\experiments\\sample_metrics.csv
    - F:\\ResearchAgent\\data\\experiments\\sample_metrics.csv
    - xxx.jsonl

    注意：这是一个轻量规则提取器，不追求覆盖所有路径格式。
    """
    pattern = r'[\w\-.\\/:\u4e00-\u9fff]+?\.(?:csv|jsonl)'

    matches = re.findall(pattern, query, flags=re.IGNORECASE)

    # 去掉可能粘上的标点
    cleaned = []
    for match in matches:
        item = match.strip().strip("，。；;,.、()（）[]【】<>《》\"'")
        cleaned.append(item)

    return cleaned


def choose_tool_for_file(file_path: str) -> str:
    """
    根据文件后缀选择工具。
    """
    suffix = Path(file_path).suffix.lower()
    return SUPPORTED_TOOL_SUFFIXES.get(suffix, "unsupported")


def run_file_analysis_tool(file_path: str) -> Dict[str, Any]:
    """
    根据文件类型调用对应工具，并统一返回格式。

    返回：
    {
        "tool_used": "csv_analyzer",
        "ok": True / False,
        "analysis": {...},
        "formatted_text": "..."
    }
    """
    tool_name = choose_tool_for_file(file_path)

    if tool_name == "csv_analyzer":
        analysis = analyze_csv_file(file_path)
        formatted_text = format_csv_analysis(analysis)

    elif tool_name == "jsonl_analyzer":
        analysis = analyze_jsonl_file(file_path)
        formatted_text = format_jsonl_analysis(analysis)

    else:
        analysis = {
            "ok": False,
            "error": "unsupported_file_type",
            "message": f"暂不支持该文件类型：{file_path}",
            "file_path": file_path,
        }
        formatted_text = f"""
文件分析失败：
- error: unsupported_file_type
- message: 暂不支持该文件类型：{file_path}
""".strip()

    return {
        "tool_used": tool_name,
        "ok": analysis.get("ok", False),
        "analysis": analysis,
        "formatted_text": formatted_text,
    }


def run_tool_from_query(query: str) -> Dict[str, Any]:
    """
    从 query 中自动识别文件路径，并调用合适的分析工具。

    如果 query 中没有 CSV / JSONL 路径，则返回 tool_used=none。
    如果有多个文件，Day 12 先只分析第一个。
    """
    file_paths = extract_file_paths_from_query(query)

    if not file_paths:
        return {
            "tool_used": "none",
            "ok": False,
            "analysis": {},
            "formatted_text": "",
            "detected_file_paths": [],
        }

    selected_file = file_paths[0]
    tool_result = run_file_analysis_tool(selected_file) 
    tool_result["detected_file_paths"] = file_paths
    tool_result["selected_file_path"] = selected_file

    return tool_result