import json
from collections import Counter, defaultdict
from typing import Any, Dict, List

from .file_utils import resolve_project_path, safe_relative_path


def analyze_jsonl_file(
    file_path: str,
    max_preview_records: int = 5,
    max_value_counts: int = 10,
) -> Dict[str, Any]:
    """
    分析 JSONL 文件，返回结构化摘要。

    当前 Day 11 只做基础分析：
    - 文件是否存在
    - 是否为 .jsonl
    - 总记录数
    - 字段集合
    - 字段类型统计
    - 缺失字段统计
    - bool 字段分布
    - list 字段长度统计
    - 低基数字段 value_counts
    - 前几条记录预览
    """
    path = resolve_project_path(file_path)

    if not path.exists():
        return {
            "ok": False,
            "error": "file_not_found",
            "message": f"JSONL 文件不存在：{file_path}",
            "file_path": file_path,
        }

    if path.suffix.lower() != ".jsonl":
        return {
            "ok": False,
            "error": "not_jsonl",
            "message": f"当前文件不是 JSONL：{file_path}",
            "file_path": safe_relative_path(path),
        }

    records = []
    parse_errors = []

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                    else:
                        parse_errors.append({
                            "line_no": line_no,
                            "error": "json_value_is_not_object",
                        })
                except json.JSONDecodeError as e:
                    parse_errors.append({
                        "line_no": line_no,
                        "error": str(e),
                    })

    except Exception as e:
        return {
            "ok": False,
            "error": "read_jsonl_failed",
            "message": str(e),
            "file_path": safe_relative_path(path),
        }

    if not records:
        return {
            "ok": False,
            "error": "no_valid_records",
            "message": "未读取到有效 JSON object 记录。",
            "file_path": safe_relative_path(path),
            "parse_errors": parse_errors[:max_preview_records],
        }

    fields = sorted(_collect_fields(records))
    field_types = _build_field_types(records, fields)
    missing_fields = _build_missing_fields(records, fields)
    boolean_summary = _build_boolean_summary(records, fields)
    list_field_summary = _build_list_field_summary(records, fields)
    value_counts = _build_value_counts(
        records,
        fields,
        max_value_counts=max_value_counts,
    )

    return {
        "ok": True,
        "file_type": "jsonl",
        "file_path": safe_relative_path(path),
        "num_records": len(records),
        "fields": fields,
        "field_types": field_types,
        "missing_fields": missing_fields,
        "boolean_summary": boolean_summary,
        "list_field_summary": list_field_summary,
        "value_counts": value_counts,
        "parse_error_count": len(parse_errors),
        "parse_errors_preview": parse_errors[:max_preview_records],
        "preview_records": records[:max_preview_records],
    }


def _collect_fields(records: List[Dict[str, Any]]) -> set:
    fields = set()

    for record in records:
        fields.update(record.keys())

    return fields


def _get_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _build_field_types(
    records: List[Dict[str, Any]],
    fields: List[str],
) -> Dict[str, Dict[str, int]]:
    result = {}

    for field in fields:
        counter = Counter()

        for record in records:
            if field in record:
                counter[_get_type_name(record[field])] += 1
            else:
                counter["missing"] += 1

        result[field] = dict(counter)

    return result


def _build_missing_fields(
    records: List[Dict[str, Any]],
    fields: List[str],
) -> Dict[str, int]:
    result = {}

    for field in fields:
        missing_count = sum(1 for record in records if field not in record)
        if missing_count > 0:
            result[field] = missing_count

    return result


def _build_boolean_summary(
    records: List[Dict[str, Any]],
    fields: List[str],
) -> Dict[str, Dict[str, int]]:
    result = {}

    for field in fields:
        values = [
            record[field]
            for record in records
            if field in record and isinstance(record[field], bool)
        ]

        if not values:
            continue

        counter = Counter(values)

        result[field] = {
            "true": int(counter.get(True, 0)),
            "false": int(counter.get(False, 0)),
        }

    return result


def _build_list_field_summary(
    records: List[Dict[str, Any]],
    fields: List[str],
) -> Dict[str, Dict[str, float]]:
    result = {}

    for field in fields:
        lengths = [
            len(record[field])
            for record in records
            if field in record and isinstance(record[field], list)
        ]

        if not lengths:
            continue

        result[field] = {
            "count": len(lengths),
            "mean_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    return result


def _build_value_counts(
    records: List[Dict[str, Any]],
    fields: List[str],
    max_value_counts: int = 10,
) -> Dict[str, Dict[str, int]]:
    """
    对简单标量字段做 value_counts。
    跳过 list / dict 等复杂字段。
    """
    result = {}

    for field in fields:
        values = []

        for record in records:
            if field not in record:
                continue

            value = record[field]

            if isinstance(value, (list, dict)):
                continue

            values.append(value)

        unique_values = set(str(v) for v in values)

        if 0 < len(unique_values) <= max_value_counts:
            counter = Counter(str(v) for v in values)
            result[field] = {
                key: int(value)
                for key, value in counter.most_common(max_value_counts)
            }

    return result


def format_jsonl_analysis(analysis: Dict[str, Any]) -> str:
    """
    将 JSONL 分析结果格式化为适合 CLI / Agent 输出的文本。
    """
    if not analysis.get("ok"):
        return f"""
JSONL 分析失败：
- error: {analysis.get("error")}
- message: {analysis.get("message")}
- file_path: {analysis.get("file_path")}
""".strip()

    lines: List[str] = []

    lines.append(f"文件：{analysis['file_path']}")
    lines.append(f"类型：{analysis['file_type']}")
    lines.append(f"记录数：{analysis['num_records']}")
    lines.append(f"解析错误数：{analysis['parse_error_count']}")
    lines.append("")

    lines.append("字段列表：")
    for field in analysis["fields"]:
        field_type_summary = analysis["field_types"].get(field, {})
        lines.append(f"- {field}: {field_type_summary}")

    if analysis["missing_fields"]:
        lines.append("")
        lines.append("缺失字段统计：")
        for field, count in analysis["missing_fields"].items():
            lines.append(f"- {field}: {count}")

    if analysis["boolean_summary"]:
        lines.append("")
        lines.append("布尔字段分布：")
        for field, counts in analysis["boolean_summary"].items():
            lines.append(f"- {field}: true={counts['true']}, false={counts['false']}")

    if analysis["list_field_summary"]:
        lines.append("")
        lines.append("列表字段长度统计：")
        for field, stats in analysis["list_field_summary"].items():
            lines.append(
                f"- {field}: count={stats['count']}, "
                f"mean_length={stats['mean_length']:.2f}, "
                f"min={stats['min_length']}, max={stats['max_length']}"
            )

    if analysis["value_counts"]:
        lines.append("")
        lines.append("低基数字段分布：")
        for field, counts in analysis["value_counts"].items():
            lines.append(f"- {field}: {counts}")

    lines.append("")
    lines.append("前几条记录预览：")
    for i, record in enumerate(analysis["preview_records"], start=1):
        lines.append(f"[{i}] {record}")

    return "\n".join(lines)