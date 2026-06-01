from typing import Any, Dict, List

import pandas as pd

from .file_utils import resolve_project_path, safe_relative_path


def analyze_csv_file(
    file_path: str,
    max_preview_rows: int = 5,
    max_unique_values: int = 10,
) -> Dict[str, Any]:
    """
    分析 CSV 文件，返回结构化摘要。

    当前 Day 10 只做基础分析：
    - 文件是否存在
    - 行列数
    - 列名
    - 数据类型
    - 缺失值统计
    - 数值列 describe
    - 前几行预览
    - 低基数字段的 value_counts
    """
    path = resolve_project_path(file_path)

    if not path.exists():
        return {
            "ok": False,
            "error": "file_not_found",
            "message": f"CSV 文件不存在：{file_path}",
            "file_path": file_path,
        }

    if path.suffix.lower() != ".csv":
        return {
            "ok": False,
            "error": "not_csv",
            "message": f"当前文件不是 CSV：{file_path}",
            "file_path": safe_relative_path(path),
        }

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {
            "ok": False,
            "error": "read_csv_failed",
            "message": str(e),
            "file_path": safe_relative_path(path),
        }

    numeric_summary = _build_numeric_summary(df)
    missing_values = _build_missing_values(df)
    categorical_summary = _build_categorical_summary(
        df,
        max_unique_values=max_unique_values,
    )

    return {
        "ok": True,
        "file_type": "csv",
        "file_path": safe_relative_path(path),
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": missing_values,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "preview_records": df.head(max_preview_rows).fillna("").to_dict(orient="records"),
    }


def _build_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    return {
        col: int(count)
        for col, count in df.isna().sum().items()
        if int(count) > 0
    }


def _build_numeric_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return {}

    desc = numeric_df.describe().fillna(0)

    result: Dict[str, Dict[str, float]] = {}

    for col in numeric_df.columns:
        result[col] = {
            "count": float(desc.loc["count", col]),
            "mean": float(desc.loc["mean", col]),
            "std": float(desc.loc["std", col]) if "std" in desc.index else 0.0,
            "min": float(desc.loc["min", col]),
            "25%": float(desc.loc["25%", col]),
            "50%": float(desc.loc["50%", col]),
            "75%": float(desc.loc["75%", col]),
            "max": float(desc.loc["max", col]),
        }

    return result


def _build_categorical_summary(
    df: pd.DataFrame,
    max_unique_values: int = 10,
) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}

    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)

        if unique_count <= max_unique_values:
            counts = df[col].fillna("MISSING").value_counts().head(max_unique_values)
            result[col] = {
                str(key): int(value)
                for key, value in counts.items()
            }

    return result


def format_csv_analysis(analysis: Dict[str, Any]) -> str:
    """
    将 CSV 分析结果格式化成适合 CLI / Agent 输出的文本。
    """
    if not analysis.get("ok"):
        return f"""
CSV 分析失败：
- error: {analysis.get("error")}
- message: {analysis.get("message")}
""".strip()

    lines: List[str] = []

    lines.append(f"文件：{analysis['file_path']}")
    lines.append(f"类型：{analysis['file_type']}")
    lines.append(f"行数：{analysis['num_rows']}")
    lines.append(f"列数：{analysis['num_columns']}")
    lines.append("")
    lines.append("列名：")
    for col in analysis["columns"]:
        dtype = analysis["dtypes"].get(col, "unknown")
        lines.append(f"- {col} ({dtype})")

    if analysis["missing_values"]:
        lines.append("")
        lines.append("缺失值字段：")
        for col, count in analysis["missing_values"].items():
            lines.append(f"- {col}: {count}")

    if analysis["numeric_summary"]:
        lines.append("")
        lines.append("数值列摘要：")
        for col, stats in analysis["numeric_summary"].items():
            lines.append(
                f"- {col}: mean={stats['mean']:.4f}, "
                f"min={stats['min']:.4f}, "
                f"max={stats['max']:.4f}, "
                f"median={stats['50%']:.4f}"
            )

    if analysis["categorical_summary"]:
        lines.append("")
        lines.append("低基数字段分布：")
        for col, counts in analysis["categorical_summary"].items():
            lines.append(f"- {col}: {counts}")

    lines.append("")
    lines.append("前几行预览：")
    for i, record in enumerate(analysis["preview_records"], start=1):
        lines.append(f"[{i}] {record}")

    return "\n".join(lines)