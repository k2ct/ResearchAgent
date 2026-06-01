from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(path: str) -> Path:
    """
    将用户输入路径解析为绝对路径。

    支持：
    1. 绝对路径：F:\\ResearchAgent\\data\\xxx.csv
    2. 相对路径：data/xxx.csv
    3. 项目内路径：outputs/xxx.csv
    """
    input_path = Path(path)

    if input_path.is_absolute():
        return input_path

    return PROJECT_ROOT / input_path


def file_exists(path: str) -> bool:
    return resolve_project_path(path).exists()


def get_file_suffix(path: str) -> str:
    return resolve_project_path(path).suffix.lower()


def safe_relative_path(path: Path) -> str:
    """
    尽量返回相对项目根目录的路径，方便展示。
    """
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")