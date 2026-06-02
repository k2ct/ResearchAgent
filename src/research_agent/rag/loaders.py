from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

from .schemas import (
    DIR_TO_SOURCE_TYPE,
    REQUIRED_METADATA_KEYS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


def read_text_file(path: Path) -> str:
    """
    读取文本文件，统一使用 utf-8 编码。
    """
    return path.read_text(encoding="utf-8")


def normalize_metadata_value(value: str):
    """
    对 metadata 字段做简单类型规范化。

    例如：
    - year: "2026" -> 2026
    - 空字符串 -> 不保留
    """
    value = value.strip()

    if not value:
        return None

    if value.isdigit():
        return int(value)

    return value


def parse_basic_info(markdown_text: str) -> Dict:
    """
    从 markdown 的 “## 基本信息” 部分解析 metadata。

    支持这种格式：

    ## 基本信息

    - source_type: paper_note
    - title: xxx
    - topic: multimodal_bias
    - year: 2026
    - path: data/papers/xxx.md

    返回：
    {
        "source_type": "paper_note",
        "title": "...",
        ...
    }
    """
    metadata = {}

    lines = markdown_text.splitlines()
    in_basic_info = False

    for line in lines:
        stripped = line.strip()

        # 找到基本信息段落
        if stripped == "## 基本信息":
            in_basic_info = True
            continue

        # 如果已经进入基本信息段落，遇到下一个二级标题就结束
        if in_basic_info and stripped.startswith("## ") and stripped != "## 基本信息":
            break

        if not in_basic_info:
            continue

        # 解析 "- key: value"
        if stripped.startswith("- ") and ":" in stripped:
            item = stripped[2:]
            key, value = item.split(":", 1)
            key = key.strip()
            value = normalize_metadata_value(value)

            if value is not None:
                metadata[key] = value

    return metadata


def infer_source_type_from_path(path: Path) -> Optional[str]:
    """
    根据文件所在目录推断 source_type。

    data/papers/*.md      -> paper_note
    data/experiments/*.md -> experiment_doc
    data/datasets/*.md    -> dataset_doc
    """
    parent_name = path.parent.name
    return DIR_TO_SOURCE_TYPE.get(parent_name)


def load_markdown_document(path: Path, project_root: Path = PROJECT_ROOT) -> Document:
    """
    加载单个 markdown 文件，返回 LangChain Document。
    """
    markdown_text = read_text_file(path)
    metadata = parse_basic_info(markdown_text)

    # 如果文档里没有 source_type，就根据目录推断
    if "source_type" not in metadata:
        inferred_source_type = infer_source_type_from_path(path)
        if inferred_source_type:
            metadata["source_type"] = inferred_source_type

    # 如果文档里没有 path，就自动写相对路径
    if "path" not in metadata:
        metadata["path"] = str(path.relative_to(project_root)).replace("\\", "/")

    # 如果文档里没有 title，就用一级标题或文件名
    if "title" not in metadata:
        metadata["title"] = extract_title(markdown_text) or path.stem

    validate_metadata(metadata, path)

    return Document(
        page_content=markdown_text,
        metadata=metadata,
    )


def extract_title(markdown_text: str) -> Optional[str]:
    """
    提取 markdown 第一个一级标题作为 title。
    """
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.replace("# ", "", 1).strip()
    return None


def validate_metadata(metadata: Dict, path: Path) -> None:
    """
    检查必要 metadata 字段是否存在。
    """
    missing_keys = [
        key for key in REQUIRED_METADATA_KEYS
        if key not in metadata or metadata[key] in ["", None]
    ]

    if missing_keys:
        raise ValueError(
            f"Missing required metadata keys {missing_keys} in file: {path}"
        )


def load_markdown_documents(directory: Path) -> List[Document]:
    """
    加载某个目录下所有 markdown 文件。
    """
    if not directory.exists():
        return []

    documents = []

    for path in sorted(directory.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue

        doc = load_markdown_document(path)
        documents.append(doc)

    return documents


def load_all_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    """
    加载 data/papers、data/experiments、data/datasets 下所有 markdown 文档。
    """
    documents = []

    for subdir in ["papers", "experiments", "datasets"]:
        directory = data_dir / subdir
        documents.extend(load_markdown_documents(directory))

    return documents


def summarize_documents(documents: List[Document]) -> Dict[str, int]:
    """
    按 source_type 统计文档数量，方便测试。
    """
    summary = {}

    for doc in documents:
        source_type = doc.metadata.get("source_type", "unknown")
        summary[source_type] = summary.get(source_type, 0) + 1

    return summary


def format_document_preview(doc: Document, max_chars: int = 200) -> str:
    """
    格式化单个 Document 的预览信息。
    """
    content_preview = doc.page_content[:max_chars].replace("\n", " ")

    metadata_lines = [
        f"{key}: {value}"
        for key, value in doc.metadata.items()
    ]

    metadata_text = "\n".join(metadata_lines)

    return f"""
Metadata:
{metadata_text}

Content Preview:
{content_preview}...
""".strip()