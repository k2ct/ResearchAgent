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
    Read a text file using utf-8 encoding.
    """
    return path.read_text(encoding="utf-8")


def normalize_metadata_value(value: str):
    """
    Normalize a metadata string value.

    - year: "2026" -> 2026
    - empty string -> None (skip)
    """
    value = value.strip()

    if not value:
        return None

    if value.isdigit():
        return int(value)

    return value


def parse_basic_info(markdown_text: str) -> Dict:
    """
    Parse metadata from "## Basic Info" section in markdown.

    Supports this format:

    ## Basic Info

    - source_type: paper_note
    - title: xxx
    - topic: multimodal_bias
    - year: 2026
    - path: data/papers/xxx.md

    Returns:
        {"source_type": "paper_note", "title": "...", ...}
    """
    metadata = {}

    lines = markdown_text.splitlines()
    in_basic_info = False

    for line in lines:
        stripped = line.strip()

        # Find the basic info section
        if stripped == "## Basic Info" or stripped == "## 基本信息":
            in_basic_info = True
            continue

        # Stop at next h2 heading
        if in_basic_info and stripped.startswith("## ") and stripped != "## Basic Info" and stripped != "## 基本信息":
            break

        if not in_basic_info:
            continue

        # Parse "- key: value"
        if stripped.startswith("- ") and ":" in stripped:
            item = stripped[2:]
            key, value = item.split(":", 1)
            key = key.strip()
            value = normalize_metadata_value(value)

            if value is not None:
                metadata[key] = value

    return metadata


def parse_front_matter(markdown_text: str) -> Dict:
    """
    Parse YAML front matter from the start of a markdown document.

    Front matter is delimited by ``---`` on the first line and a matching
    ``---`` on a subsequent line. Returns an empty dict if no front matter
    is found or if the YAML is invalid.

    Used by the ingestion pipeline (data/ingested/) and provides a richer
    metadata format than the legacy ``## Basic Info`` section.
    """
    import yaml

    lines = markdown_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    # Find closing --- delimiter
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return {}

    yaml_text = "\n".join(lines[1:end_idx])

    try:
        raw = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        return {}

    if not isinstance(raw, dict):
        return {}

    metadata = {}
    for key, value in raw.items():
        if value is None:
            continue
        key_str = str(key)
        # Normalize string values (e.g. year: "2026" -> 2026)
        if isinstance(value, str):
            normalized = normalize_metadata_value(value)
            if normalized is not None:
                metadata[key_str] = normalized
        else:
            metadata[key_str] = value

    return metadata


def infer_source_type_from_path(path: Path) -> Optional[str]:
    """
    Infer source_type from parent directory name.

    data/papers/*.md      -> paper_note
    data/experiments/*.md -> experiment_doc
    data/datasets/*.md    -> dataset_doc
    """
    parent_name = path.parent.name
    return DIR_TO_SOURCE_TYPE.get(parent_name)


def load_markdown_document(path: Path, project_root: Path = PROJECT_ROOT) -> Document:
    """
    Load a single markdown file as a LangChain Document.
    """
    markdown_text = read_text_file(path)
    # Try YAML front matter first (ingested docs), fall back to ## Basic Info (legacy docs)
    metadata = parse_front_matter(markdown_text)
    if not metadata:
        metadata = parse_basic_info(markdown_text)

    # If no source_type in document, infer from directory
    if "source_type" not in metadata:
        inferred_source_type = infer_source_type_from_path(path)
        if inferred_source_type:
            metadata["source_type"] = inferred_source_type

    # Auto-fill path if missing
    if "path" not in metadata:
        metadata["path"] = str(path.relative_to(project_root)).replace("\\", "/")

    # Auto-fill title if missing
    if "title" not in metadata:
        metadata["title"] = extract_title(markdown_text) or path.stem

    validate_metadata(metadata, path)

    return Document(
        page_content=markdown_text,
        metadata=metadata,
    )


def extract_title(markdown_text: str) -> Optional[str]:
    """
    Extract the first H1 heading as the title.
    """
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.replace("# ", "", 1).strip()
    return None


def validate_metadata(metadata: Dict, path: Path) -> None:
    """
    Check that required metadata keys are present and non-empty.
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
    Load all markdown files from a directory.
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
    Load all markdown documents from data/papers, data/experiments,
    data/datasets, and data/ingested.
    """
    documents = []

    for subdir in ["papers", "experiments", "datasets", "ingested"]:
        directory = data_dir / subdir
        documents.extend(load_markdown_documents(directory))

    return documents


def summarize_documents(documents: List[Document]) -> Dict[str, int]:
    """
    Count documents by source_type.
    """
    summary = {}

    for doc in documents:
        source_type = doc.metadata.get("source_type", "unknown")
        summary[source_type] = summary.get(source_type, 0) + 1

    return summary


def format_document_preview(doc: Document, max_chars: int = 200) -> str:
    """
    Format a human-readable preview of a Document.
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
