from typing import Dict, List, Optional

from langchain_core.documents import Document

from .indexer import load_vector_store
from .schemas import (
    SOURCE_TYPE_PAPER,
    SOURCE_TYPE_EXPERIMENT,
    SOURCE_TYPE_DATASET,
)


TASK_TYPE_TO_SOURCE_TYPE = {
    "paper_question": SOURCE_TYPE_PAPER,
    "experiment_analysis": SOURCE_TYPE_EXPERIMENT,
    "dataset_recommendation": SOURCE_TYPE_DATASET,

    # 兼容一些简写
    "paper": SOURCE_TYPE_PAPER,
    "experiment": SOURCE_TYPE_EXPERIMENT,
    "dataset": SOURCE_TYPE_DATASET,
}


def get_source_type_for_task(task_type: str) -> Optional[str]:
    """
    根据 task_type 决定应该检索哪类资料。

    返回：
    - paper_question -> paper_note
    - experiment_analysis -> experiment_doc
    - dataset_recommendation -> dataset_doc
    - code_help / general / report_generation -> None
    """
    return TASK_TYPE_TO_SOURCE_TYPE.get(task_type)


def build_metadata_filter(task_type: str) -> Optional[Dict]:
    """
    构建 Chroma metadata filter。

    Chroma 的简单过滤形式：
    {"source_type": "dataset_doc"}

    如果返回 None，表示不使用 metadata filter，检索全部文档。
    """
    source_type = get_source_type_for_task(task_type)

    if source_type is None:
        return None

    return {
        "source_type": source_type
    }


def retrieve_documents(
    query: str,
    task_type: str,
    top_k: int = 3,
    use_filter: bool = True,
) -> List[Document]:
    """
    根据 query 和 task_type 检索相关文档。

    参数：
    - query: 用户问题
    - task_type: 阶段一 classify_task 得到的任务类型
    - top_k: 返回前几个结果
    - use_filter: 是否启用 metadata filter

    返回：
    - List[Document]
    """
    vector_store = load_vector_store()

    metadata_filter = build_metadata_filter(task_type) if use_filter else None

    if metadata_filter:
        docs = vector_store.similarity_search(
            query=query,
            k=top_k,
            filter=metadata_filter,
        )
    else:
        docs = vector_store.similarity_search(
            query=query,
            k=top_k,
        )

    return docs


def document_to_dict(doc: Document) -> Dict:
    """
    把 LangChain Document 转成普通 dict，方便后续放进 AgentState。
    """
    return {
        "content": doc.page_content,
        "metadata": dict(doc.metadata),
    }


def retrieve_documents_as_dicts(
    query: str,
    task_type: str,
    top_k: int = 3,
    use_filter: bool = True,
) -> List[Dict]:
    """
    检索文档，并转成 dict 列表。

    Day 9 接入 LangGraph 时，AgentState 里建议保存 dict，
    而不是直接保存 Document 对象。
    """
    docs = retrieve_documents(
        query=query,
        task_type=task_type,
        top_k=top_k,
        use_filter=use_filter,
    )

    return [document_to_dict(doc) for doc in docs]


def format_retrieved_docs(
    docs: List[Document],
    max_chars_per_doc: int = 500,
) -> str:
    """
    把检索到的 Document 格式化成可读文本。
    后续可以给 final_answer 或 LLM prompt 使用。
    """
    if not docs:
        return "未检索到相关资料。"

    formatted_parts = []

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata
        content = doc.page_content[:max_chars_per_doc].replace("\n", " ")

        source = metadata.get("path", "unknown")
        source_type = metadata.get("source_type", "unknown")
        title = metadata.get("title", "")
        dataset = metadata.get("dataset", "")
        run_tag = metadata.get("run_tag", "")

        header_parts = [
            f"[{i}]",
            f"source_type={source_type}",
            f"path={source}",
        ]

        if title:
            header_parts.append(f"title={title}")

        if dataset:
            header_parts.append(f"dataset={dataset}")

        if run_tag:
            header_parts.append(f"run_tag={run_tag}")

        header = " | ".join(header_parts)

        formatted_parts.append(
            f"{header}\n{content}"
        )

    return "\n\n".join(formatted_parts)


def extract_sources_from_docs(docs: List[Document]) -> List[Dict]:
    """
    从检索结果中提取 sources，去重后用于最终回答展示。
    """
    seen = set()
    sources = []

    for doc in docs:
        metadata = doc.metadata
        path = metadata.get("path", "unknown")

        if path in seen:
            continue

        seen.add(path)

        sources.append({
            "path": path,
            "source_type": metadata.get("source_type", "unknown"),
            "title": metadata.get("title", ""),
            "dataset": metadata.get("dataset", ""),
            "run_tag": metadata.get("run_tag", ""),
        })

    return sources


# 十、可选优化：加入 score

#如果你想看相似度分数，可以额外在 retriever.py 加一个函数：

def retrieve_documents_with_scores(
    query: str,
    task_type: str,
    top_k: int = 3,
    use_filter: bool = True,
):
    vector_store = load_vector_store()

    metadata_filter = build_metadata_filter(task_type) if use_filter else None

    if metadata_filter:
        results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=metadata_filter,
        )
    else:
        results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
        )

    return results