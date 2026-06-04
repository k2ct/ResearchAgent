"""
Hybrid RAG v1 — Vector Search + Keyword Search + Score Fusion

独立模块，不依赖对 retriever.py / indexer.py / loaders.py 的修改。
暂不接入主 Agent 流程；通过 scripts/test_hybrid_retriever.py 独立测试。

Author: Hybrid RAG v1 (standalone module)
"""

from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .indexer import load_vector_store
from .loaders import load_all_documents
from .schemas import (
    SOURCE_TYPE_PAPER,
    SOURCE_TYPE_EXPERIMENT,
    SOURCE_TYPE_DATASET,
)


# ---------------------------------------------------------------------------
# 1. 中英文混合 tokenizer（不引入 jieba 等新依赖）
# ---------------------------------------------------------------------------

def tokenize_text(text: str) -> List[str]:
    """
    中英文混合分词。

    规则：
    - 英文按单词切分（字母序列）
    - 中文按连续中文字符片段切分
    - 保留数字、下划线、连字符相关 token（例如 coco_val_n300_g1, hrs_v1）
    - 过滤空白 token
    """
    if not text:
        return []

    tokens: List[str] = []
    i = 0
    n = len(text)
    current = ""

    while i < n:
        ch = text[i]

        # 空白字符：结束当前 token
        if ch.isspace():
            if current:
                tokens.append(current)
                current = ""
            i += 1
            continue

        # 标点 / 特殊字符（但保留下划线和连字符，它们在 token 中间有意义）
        if ch in ",.!?;:()[]{}，。！？；：""''（）【】《》\"'":
            if current:
                tokens.append(current)
                current = ""
            i += 1
            continue

        # 中文字符（Unicode 范围）
        if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            if current and not _is_cjk(current[-1]):
                # 前一个 token 是英文/数字，先保存
                tokens.append(current)
                current = ""
            current += ch
            i += 1
            # 中文字符每个独立成 token（字符级切分），
            # 同时也把连续中文整体记录下来
            continue

        # 其他可打印字符（字母、数字、下划线、连字符等）
        current += ch
        i += 1

    if current:
        tokens.append(current)

    # 对中文连续片段做进一步处理：每个字符单独成 token，
    # 同时保留原连续片段作为 token
    expanded: List[str] = []
    for token in tokens:
        if token and _is_all_cjk(token):
            # 连续中文：既保留整体，也保留每个字符
            expanded.append(token)
            expanded.extend(list(token))
        else:
            expanded.append(token)

    # 过滤空白、纯标点、过短无意义 token
    result = []
    for t in expanded:
        t = t.strip().lower()
        if not t:
            continue
        if len(t) == 1 and t in ",.!?;:()[]{}，。！？；：""''（）【】《》\"'-_":
            continue
        result.append(t)

    return result


def _is_cjk(ch: str) -> bool:
    """判断单个字符是否为 CJK 汉字。"""
    return '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿'


def _is_all_cjk(s: str) -> bool:
    """判断字符串是否全为 CJK 汉字。"""
    return all(_is_cjk(ch) for ch in s)


# ---------------------------------------------------------------------------
# 2. task_type -> source_type 本地映射
# ---------------------------------------------------------------------------

TASK_TYPE_TO_SOURCE_TYPE_LOCAL = {
    "paper_question": SOURCE_TYPE_PAPER,
    "experiment_analysis": SOURCE_TYPE_EXPERIMENT,
    "dataset_recommendation": SOURCE_TYPE_DATASET,
    # 兼容简写
    "paper": SOURCE_TYPE_PAPER,
    "experiment": SOURCE_TYPE_EXPERIMENT,
    "dataset": SOURCE_TYPE_DATASET,
}


def get_source_type_for_task_local(task_type: str) -> Optional[str]:
    """
    根据 task_type 决定应该检索哪类资料（本地版，不依赖 retriever.py）。

    返回：
    - paper_question -> paper_note
    - experiment_analysis -> experiment_doc
    - dataset_recommendation -> dataset_doc
    - report_generation / general / code_help -> None（查全库）
    """
    return TASK_TYPE_TO_SOURCE_TYPE_LOCAL.get(task_type)


# ---------------------------------------------------------------------------
# 3. 按 task_type 过滤文档
# ---------------------------------------------------------------------------

def filter_documents_by_task(
    docs: List[Document],
    task_type: str,
) -> List[Document]:
    """
    根据 metadata["source_type"] 过滤文档。

    如果 get_source_type_for_task_local(task_type) 返回 None，则不过滤。
    """
    target = get_source_type_for_task_local(task_type)
    if target is None:
        return list(docs)

    return [
        doc for doc in docs
        if doc.metadata.get("source_type") == target
    ]


# ---------------------------------------------------------------------------
# 4. 构建 keyword search 用文本
# ---------------------------------------------------------------------------

def build_keyword_text(doc: Document) -> str:
    """
    把以下信息拼接成可搜索文本，使 keyword search 能命中专有名词、
    路径、run_tag、指标字段：

    - doc.page_content
    - metadata["title"]
    - metadata["path"]
    - metadata["dataset"]
    - metadata["run_tag"]
    - metadata["source_type"]
    - metadata["topic"]
    - metadata["tags"]
    """
    metadata = doc.metadata
    parts = [doc.page_content]

    for key in ["title", "path", "dataset", "run_tag", "source_type", "topic", "tags"]:
        value = metadata.get(key, "")
        if value:
            if isinstance(value, list):
                parts.append(" ".join(str(v) for v in value))
            else:
                parts.append(str(value))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# 5. Keyword Search
# ---------------------------------------------------------------------------

def keyword_search_documents(
    query: str,
    task_type: str,
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Keyword-based document search.

    流程：
    1. 加载所有原始 markdown documents
    2. 按 task_type 过滤
    3. 对 query 和每个 doc 计算关键词分数
    4. 返回 top_k 结果（分数越大越相关）

    不使用 rank_bm25，不新增依赖。
    """
    all_docs = load_all_documents()
    filtered_docs = filter_documents_by_task(all_docs, task_type)

    if not filtered_docs:
        return []

    query_tokens = tokenize_text(query)

    # 提取 query 中的专有名词片段（连续字母数字下划线连字符序列）
    query_noun_phrases = _extract_noun_phrases(query)

    scored: List[Tuple[Document, float]] = []

    for doc in filtered_docs:
        search_text = build_keyword_text(doc)
        doc_tokens = tokenize_text(search_text)

        # 用 token set 加速查找
        doc_token_set = set(doc_tokens)

        score = 0.0

        # 1) query token 在 doc 中出现：+1
        for qt in query_tokens:
            if qt in doc_token_set:
                score += 1.0

        # 2) query token 在 title / path / dataset / run_tag 中出现：额外 +2
        metadata_bonus_text = " ".join(
            str(doc.metadata.get(k, ""))
            for k in ["title", "path", "dataset", "run_tag"]
        )
        metadata_bonus_tokens = set(tokenize_text(metadata_bonus_text))
        for qt in query_tokens:
            if qt in metadata_bonus_tokens:
                score += 2.0

        # 3) 完整 query 中的专有名词片段在文档中出现：额外 +3
        for phrase in query_noun_phrases:
            if phrase.lower() in search_text.lower():
                score += 3.0

        if score > 0:
            scored.append((doc, score))

    # 按分数降序排序
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def _extract_noun_phrases(text: str) -> List[str]:
    """
    从文本中提取专有名词片段（连续字母数字下划线连字符序列，长度 >= 3）。
    例如：coco_val_n300_g1, OpenImages-MIAP, hrs_v1, Guardrail-Agnostic
    """
    import re
    pattern = r'[a-zA-Z0-9][a-zA-Z0-9_\-]*[a-zA-Z0-9]|[a-zA-Z0-9]'
    matches = re.findall(pattern, text)
    # 过滤过短的匹配
    return [m for m in matches if len(m) >= 3]


# ---------------------------------------------------------------------------
# 6. Vector Search
# ---------------------------------------------------------------------------

def build_metadata_filter_local(task_type: str) -> Optional[Dict]:
    """
    构建 Chroma metadata filter（本地版，不依赖 retriever.py）。

    如果返回 None，表示不使用 metadata filter，检索全部文档。
    """
    source_type = get_source_type_for_task_local(task_type)
    if source_type is None:
        return None
    return {"source_type": source_type}


def vector_search_documents(
    query: str,
    task_type: str,
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Vector-based document search using Chroma。

    流程：
    1. 加载 Chroma 向量库
    2. 根据 task_type 构造 metadata filter
    3. 执行 similarity_search_with_score
    4. 将 Chroma distance 转为 similarity（越大越相关）
    5. 返回 List[Tuple[Document, float]]
    """
    vector_store = load_vector_store()
    metadata_filter = build_metadata_filter_local(task_type)

    if metadata_filter:
        raw_results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=metadata_filter,
        )
    else:
        raw_results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
        )

    # Chroma 返回的 score 是 distance（越小越相关），转成 similarity
    results: List[Tuple[Document, float]] = []
    for doc, distance in raw_results:
        similarity = 1.0 / (1.0 + distance)
        results.append((doc, similarity))

    return results


# ---------------------------------------------------------------------------
# 7. Score Normalization
# ---------------------------------------------------------------------------

def normalize_scores(
    results: List[Tuple[Document, float]],
) -> List[Tuple[Document, float]]:
    """
    Min-max normalization。

    输入 List[Tuple[Document, float]]
    输出 List[Tuple[Document, normalized_score]]

    如果所有分数相同，则都设为 1.0。
    如果列表为空，返回空列表。
    """
    if not results:
        return []

    scores = [s for _, s in results]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [(doc, 1.0) for doc, _ in results]

    normalized = []
    for doc, score in results:
        norm_score = (score - min_score) / (max_score - min_score)
        normalized.append((doc, norm_score))

    return normalized


# ---------------------------------------------------------------------------
# 8. 文档去重 key
# ---------------------------------------------------------------------------

def doc_key(doc: Document) -> str:
    """
    生成文档去重标识。

    使用 metadata["path"] + page_content 前 120 字符。
    """
    path = doc.metadata.get("path", "")
    content_prefix = doc.page_content[:120]
    return f"{path}::{content_prefix}"


# ---------------------------------------------------------------------------
# 9. Score Fusion
# ---------------------------------------------------------------------------

def fuse_retrieval_results(
    vector_results: List[Tuple[Document, float]],
    keyword_results: List[Tuple[Document, float]],
    top_k: int = 5,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List[Tuple[Document, float]]:
    """
    融合 vector search 和 keyword search 的结果。

    流程：
    1. 对两路结果分别做 min-max normalization
    2. 按 doc_key 合并
    3. final_score = vector_weight * vector_score + keyword_weight * keyword_score
    4. 如果文档只出现在一路结果中，另一路分数为 0
    5. 按 final_score 降序排序
    6. 截断 top_k
    """
    # 分别归一化
    norm_vector = normalize_scores(vector_results)
    norm_keyword = normalize_scores(keyword_results)

    # 建立索引
    vector_map: Dict[str, float] = {}
    for doc, score in norm_vector:
        key = doc_key(doc)
        # 如果同一个 key 出现多次，取最高分
        if key not in vector_map or score > vector_map[key]:
            vector_map[key] = score

    keyword_map: Dict[str, float] = {}
    keyword_doc_map: Dict[str, Document] = {}
    for doc, score in norm_keyword:
        key = doc_key(doc)
        if key not in keyword_map or score > keyword_map[key]:
            keyword_map[key] = score
            keyword_doc_map[key] = doc

    # 保存 vector doc 引用
    vector_doc_map: Dict[str, Document] = {}
    for doc, _ in norm_vector:
        key = doc_key(doc)
        if key not in vector_doc_map:
            vector_doc_map[key] = doc

    # 合并所有 key
    all_keys = set(vector_map.keys()) | set(keyword_map.keys())

    fused: List[Tuple[Document, float]] = []
    for key in all_keys:
        v_score = vector_map.get(key, 0.0)
        k_score = keyword_map.get(key, 0.0)
        final_score = vector_weight * v_score + keyword_weight * k_score

        # 优先使用 vector 结果中的 doc（chunk 级），回退到 keyword doc
        doc = vector_doc_map.get(key) or keyword_doc_map.get(key)
        if doc is not None:
            fused.append((doc, final_score))

    # 按 final_score 降序
    fused.sort(key=lambda x: x[1], reverse=True)

    return fused[:top_k]


# ---------------------------------------------------------------------------
# 10. 主检索函数（带分数）
# ---------------------------------------------------------------------------

def hybrid_retrieve_documents_with_scores(
    query: str,
    task_type: str,
    top_k: int = 5,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    use_reranker: bool = False,
) -> List[Tuple[Document, float]]:
    """
    Hybrid RAG v1 main retrieval function.

    Flow:
    1. vector_top_k = max(top_k * 2, 8)
    2. keyword_top_k = max(top_k * 2, 8)
    3. Call vector_search_documents and keyword_search_documents
    4. Call fuse_retrieval_results to merge
    5. Optionally rerank with heuristic reranker
    6. Return List[Tuple[Document, float]] sorted by fusion score descending

    Parameters
    ----------
    use_reranker : bool
        If True, apply heuristic reranker after fusion (default: False).
    """
    vector_top_k = max(top_k * 2, 8)
    keyword_top_k = max(top_k * 2, 8)

    vector_results = vector_search_documents(
        query=query,
        task_type=task_type,
        top_k=vector_top_k,
    )

    keyword_results = keyword_search_documents(
        query=query,
        task_type=task_type,
        top_k=keyword_top_k,
    )

    fused = fuse_retrieval_results(
        vector_results=vector_results,
        keyword_results=keyword_results,
        top_k=top_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )

    if use_reranker:
        from research_agent.rag.reranker import rerank_candidates
        fused = rerank_candidates(query, fused, top_k=top_k)

    return fused


# ---------------------------------------------------------------------------
# 11. 主检索函数（不带分数，方便接入主流程）
# ---------------------------------------------------------------------------

def hybrid_retrieve_documents(
    query: str,
    task_type: str,
    top_k: int = 5,
    use_reranker: bool = False,
) -> List[Document]:
    """
    Hybrid RAG v1 retrieval, returning Document list only (no scores).

    Convenience wrapper for integration into retriever.py main flow.

    Parameters
    ----------
    use_reranker : bool
        If True, apply heuristic reranker after fusion (default: False).
    """
    results = hybrid_retrieve_documents_with_scores(
        query=query,
        task_type=task_type,
        top_k=top_k,
        use_reranker=use_reranker,
    )
    return [doc for doc, _ in results]
