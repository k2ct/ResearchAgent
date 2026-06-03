from pathlib import Path
import shutil
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loaders import load_all_documents


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma_db"

COLLECTION_NAME = "research_agent_docs"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    创建 embedding 模型。

    Day 7 默认使用本地 sentence-transformers 模型，
    避免依赖 API key。
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[Document]:
    """
    将原始 Document 切分成更小的 chunk。

    chunk_size:
        每个片段的大致长度。

    chunk_overlap:
        相邻片段之间保留的重叠长度，避免上下文断裂。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


def reset_vector_index(persist_directory: Path = CHROMA_DIR) -> None:
    """
    删除旧的 Chroma 向量库目录。
    """
    if persist_directory.exists():
        shutil.rmtree(persist_directory)


def build_vector_index(
    reset: bool = True,
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    构建 Chroma 向量索引。

    流程：
    1. 加载 markdown Documents
    2. 切分成 chunks
    3. 创建 embedding 模型
    4. 写入 Chroma
    5. 持久化到 storage/chroma_db
    """
    if reset:
        reset_vector_index(persist_directory)

    documents = load_all_documents()
    print(f"Loaded {len(documents)} documents.")

    if not documents:
        raise ValueError("No documents loaded. Please check the data directory.")

    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    embedding_model = get_embedding_model()
    print(f"Using embedding model: {DEFAULT_EMBEDDING_MODEL}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )

    print(f"Chroma index saved to: {persist_directory}")

    return vector_store


def load_vector_store(
    persist_directory: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """
    加载已经构建好的 Chroma 向量库。
    """
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Chroma index not found at {persist_directory}. "
            f"Please run scripts/build_index.py first."
        )

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=str(persist_directory),
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

    return vector_store


def similarity_search(
    query: str,
    k: int = 3,
) -> List[Document]:
    """
    最小检索测试函数。
    """
    vector_store = load_vector_store()
    return vector_store.similarity_search(query, k=k)



'''
你这几行 import 其实对应 **RAG 向量索引构建流程里的 4 个核心组件**：

```python
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

它们之间的关系可以先看这张流程图：

```text
Markdown 文件
↓
Document
↓
RecursiveCharacterTextSplitter
↓
多个 chunk Document
↓
HuggingFaceEmbeddings
↓
文本向量
↓
Chroma
↓
本地向量库 storage/chroma_db
```

下面逐个讲。

---

# 1. `Document`：RAG 里的标准资料单元

导入：

```python
from langchain_core.documents import Document
```

`Document` 是 LangChain 里用于检索工作流的基础数据结构。官方定义里，它用于保存“一段文本及其关联 metadata”，并且主要服务于 retrieval workflow，也就是 RAG 检索流程。([LangChain引用文档][1])

它最核心只有两个部分：

```python
Document(
    page_content="正文内容",
    metadata={
        "source_type": "dataset_doc",
        "path": "data/datasets/openimages_miap.md"
    }
)
```

## 在你项目里的作用

你 Day 5 写的是 markdown 文件：

```text
data/datasets/openimages_miap.md
```

Day 6 的 loader 会把它变成：

```python
Document(
    page_content="整篇 markdown 正文……",
    metadata={
        "source_type": "dataset_doc",
        "dataset": "OpenImages-MIAP",
        "annotation_type": "bbox_level_sensitive_attribute",
        "task": "bias_evaluation",
        "path": "data/datasets/openimages_miap.md"
    }
)
```

也就是说：

| 字段             | 作用                     |
| -------------- | ---------------------- |
| `page_content` | 用于切分、embedding、检索、回答生成 |
| `metadata`     | 用于来源追踪、任务过滤、显示 Sources |

## 可以怎么理解？

`Document` 就像一张“资料卡片”：

```text
正面：资料正文 page_content
背面：来源标签 metadata
```

例如用户问：

```text
OpenImages-MIAP 的性别标注是图像级还是 bbox 级？
```

检索到相关 `Document` 后，Agent 不仅能读正文，还能知道来源是：

```python
doc.metadata["path"]
# data/datasets/openimages_miap.md
```

所以后面才能输出：

```text
Sources:
- data/datasets/openimages_miap.md
```

---

# 2. `RecursiveCharacterTextSplitter`：把长文档切成小块

导入：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

它的作用是：

> 把一个长 `Document` 切成多个更短的 chunk `Document`。

LangChain 文档中推荐递归字符切分器作为通用文本切分方式，它会按一组分隔符递归切分文本，尽量保留段落、句子等自然结构。([LangChain][2])

你代码里是这样用的：

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)

chunks = splitter.split_documents(documents)
```

## 为什么要切 chunk？

如果一整篇 markdown 直接进入向量库，会有几个问题：

```text
1. 文档太长，embedding 表达会变得笼统
2. 一个文档里可能有多个主题，检索不够精准
3. 检索回来一整篇文档，后续 prompt 会很长
4. 用户只问一个细节，但返回了太多无关内容
```

所以 RAG 通常会先切块。

例如原始文档：

```python
Document(
    page_content="OpenImages-MIAP 基本信息……数据集内容……适合任务……局限……",
    metadata={"source_type": "dataset_doc", "path": "data/datasets/openimages_miap.md"}
)
```

切分后变成多个 chunk：

```python
Document(
    page_content="OpenImages-MIAP 是基于 OpenImages 的人物属性相关标注数据……",
    metadata={"source_type": "dataset_doc", "path": "data/datasets/openimages_miap.md"}
)
```

```python
Document(
    page_content="该数据集的关键特点是属性更接近 person-level 或 bbox-level……",
    metadata={"source_type": "dataset_doc", "path": "data/datasets/openimages_miap.md"}
)
```

注意：**切分后 metadata 会继承下来**。这点特别重要。

也就是说，即使检索到的是一个小 chunk，它仍然知道自己来自：

```text
data/datasets/openimages_miap.md
```

---

## `chunk_size=500` 是什么？

```python
chunk_size=500
```

表示每个 chunk 尽量控制在约 500 个字符。

对你现在的 sample docs 来说，500 比较合适。

太小会导致：

```text
上下文太少
一个概念被切碎
检索结果看起来零散
```

太大会导致：

```text
检索不精准
无关内容混进来
后续 LLM prompt 太长
```

---

## `chunk_overlap=80` 是什么？

```python
chunk_overlap=80
```

表示相邻 chunk 之间保留约 80 个字符的重叠。

为什么要重叠？因为如果刚好在关键信息中间切开，就会丢上下文。

比如没有 overlap 时可能变成：

```text
chunk 1：OpenImages-MIAP 的属性标注更接近
chunk 2：bbox/person-level，而不是 image-level
```

如果用户问“是不是 bbox 级”，chunk 1 单独看不完整，chunk 2 又缺主语。

有 overlap 后，两个 chunk 都会保留一部分上下文，检索更稳。

---

## `separators` 是什么？

你代码里写的是：

```python
separators=["\n\n", "\n", "。", "，", " ", ""]
```

意思是递归切分时优先按这些边界切：

```text
先按段落空行切：\n\n
不行再按换行切：\n
再按中文句号切：。
再按中文逗号切：，
再按空格切
最后实在不行按字符硬切
```

这对中文 markdown 很有用。

---

# 3. `HuggingFaceEmbeddings`：把文本变成向量

导入：

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

`HuggingFaceEmbeddings` 是 LangChain 的 Hugging Face embedding 集成类，使用 `sentence_transformers` 模型生成文本向量；官方也说明，新的项目推荐使用 `langchain_huggingface` 里的 `HuggingFaceEmbeddings`，而不是旧的 community 版本。([LangChain引用文档][3])

你代码里是这样：

```python
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

它的作用是：

> 把文本 chunk 转成数字向量。

比如文本：

```text
OpenImages-MIAP 的属性标注更接近 bbox/person-level
```

会变成类似：

```python
[0.023, -0.118, 0.451, ..., 0.072]
```

这个向量代表这段文本的语义位置。

用户问题也会被转成向量：

```text
OpenImages-MIAP 的性别标注是图像级还是 bbox 级？
```

然后向量数据库会比较：

```text
用户问题向量
vs
所有 chunk 向量
```

找出最相似的 chunks。

---

## `model_name` 是什么？

```python
model_name="sentence-transformers/all-MiniLM-L6-v2"
```

表示使用 Hugging Face 上的一个 sentence-transformers embedding 模型。

优点：

```text
本地运行
不需要 API Key
适合 demo
速度快
```

缺点：

```text
中文效果不是最强
对你的中英文混合资料能跑通，但后续可以换更适合中文的 embedding 模型
```

---

## `model_kwargs={"device": "cpu"}` 是什么？

```python
model_kwargs={"device": "cpu"}
```

意思是让模型在 CPU 上运行。

适合你现在本地 demo：

```text
不用 GPU
部署简单
更稳定
```

如果你以后在有 GPU 的服务器上跑，可以考虑改成：

```python
model_kwargs={"device": "cuda"}
```

但现在没必要。

---

## `encode_kwargs={"normalize_embeddings": True}` 是什么？

```python
encode_kwargs={"normalize_embeddings": True}
```

意思是把 embedding 向量归一化。

简单理解：

> 让向量长度统一，后面做相似度比较更稳定。

很多检索场景会使用归一化 embedding，因为这样向量之间的 cosine similarity / dot product 比较会更一致。

---

# 4. `Chroma`：本地向量数据库

导入：

```python
from langchain_chroma import Chroma
```

Chroma 是一个开源向量数据库，LangChain 当前推荐通过独立的 `langchain-chroma` 包集成使用；`Chroma` 可以把文本、embedding 和 metadata 存起来，并支持后续相似度检索。([LangChain Docs][4])

你代码里有两种典型用法。

---

## 用法一：从文档构建向量库

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(persist_directory),
    collection_name=collection_name,
)
```

这句做了很多事：

```text
1. 接收 chunks
2. 对每个 chunk 调用 embedding_model
3. 得到每个 chunk 的向量
4. 把 chunk 文本、metadata、向量写入 Chroma
5. 保存到 persist_directory
```

对应到你的项目：

```text
chunks
↓
HuggingFaceEmbeddings
↓
Chroma
↓
storage/chroma_db
```

---

## 参数解释

### `documents=chunks`

```python
documents=chunks
```

表示要存入向量库的文档片段。

这里不是 Day 6 原始整篇 Document，而是切分后的 chunk Documents。

---

### `embedding=embedding_model`

```python
embedding=embedding_model
```

告诉 Chroma：

> 用这个 embedding 模型把文本转成向量。

---

### `persist_directory=str(persist_directory)`

```python
persist_directory=str(persist_directory)
```

表示向量库保存到本地哪个目录。

你项目里是：

```text
F:\ResearchAgent\storage\chroma_db
```

有了这个目录后，下次不用重新构建索引，可以直接加载。

---

### `collection_name=collection_name`

```python
collection_name="research_agent_docs"
```

表示 Chroma 里的集合名称。

你可以理解成数据库里的一个表：

```text
Chroma 数据库
└── research_agent_docs 这个 collection
    ├── chunk 1
    ├── chunk 2
    └── chunk 3
```

---

## 用法二：加载已经存在的向量库

```python
vector_store = Chroma(
    persist_directory=str(persist_directory),
    collection_name=collection_name,
    embedding_function=embedding_model,
)
```

这句表示：

> 不重新创建索引，而是从 `storage/chroma_db` 读取已有向量库。

为什么还要传 `embedding_function`？

因为用户查询时，Chroma 需要把 query 也转成向量。

例如：

```python
docs = vector_store.similarity_search(
    "OpenImages-MIAP 的性别标注是图像级还是 bbox 级？",
    k=3,
)
```

这时 Chroma 要先做：

```text
query 文本
↓
embedding_function
↓
query 向量
↓
和库里的 chunk 向量比较
↓
返回 top 3
```

---

# 5. 这四个组件在你的 `indexer.py` 里怎么配合？

你 Day 7 的核心函数大概是：

```python
def build_vector_index():
    documents = load_all_documents()
    chunks = split_documents(documents)
    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    return vector_store
```

可以拆成四步：

---

## 第一步：加载 Document

```python
documents = load_all_documents()
```

得到：

```python
[
    Document(page_content="论文笔记正文", metadata={"source_type": "paper_note", ...}),
    Document(page_content="实验说明正文", metadata={"source_type": "experiment_doc", ...}),
    Document(page_content="数据集说明正文", metadata={"source_type": "dataset_doc", ...}),
]
```

这里用的是 `Document`。

---

## 第二步：切成 chunks

```python
chunks = split_documents(documents)
```

内部用：

```python
RecursiveCharacterTextSplitter
```

得到：

```python
[
    Document(page_content="论文笔记 chunk 1", metadata={...}),
    Document(page_content="论文笔记 chunk 2", metadata={...}),
    Document(page_content="实验说明 chunk 1", metadata={...}),
]
```

---

## 第三步：创建 embedding 模型

```python
embedding_model = get_embedding_model()
```

内部用：

```python
HuggingFaceEmbeddings
```

它负责把每个 chunk 变成向量。

---

## 第四步：写入 Chroma

```python
Chroma.from_documents(...)
```

Chroma 会保存：

```text
chunk page_content
chunk metadata
chunk embedding
```

最终在本地形成：

```text
storage/chroma_db
```

---

# 6. 用一个完整例子串起来

假设你有文档：

```text
data/datasets/openimages_miap.md
```

内容中有一句：

```text
OpenImages-MIAP 的关键特点是属性更接近 person-level 或 bbox-level，而不是简单的 image-level 标签。
```

Day 7 发生的事情是：

## 1. Loader 生成 Document

```python
Document(
    page_content="OpenImages-MIAP 的关键特点是属性更接近 person-level 或 bbox-level……",
    metadata={
        "source_type": "dataset_doc",
        "dataset": "OpenImages-MIAP",
        "path": "data/datasets/openimages_miap.md"
    }
)
```

## 2. Splitter 切 chunk

```python
Document(
    page_content="OpenImages-MIAP 的关键特点是属性更接近 person-level 或 bbox-level，而不是简单的 image-level 标签。",
    metadata={
        "source_type": "dataset_doc",
        "dataset": "OpenImages-MIAP",
        "path": "data/datasets/openimages_miap.md",
        "chunk_id": 12
    }
)
```

## 3. Embedding 转向量

```python
[0.031, -0.214, 0.087, ...]
```

## 4. Chroma 保存

```text
collection: research_agent_docs

存储内容：
- text: OpenImages-MIAP 的关键特点……
- vector: [0.031, -0.214, 0.087, ...]
- metadata: source_type=dataset_doc, path=data/datasets/openimages_miap.md
```

## 5. 用户查询时

用户问：

```text
OpenImages-MIAP 的性别标注是图像级还是 bbox 级？
```

Chroma 做：

```text
query → embedding → 和所有 chunk 向量比较 → 返回最相似的 chunks
```

返回的结果里就会包含：

```python
doc.page_content
doc.metadata
```

所以你能打印：

```python
print(doc.page_content)
print(doc.metadata["path"])
```

---

# 7. 四者一句话总结

| 组件                               | 一句话作用         | 在你的项目里对应                  |
| -------------------------------- | ------------- | ------------------------- |
| `Document`                       | 包装文本和来源信息     | 把 markdown 变成 RAG 标准资料    |
| `RecursiveCharacterTextSplitter` | 把长文档切成小 chunk | 提高检索精度                    |
| `HuggingFaceEmbeddings`          | 把文本变成语义向量     | 让 Chroma 能做相似度搜索          |
| `Chroma`                         | 保存向量并检索相似文本   | 本地知识库 `storage/chroma_db` |

最核心流程：

```text
Document 负责“资料是什么”
TextSplitter 负责“切成多大”
Embedding 负责“变成向量”
Chroma 负责“存起来并查回来”
```

---

# 8. 你现在最该记住的代码关系

```python
# 1. 原始资料
documents: list[Document] = load_all_documents()

# 2. 切分资料
splitter = RecursiveCharacterTextSplitter(...)
chunks: list[Document] = splitter.split_documents(documents)

# 3. 创建 embedding 模型
embedding_model = HuggingFaceEmbeddings(...)

# 4. 建向量库
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="storage/chroma_db",
    collection_name="research_agent_docs",
)

# 5. 检索
docs = vector_store.similarity_search("用户问题", k=3)
```

这就是 Day 7 的主线。

[1]: https://reference.langchain.com/python/langchain-core/documents/base/Document?utm_source=chatgpt.com "Document | langchain_core"
[2]: https://www.langchain.com/?utm_source=chatgpt.com "LangChain: Observe, Evaluate, and Deploy Reliable AI Agents"
[3]: https://reference.langchain.com/python/langchain-huggingface/embeddings/huggingface/HuggingFaceEmbeddings?utm_source=chatgpt.com "HuggingFaceEmbeddings | langchain_huggingface"
[4]: https://docs.langchain.com/oss/python/integrations/vectorstores/chroma?utm_source=chatgpt.com "Chroma integration - Docs by LangChain"

'''