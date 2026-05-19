from typing import Literal, TypedDict


# 三类核心资料类型
SOURCE_TYPE_PAPER = "paper_note"
SOURCE_TYPE_EXPERIMENT = "experiment_doc"
SOURCE_TYPE_DATASET = "dataset_doc"
SOURCE_TYPE_CODE = "code_doc"


SourceType = Literal[
    "paper_note",
    "experiment_doc",
    "dataset_doc",
    "code_doc",
]


class DocumentMetadata(TypedDict, total=False):
    """
    RAG 文档 metadata 规范。

    total=False 表示不是每个字段都必须存在。
    比如 paper 文档可能有 title/topic/year，
    experiment 文档可能有 run_tag/dataset/task。
    """
    source_type: str
    title: str
    path: str

    topic: str
    year: int

    run_tag: str
    dataset: str
    task: str

    annotation_type: str


SOURCE_TYPE_TO_DIR = {
    SOURCE_TYPE_PAPER: "papers",
    SOURCE_TYPE_EXPERIMENT: "experiments",
    SOURCE_TYPE_DATASET: "datasets",
}


DIR_TO_SOURCE_TYPE = {
    "papers": SOURCE_TYPE_PAPER,
    "experiments": SOURCE_TYPE_EXPERIMENT,
    "datasets": SOURCE_TYPE_DATASET,
}


REQUIRED_METADATA_KEYS = [
    "source_type",
    "path",
]


COMMON_OPTIONAL_KEYS = [
    "title",
    "topic",
    "year",
    "run_tag",
    "dataset",
    "task",
    "annotation_type",
]