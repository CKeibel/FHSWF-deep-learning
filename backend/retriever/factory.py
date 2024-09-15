from typing import Literal

from backend.retriever.base_retriever import RetrieverBase
from backend.retriever.dense_retriever import ClipRetriever

DenseRetrieverOptions = Literal["clip"]
SparseRetrieverOptions = Literal["bm25", "bmx"]


class DenseRetrieverFactory:
    model_types = {
        "clip": ClipRetriever,
    }

    @staticmethod
    def get_model(option: DenseRetrieverOptions) -> RetrieverBase:
        return DenseRetrieverFactory.model_types[option]()


class SparseRetrieverFactory:
    pass
