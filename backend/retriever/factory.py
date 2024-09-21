from backend.retriever.dense_retriever import (BertRetriever, ClipRetriever,
                                               DenseRetrieverBase,
                                               JinaClipRetriever)
from backend.retriever.sparse_retriever import BM25, SparseRetrieverBase


class DenseRetrieverFactory:
    model_types = {
        "sentence-transformers/all-MiniLM-L6-v2": BertRetriever,
        "openai/clip-vit-base-patch32": ClipRetriever,
        "jinaai/jina-clip-v1": JinaClipRetriever,
    }

    @staticmethod
    def get_model(model_id: str) -> DenseRetrieverBase:
        retriever_class = DenseRetrieverFactory.model_types.get(model_id)
        if retriever_class:
            return retriever_class(model_id)
        else:
            raise ValueError(f"Dense retriever {model_id} not found")


class SparseRetrieverFactory:
    model_types = {"bm25": BM25}

    @staticmethod
    def get_model(model_id: str) -> SparseRetrieverBase:
        sparse_retriever = SparseRetrieverFactory.model_types.get(model_id)()
        if sparse_retriever:
            return sparse_retriever
        else:
            raise ValueError(f"Sparse retriever {model_id} not found")
