from backend.retriever.dense_retriever import (ClipRetriever,
                                               DenseRetrieverBase,
                                               JinaClipRetriever)
from backend.retriever.sparse_retriever import SparseRetrieverBase

# Mapping of model names to retriever types
mapping: dict[str, str] = {"jinaai/jina-clip-v1": "jina-clip"}


class DenseRetrieverFactory:
    model_types = {"clip": ClipRetriever, "jina-clip": JinaClipRetriever}

    @staticmethod
    def get_model(model_name: str) -> DenseRetrieverBase:
        model_type = mapping.get(model_name)
        if model_type:
            return DenseRetrieverFactory.model_types[model_type](model_name)
        else:
            raise ValueError(f"Model {model_name} not found")


class SparseRetrieverFactory:
    pass
