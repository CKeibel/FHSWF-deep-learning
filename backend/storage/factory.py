from backend.storage.vector_store import (
    VectorStore,
    LanceDB,
    ChromaDB,
    PGVector,
    Qdrant,
)
from typing import Literal

VectorStores = Literal["lancedb", "chromadb", "pgvector", "qdrant"]


class VectorStorageFactory:
    @staticmethod
    def create_vector_storage(name: VectorStores) -> VectorStore:
        stores = {
            "lancedb": LanceDB,
            "chromadb": ChromaDB,
            "pgvector": PGVector,
            "qdrant": Qdrant,
        }
        return stores[name]()
