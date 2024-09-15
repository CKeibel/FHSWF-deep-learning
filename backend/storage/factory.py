from typing import Literal

from backend.storage.chromadb import ChromaDB
from backend.storage.lancedb import LanceDB
from backend.storage.store_base import VectorStoreBase

VectorStoreOptions = Literal["lancedb", "chromadb", "pgvector", "qdrant"]


class VectorStoreFactory:
    @staticmethod
    def create_vector_storage(option: VectorStoreOptions) -> VectorStoreBase:
        stores = {
            "lancedb": LanceDB,
            "chromadb": ChromaDB,
        }
        return stores[option]()
