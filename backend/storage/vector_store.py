from abc import ABC, abstractmethod
import numpy as np


class VectorStoreBase(ABC):
    @abstractmethod
    def query(self, query_vector: np.ndarray, k=1):  # TODO: define return type
        pass

    @abstractmethod
    def insert(self, documents) -> None:  # TODO: Document type
        pass

    @abstractmethod
    def delete(self, document_ids) -> None:  # TODO: param
        pass


class LanceDB(VectorStoreBase):
    pass


class ChromaDB(VectorStoreBase):
    pass


class PGVector(VectorStoreBase):
    pass


class Qdrant(VectorStoreBase):
    pass
