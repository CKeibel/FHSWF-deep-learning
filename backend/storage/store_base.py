from abc import ABC, abstractmethod

import numpy as np

from backend.schemas import SearchResult, StoreEntry


class VectorStoreBase(ABC):
    @abstractmethod
    def query(self, query_vector: np.ndarray, k=10) -> list[SearchResult]:
        pass

    @abstractmethod
    def insert(self, entry: StoreEntry) -> None:
        pass

    @abstractmethod
    def delete(self, document_ids) -> None:  # TODO: param
        pass
