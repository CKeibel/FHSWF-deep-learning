from abc import ABC, abstractmethod

import numpy as np

from backend.schemas import StoreEntry


class VectorStoreBase(ABC):
    @abstractmethod
    def query(self, query_vector: np.ndarray, k=1):  # TODO: define return type
        pass

    @abstractmethod
    def insert(self, entry: StoreEntry) -> None:  # TODO: Document type
        pass

    @abstractmethod
    def delete(self, document_ids) -> None:  # TODO: param
        pass
