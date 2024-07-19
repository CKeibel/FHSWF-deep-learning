from abc import ABC, abstractmethod
import numpy as np


class VectorStoreBase(ABC):
    @abstractmethod
    def search(self, query_vector: np.ndarray, k=1):  # TODO: define return type
        pass


class LanceDB(VectorStoreBase):
    pass
