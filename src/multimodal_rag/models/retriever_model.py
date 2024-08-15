from abc import ABC, abstractmethod
import numpy as np


class RetrieverBase(ABC):
    @abstractmethod
    def vectorize(self, query: str, k=1) -> np.ndarray:
        pass


class DenseRetriever(RetrieverBase):
    pass


class SparseRetriever(RetrieverBase):
    pass


# Mixbread BMX: https://www.mixedbread.ai/blog/intro-bmx
