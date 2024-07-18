from abc import ABC, abstractmethod
import numpy as np

class RetrieverBase(ABC):
    @abstractmethod
    def create_query(self, query: str, k=1) -> np.ndarray:
        pass

class BertRetriever(RetrieverBase):
    pass

# TODO: Implement retriever factory