from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
from rank_bm25 import BM25Okapi


class SparseRetrieverBase(ABC):
    @abstractmethod
    def add_documents(self, input: list[str | Image.Image]) -> np.ndarray:
        pass

    @abstractmethod
    def search(self, query, k=10):
        pass


class BM25(SparseRetrieverBase):
    def __init__(self):
        self.corpus = []
        self.bm25 = None

    def add_documents(self, chunks: str):
        self.corpus.extend(chunks)
        self.bm25 = BM25Okapi([doc.split() for doc in self.corpus])

    def search(self, query, k=10) -> list[str]:
        # BM25-Suche
        tokenized_query = query.split()
        return self.bm25.get_top_n(tokenized_query, self.corpus, n=k)
