import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
from dynaconf import settings
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
    def __init__(
        self,
        corpus_file=f"{settings.DATABASE_PATH}/{settings.DATABASE_NAME}_sparse.pkl",
    ):
        self.corpus_file = corpus_file
        self.corpus = []
        self.bm25 = None
        self._load_corpus()

    def _load_corpus(self):
        if os.path.exists(self.corpus_file):
            with open(self.corpus_file, "rb") as f:
                self.corpus = pickle.load(f)
            self.bm25 = BM25Okapi([doc.split() for doc in self.corpus])

    def _save_corpus(self):
        with open(self.corpus_file, "wb") as f:
            pickle.dump(self.corpus, f)

    def add_documents(self, chunks: list[str]):
        self.corpus.extend(chunks)
        self.bm25 = BM25Okapi([doc.split() for doc in self.corpus])
        self._save_corpus()

    def search(self, query, k=10) -> list[str]:
        # BM25-Suche
        tokenized_query = query.split()
        return self.bm25.get_top_n(tokenized_query, self.corpus, n=k)
