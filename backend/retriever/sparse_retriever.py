from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class SparseRetrieverBase(ABC):
    @abstractmethod
    def vectorize(self, input: list[str | Image.Image]) -> np.ndarray:
        pass


class BM25(SparseRetrieverBase):
    pass


# Mixbread BMX: https://www.mixedbread.ai/blog/intro-bmx
