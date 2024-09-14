from abc import ABC, abstractmethod
import numpy as np
from PIL import Image


class RetrieverBase(ABC):
    @abstractmethod
    def vectorize(self, input: list[str | Image.Image], k=1) -> np.ndarray:
        pass


class BertRetriever(RetrieverBase):
    pass


class ClipRetriever(RetrieverBase):
    pass


class JinaClipRetriever(RetrieverBase):
    pass


class SparseRetriever(RetrieverBase):
    pass


# Mixbread BMX: https://www.mixedbread.ai/blog/intro-bmx
