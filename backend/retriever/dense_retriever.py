from abc import ABC, abstractmethod

import numpy as np
import torch
from loguru import logger
from PIL import Image
from transformers import AutoModel


class DenseRetrieverBase(ABC):
    @abstractmethod
    def __init__(self, model_id: str) -> None:
        pass

    @abstractmethod
    def vectorize(self, inputs: list[str | Image.Image]) -> np.ndarray:
        pass


class BertRetriever(DenseRetrieverBase):
    pass


class ClipRetriever(DenseRetrieverBase):
    pass


class JinaClipRetriever(DenseRetrieverBase):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading '{model_id}' to {self.device}...")
        self.model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def vectorize(self, inputs: list[str | Image.Image]) -> np.ndarray:
        if isinstance(inputs[0], Image.Image):
            logger.info("Vectorizing images...")
            vectors = self.model.encode_image(inputs)
            return vectors
        else:
            logger.info("Vectorizing texts...")
            vectors = self.model.encode_text(inputs)
            return vectors
