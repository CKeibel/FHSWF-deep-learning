from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

from backend.schemas import ExtractedImage


class DenseRetrieverBase(ABC):
    @abstractmethod
    def __init__(self, model_id: str) -> None:
        pass

    @abstractmethod
    def is_multimodal(self) -> bool:
        pass

    @abstractmethod
    @torch.no_grad()
    def vectorize(self, inputs: list[str | ExtractedImage]) -> np.ndarray | None:
        pass


class BertRetriever(DenseRetrieverBase):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading '{model_id}' to {self.device}...")
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model.eval()

    def is_multimodal(self) -> bool:
        return False

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @torch.no_grad()
    def vectorize(self, inputs: list[str | ExtractedImage]) -> np.ndarray | None:
        if isinstance(inputs[0], ExtractedImage):
            logger.debug("Passed image input to text only model.")
            return None
        else:
            logger.info("Vectorizing texts...")
            inputs = self.tokenizer(
                inputs, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings = self.mean_pooling(outputs, inputs["attention_mask"])
            return F.normalize(embeddings, p=2, dim=1).cpu().numpy()


class ClipRetriever(DenseRetrieverBase):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading '{model_id}' to {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def is_multimodal(self) -> bool:
        return True

    @torch.no_grad()
    def vectorize(self, inputs: list[str | ExtractedImage]) -> np.ndarray | None:
        if isinstance(inputs[0], ExtractedImage):
            logger.info("Vectorizing images...")
            inputs = self.processor(
                images=inputs, return_tensors="pt", padding=True
            ).to(self.device)
            vectors = self.model.get_image_features(**inputs)
            return vectors.cpu().detach().numpy()
        else:
            logger.info("Vectorizing texts...")
            inputs = self.processor(text=inputs, return_tensors="pt")
            vectors = self.model.get_text_features(**inputs)
            return vectors.cpu().detach().numpy()


class JinaClipRetriever(DenseRetrieverBase):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading '{model_id}' to {self.device}...")
        self.model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def is_multimodal(self) -> bool:
        return True

    @torch.no_grad()
    def vectorize(self, inputs: list[str | ExtractedImage]) -> np.ndarray | None:
        if isinstance(inputs[0], ExtractedImage):
            logger.info("Vectorizing images...")
            vectors = self.model.encode_image([img.image for img in inputs])
            return vectors
        else:
            logger.info("Vectorizing texts...")
            vectors = self.model.encode_text(inputs)
            return vectors
