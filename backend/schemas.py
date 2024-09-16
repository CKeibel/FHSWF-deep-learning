from typing import Literal

import numpy as np
from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field


class ExtractedImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: Image
    caption: str


class ExtractedDocument(BaseModel):
    document_name: str
    full_text: str
    images: list[ExtractedImage] = Field(default_factory=list)


class StoreEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["text", "image"]
    document_name: str
    content: list[str] | list[Image] = Field(default_factory=list)
    vector: np.ndarray


class GenerationConfig(BaseModel):
    max_new_tokens: int
    top_p: float
    top_k: float
    # ... TODO
