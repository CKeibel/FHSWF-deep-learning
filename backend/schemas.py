from typing import Literal

import numpy as np
from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field


class ExtractedImage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    image: Image
    caption: str
    document_name: str


class ExtractedContent(BaseModel):
    document_name: str
    full_text: str
    images: list[ExtractedImage] = Field(default_factory=list)


class StoreEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["text", "image", "caption"]
    document_name: str
    content: list[str] | list[ExtractedImage] = Field(default_factory=list)
    vector: np.ndarray


class SearchResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: str
    document_name: str
    image: Image | None = None


class GenerationConfig(BaseModel):
    max_new_tokens: int
    no_repeat_ngram_size: int
    top_k: int
    temperature: float
    num_beams: int
    do_sample: bool
