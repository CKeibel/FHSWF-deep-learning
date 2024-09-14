from pydantic import BaseModel


class ExtractedFileContent(BaseModel):
    text: str
    images: list


class GenerationConfig(BaseModel):
    max_new_tokens: int
    top_p: float
    top_k: float
    # ... TODO
