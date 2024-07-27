from pydantic import BaseModel


class GenerationConfig(BaseModel):
    max_new_tokens: int
    top_p: float
    top_k: float
    # ... TODO
