from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    path: str
    multimodal: bool = False
