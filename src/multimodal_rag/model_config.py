from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    path: str
    multimodal: bool = False


decoder_models = [
    ModelConfig(name="Zephyr 7b beta", path="HuggingFaceH4/zephyr-7b-beta")
]