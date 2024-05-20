from pydantic import BaseModel
import yaml


class ModelConfig(BaseModel):
    name: str
    path: str
    multimodal: bool = False


def read_model_config() -> dict[ModelConfig]:
    with open("./FHSWF-deep-learning/models.yml", "r") as file:
        models = yaml.safe_load(file)
    return {model["name"]: ModelConfig(**model) for model in models["models"]}
