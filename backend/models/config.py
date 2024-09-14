from pydantic import BaseModel
import yaml


class RetrieverConfig(BaseModel):
    name: str
    path: str
    type: str = "bert"


class CausalLMConfig(BaseModel):
    name: str
    path: str
    type: str = "language"


def read_model_config() -> dict[CausalLMConfig]:
    with open("./llms.yml", "r") as file:
        models = yaml.safe_load(file)
    return {model["name"]: CausalLMConfig(**model) for model in models["causal_models"]}
