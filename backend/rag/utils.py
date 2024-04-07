from enum import Enum
from pathlib import Path

import pydantic
import yaml


class ConfigType(Enum):
    GENERATION = "generation"
    MODEL = "model"
    TRAINING = "training"


class TrainingConfig(pydantic.BaseModel):
    epochs: int
    learning_rate: float


class GenerationConfig(pydantic.BaseModel):
    pass


class CausalLMConfig(pydantic.BaseModel):
    name: str
    load_4bit: bool
    load_8bit: bool
    path: Path | None = None


class EncoderConfig(pydantic.BaseModel):
    name: str
    path: Path | None = None


class ModelConfig(pydantic.BaseModel):
    causal_lm: CausalLMConfig | None = None
    encoder: EncoderConfig | None = None

    @pydantic.root_validator(pre=True)
    @classmethod
    def check_model(cls, values: dict[str, any]) -> dict[str, any]:
        assert (
            "causal_lm" in values or "encoder" in values
        ), "At least one of 'causal_lm' or 'encoder' must be provided."
        return values


def load_yaml(path: Path, type: ConfigType) -> GenerationConfig | ModelConfig | TrainingConfig:
    """
    Loads a YAML file and returns the configuration object.

    Args:
        path (Path): The path to the YAML file.
        type (ConfigType): The type of configuration to load.
    Returns:
        Union[GenerationConfig, ModelConfig]: The loaded configuration object.
    """
    try:
        with open(path) as file:
            config: dict = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file at path: {path}")

    if type == ConfigType.GENERATION:
        return GenerationConfig(**config)
    elif type == ConfigType.MODEL:
        return ModelConfig(**config)
    else:
        return TrainingConfig(**config)


