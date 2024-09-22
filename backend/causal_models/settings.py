from abc import ABC

from pydantic import BaseModel, ConfigDict
from transformers import (AutoModelForCausalLM, AutoModelForVision2Seq,
                          AutoProcessor, AutoTokenizer, PreTrainedModel)


class CausalModelSettings(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class HuggingFaceModelSettings(CausalModelSettings):
    model_id: str
    chat_template: str
    architecture: PreTrainedModel
    tokenizer: AutoTokenizer | AutoProcessor
    multimodal: bool


class Llama3SmallSettings(HuggingFaceModelSettings):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_id: str = "meta-llama/Meta-Llama-3.1-8B"
    chat_template: str = "language_prompt.j2"
    architecture: PreTrainedModel = AutoModelForCausalLM
    tokenizer: AutoTokenizer | AutoProcessor = AutoTokenizer
    multimodal: bool = False


class Llama3SmallInstruct(HuggingFaceModelSettings):
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    chat_template: str = "language_prompt.j2"
    architecture: PreTrainedModel = AutoModelForCausalLM
    tokenizer: AutoTokenizer | AutoProcessor = AutoTokenizer
    multimodal: bool = False


class ZephyrBeta(HuggingFaceModelSettings):
    model_id: str = "HuggingFaceH4/zephyr-7b-beta"
    chat_template: str = "language_prompt.j2"
    architecture: PreTrainedModel = AutoModelForCausalLM
    tokenizer: AutoTokenizer | AutoProcessor = AutoTokenizer
    multimodal: bool = False


class Idefics2Chat(HuggingFaceModelSettings):
    model_id: str = "HuggingFaceM4/idefics2-8b-chatty"
    chat_template: str = "multimodal_prompt.j2"
    architecture: PreTrainedModel = AutoModelForVision2Seq
    tokenizer: AutoTokenizer | AutoProcessor = AutoProcessor
    multimodal: str = True


class Settings:
    models: dict[str, CausalModelSettings] = {
        "meta-llama/Meta-Llama-3.1-8B": Llama3SmallSettings(),
        "meta-llama/Meta-Llama-3.1-8B-Instruct": Llama3SmallInstruct(),
        "HuggingFaceH4/zephyr-7b-beta": ZephyrBeta(),
        "HuggingFaceM4/idefics2-8b-chatty": Idefics2Chat(),
    }


def get_settings(model_id: str) -> HuggingFaceModelSettings:
    settings = Settings()
    for model, setting in settings.models.items():
        if model == model_id:
            return setting
    raise ValueError(f"Settings for model_id '{model_id}' not found.")
