from multimodal_rag.models.causal_model import (
    CausalLMBase,
    LanguageModel,
    MultimodalModel,
)
from multimodal_rag.models.config import CausalLMConfig
import huggingface_hub
from dotenv import load_dotenv
import os
from loguru import logger


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ModelService(metaclass=Singleton):
    def __init__(self, config: CausalLMConfig) -> None:
        ModelService.hf_login()
        self.config = config
        self.model = ModelService.load_model(config)

    @staticmethod
    def load_model(config: CausalLMConfig) -> CausalLMBase:
        model_types = {"language": LanguageModel, "multimodal": MultimodalModel}
        return model_types[config.type](config)

    def change_model(self, config: CausalLMConfig) -> None:
        self.model = ModelService.load_model(config)

    @staticmethod
    def hf_login(secret: str | None = None) -> None:
        if secret is None:
            load_dotenv()
            secret = os.getenv("HUGGINGFACE_TOKEN")

        if secret:
            huggingface_hub.login(secret)
            logger.info("Logged in to Hugging Face.")
        else:
            logger.warning(
                "No Hugging Face token found. Please set the HUGGINGFACE_TOKEN environment variable."
            )
