from typing import Literal

from backend.causal_models.models import (CausalLMBase, LanguageModel,
                                          MultimodalModel)
from backend.causal_models.settings import get_settings

CausalModelOptions = Literal["hf_language_model", "hf_multimodal_model"]


class CausalLMFactory:
    model_types = {
        "hf_language_model": LanguageModel,
        "hf_multimodal_model": MultimodalModel,
    }  # TODO: Add OpenAI and others

    @staticmethod
    def get_model(option: CausalModelOptions) -> CausalLMBase:
        settings = getattr(get_settings, option)
        return CausalLMFactory.model_types[option](settings)
