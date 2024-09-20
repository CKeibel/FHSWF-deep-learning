from backend.causal_models.models import CausalLMBase, HuggingFaceModel
from backend.causal_models.settings import get_settings

language_models = {
    "llama3_8b": HuggingFaceModel,
    "llama3_8b_instruct": HuggingFaceModel,
}

multimodal_models = {
    "idefics2_chat": HuggingFaceModel,
}


class CausalLMFactory:
    model_types = language_models | multimodal_models

    @staticmethod
    def get_model(choice: str) -> CausalLMBase:
        settings = getattr(get_settings(), choice)
        return CausalLMFactory.model_types[choice](settings)
