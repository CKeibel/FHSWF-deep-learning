# factory.py

from backend.causal_models.models import CausalLMBase, HuggingFaceModel
from backend.causal_models.settings import get_settings


class CausalLMFactory:
    model_types = {
        "meta-llama/Meta-Llama-3.1-8B": HuggingFaceModel,
        "meta-llama/Meta-Llama-3.1-8B-Instruct": HuggingFaceModel,
        "HuggingFaceM4/idefics2-8b-chatty": HuggingFaceModel,
    }

    @staticmethod
    def get_model(model_id: str) -> CausalLMBase:
        settings = get_settings(model_id)
        model_class = CausalLMFactory.model_types[model_id]
        return model_class(settings)
