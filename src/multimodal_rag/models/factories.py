from multimodal_rag.models.causal_model import (
    CausalLMBase,
    LanguageModel,
    MultimodalModel,
)
from multimodal_rag.models.config import CausalLMConfig, RetrieverConfig
from multimodal_rag.models.retriever_model import RetrieverBase, BertRetriever


class RerieverFactory:
    model_types = {"Bert": BertRetriever}  # TODO: Add others,

    @staticmethod
    def get_model(config: RetrieverConfig) -> RetrieverBase:
        return RerieverFactory.model_types[config.type](config)


class CausalLMFactory:
    model_types = {
        "language": LanguageModel,
        "multimodal": MultimodalModel,
    }  # TODO: Add OpenAI and others

    @staticmethod
    def get_model(config: CausalLMConfig) -> CausalLMBase:
        return CausalLMFactory.model_types[config.type](config)
