from multimodal_rag.llm import CausalLMBase, LanguageModel, MultimodalModel
from multimodal_rag.llm_config import CausalLMConfig

class Singleton (type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ModelManager(metaclass=Singleton):
    def __init__(self, config: CausalLMConfig) -> None:
        self.config = config
        self.model = ModelManager.load_model(config)

    @staticmethod
    def load_model(config: CausalLMConfig) -> CausalLMBase:
        model_types = {
            "language": LanguageModel,
            "multimodal": MultimodalModel
        }
        return model_types[config.type](config)
    
    def change_model(self, config: CausalLMConfig) -> None:
        self.model = ModelManager.load_model(config)
