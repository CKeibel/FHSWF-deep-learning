from pydantic import BaseModel


# Huggingface
class HuggingfaceModel(BaseModel):
    name: str
    adapter_path: str | None = None


class HFLanguageModel(HuggingfaceModel):
    pass


class HFMultimodalModel(HuggingfaceModel):
    pass


class Settings(BaseModel):
    hf_language_model = HuggingfaceModel()
    hf_multimodal_model = HFMultimodalModel()


def get_settings() -> Settings:
    return Settings()
