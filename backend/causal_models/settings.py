from abc import ABC

from pydantic import BaseModel


class CausalModelSettings(ABC, BaseModel):
    pass


class HuggingFaceModelSettings(ABC, CausalModelSettings):
    model_id: str
    chat_template: str


class Llama3Settings(HuggingFaceModelSettings):
    model_id: str = "meta-llama/Meta-Llama-3.1-8B"
    chat_template: str = "language_prompt.j2"


class Idefics2Chat(HuggingFaceModelSettings):
    model_id: str = "HuggingFaceM4/idefics2-8b-chatty"
    chat_template: str = "multimodal_prompt.j2"


class Settings(BaseModel):
    llama3 = Llama3Settings
    idefics2_chat = Idefics2Chat


def get_settings() -> Settings:
    return Settings()
