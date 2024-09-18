from abc import ABC

from pydantic import BaseModel


class CausalModelSettings(ABC, BaseModel):
    pass


class HuggingFaceModelSettings(CausalModelSettings):
    model_id: str
    chat_template: str


class Llama3SmallSettings(HuggingFaceModelSettings):
    model_id: str = "meta-llama/Meta-Llama-3.1-8B"
    chat_template: str = "language_prompt.j2"


class Llama3SmallInstruct(HuggingFaceModelSettings):
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    chat_template: str = "language_prompt.j2"


class Idefics2Chat(HuggingFaceModelSettings):
    model_id: str = "HuggingFaceM4/idefics2-8b-chatty"
    chat_template: str = "multimodal_prompt.j2"


class Settings:
    llama3_8b: Llama3SmallSettings = Llama3SmallSettings()
    llama3_8b_instruct: Llama3SmallInstruct = Llama3SmallInstruct()
    idefics2_chat: Idefics2Chat = Idefics2Chat()


def get_settings() -> Settings:
    return Settings()
