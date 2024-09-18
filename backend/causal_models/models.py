from abc import ABC, abstractmethod

import torch
from jinja2 import Template
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.causal_models.settings import (CausalModelSettings,
                                            HuggingFaceModelSettings)
from backend.prompts.prompt_manager import PromptManager
from backend.schemas import SearchResult


class CausalLMBase(ABC):
    @abstractmethod
    def __init__(self, settingd: CausalModelSettings) -> None:
        pass

    @abstractmethod
    def generate(
        self, question: str, search_results: list[SearchResult], **kwargs
    ) -> str:
        pass


class LanguageModel(CausalLMBase):
    def __init__(self, settings: HuggingFaceModelSettings) -> None:
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Loading '{self.settings.model_id}' to {self.device}... This can take a while..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_id, device_map=self.device, torch_dtype=torch.float16
        )
        self.template: Template = PromptManager.get_prompt_template(
            self.settings.chat_template
        )

    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def _construct_prompt(
        self, question: str, search_results: list[SearchResult]
    ) -> str:
        context = {"role": "context", "content": list()}
        for result in search_results:
            context["content"].append(result.text)

        user_message = {"role": "user", "content": [question]}
        return self.template.render(messages=[context, user_message])

    @torch.no_grad()
    def generate(
        self, question: str, search_results: list[SearchResult], **kwargs
    ) -> str:
        prompt = self._construct_prompt(question, search_results)
        inputs_ids = self._tokenize(prompt)
        outputs = self.model.generate(
            inputs_ids, **kwargs, max_new_tokens=250
        )  # TODO: generation config
        # decode only new tokens to string
        answer = self.tokenizer.decode(
            outputs[0][len(inputs_ids[0]) :], skip_special_tokens=True
        )
        torch.cuda.empty_cache()
        return answer


class MultimodalModel(CausalLMBase):
    pass
