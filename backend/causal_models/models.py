from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch
from jinja2 import Template
from loguru import logger
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoModelForVision2Seq,
                          AutoProcessor, AutoTokenizer)

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
        return self.template.render(
            messages=[context, user_message], add_generation_prompt=True
        )

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
    def __init__(self, settings: HuggingFaceModelSettings) -> None:
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Loading '{self.settings.model_id}' to {self.device}... This can take a while..."
        )
        self.processor = AutoProcessor.from_pretrained(self.settings.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.settings.model_id, device_map=self.device, torch_dtype=torch.float16
        )
        self.template: Template = PromptManager.get_prompt_template(
            self.settings.chat_template
        )

    def _tokenize(self, text: str, images: list[Image.Image]) -> torch.Tensor:
        if len(images) > 0:
            return self.processor(
                text, images=images, return_tensors="pt"
            ).input_ids.to(self.device)
        else:
            return self.processor(text, return_tensors="pt").input_ids.to(self.device)

    def _construct_prompt(
        self, question: str, search_results: list[SearchResult]
    ) -> str:
        context = {"role": "context", "content": list()}
        for result in search_results:
            if result.image is not None:
                context["content"].append(
                    {
                        "type": "image",
                        "content": result.image,
                    }
                )
            context["content"].append(
                {
                    "type": "text",
                    "text": result.text,
                }
            )

            user_message = {
                "role": "user",
                "content": [{"type": "text", "content": [question]}],
            }
        return self.template.render(
            messages=[context, user_message], add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self, question: str, search_results: list[SearchResult], **kwargs
    ) -> str:
        prompt = self._construct_prompt(question, search_results)
        images = [result.image for result in search_results if result.image is not None]
        inputs_ids = self._tokenize(prompt, images)
        outputs = self.model.generate(inputs_ids, **kwargs, max_new_tokens=250)
        # decode only new tokens to string
        answer = self.processor.decode(
            outputs[0][len(inputs_ids[0]) :], skip_special_tokens=True
        )
        torch.cuda.empty_cache()
        return answer
