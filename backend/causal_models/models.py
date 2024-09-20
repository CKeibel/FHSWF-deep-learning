from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch
from jinja2 import Template
from loguru import logger
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoModelForVision2Seq,
                          AutoProcessor, AutoTokenizer)
from transformers.generation.utils import GenerateOutput

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


class HuggingFaceModel(CausalLMBase):
    def __init__(self, settings: HuggingFaceModelSettings) -> None:
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Loading '{self.settings.model_id}' to {self.device}... This can take a while..."
        )
        self.processor = self.settings.tokenizer.from_pretrained(self.settings.model_id)
        self.model = self.settings.architecture.from_pretrained(
            self.settings.model_id, device_map=self.device, torch_dtype=torch.float16
        )
        self.template: Template = PromptManager.get_prompt_template(
            self.settings.chat_template
        )
        self.multimodal = self.settings.multimodal

    def _tokenize(self, text: str, images: list[Image.Image]) -> torch.Tensor:
        if len(images) > 0 and self.multimodal:
            return self.processor(
                text, images=images, return_tensors="pt"
            ).input_ids.to(self.device)
        else:
            return self.processor(text, return_tensors="pt")

    def _construct_prompt(
        self, question: str, search_results: list[SearchResult]
    ) -> str:
        context = {"role": "context", "content": list()}
        for result in search_results:
            if result.image is not None and self.multimodal:
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
        return self.template.render(messages=[context, user_message])

    def _get_input_prompt_length(self, inputs: dict | torch.LongTensor) -> int:
        if self.multimodal:
            return len(inputs[0])
        return len(inputs["input_ids"][0])

    def _generate(
        self, inputs: dict | torch.LongTensor, **kwargs
    ) -> GenerateOutput | torch.LongTensor:
        logger.debug(f"Starting generation with {kwargs}...")
        if self.multimodal:
            inputs = inputs.to(self.device)
            return self.model.generate(inputs, **kwargs).to("cpu")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model.generate(**inputs, **kwargs).to("cpu")

    @torch.no_grad()
    def generate(
        self, question: str, search_results: list[SearchResult], **kwargs
    ) -> str:
        prompt = self._construct_prompt(question, search_results)
        images = [result.image for result in search_results if result.image is not None]
        inputs = self._tokenize(prompt, images)
        input_prompt_length = self._get_input_prompt_length(inputs)
        outputs = self._generate(inputs, **kwargs)
        # decode only new tokens to string
        answer = self.processor.decode(
            outputs[0][input_prompt_length:], skip_special_tokens=True
        )
        torch.cuda.empty_cache()
        return answer
