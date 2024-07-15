import torch
import torch.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod

from multimodal_rag.llm_config import CausalLMConfig


class CausalLMBase(ABC):
    @abstractmethod
    def generate(self, input: str, **kwargs) -> str:
        pass


class LanguageModel(CausalLMBase):
    def __init__(self, config: CausalLMConfig) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(config.path)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.path, device_map=self.device, torch_dtype=torch.bfloat16
        )

    def __tokenize(self, text: str) -> torch.Tensor:
        """Function to tokenize the input text and return the input_ids tensor.
        Args:
            text (str): The input text to be tokenized.
        Returns:
            torch.Tensor: The input_ids tensor.
        """
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    @torch.no_grad()
    def generate(self, input: str, **kwargs) -> str:
        """Function to generate the answer for the given input text.
        Args:
            input (str): The input text for which the answer needs to be generated.
        Returns:
            str: The generated answer.
        """
        inputs_ids = self.__tokenize(input)  # TODO: RAG Template prompt
        outputs = self.model.generate(inputs_ids, **kwargs)  # TODO: generation config
        # decode only new tokens to string
        answer = self.tokenizer.decode(
            outputs[0][len(inputs_ids[0]) :], skip_special_tokens=True
        )
        torch.cuda.empty_cache()
        return answer


class MultimodalModel(CausalLMBase):
    pass
