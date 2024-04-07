import pytest
from rag.utils import ModelConfig


@pytest.mark.parametrize(
    "config",
    [
        {
            "causal_lm": {"name": "gpt2", "load_4bit": True, "load_8bit": False},
            "encoder": {"name": "sbert", "path": "path"},
        },
        {"causal_lm": {"name": "gpt2", "load_4bit": True, "load_8bit": False}},
        {"encoder": {"name": "sbert", "path": "path"}},
        {},
    ],
)
def test_model_config(config: dict):
    if "causal_lm" in config:
        assert ModelConfig(**config).causal_lm != None
    if "encoder" in config:
        assert ModelConfig(**config).encoder != None
    if "causal_lm" not in config and "encoder" not in config:
        with pytest.raises(Exception):
            ModelConfig(**config)
