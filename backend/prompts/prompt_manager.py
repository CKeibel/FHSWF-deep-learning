from jinja2 import Environment, FileSystemLoader, Template
from loguru import logger


class PromptManager:
    @staticmethod
    def get_prompt_template(template_name: str) -> Template:
        env = Environment(loader=FileSystemLoader("./backend/prompts/templates"))
        logger.debug(f"Loading prompt template: {template_name}")
        return env.get_template(template_name)
