from frontend.gradio_frontend import gradio_frontend
from loguru import logger


def main() -> None:
    logger.info("Starting Gradio frontend...")
    with gradio_frontend() as frontend:
        frontend.launch(share=True)


if __name__ == "__main__":
    main()
