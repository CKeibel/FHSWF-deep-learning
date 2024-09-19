from contextlib import contextmanager

import gradio as gr

from backend.schemas import GenerationConfig
from backend.service import service


def answer(message: str, history: list[str]) -> tuple[str, list[list[str]]]:
    reply = service.inference(message)
    history.append([message, reply])
    return "", history


def update_generation_config(*args):

    service.update_generation_config(
        GenerationConfig(
            max_new_tokens=args[0],
            no_repeat_ngram_size=args[1],
            temperature=args[2],
            top_k=int(args[3]),
            num_beams=args[4],
            do_sample=True,
        )
    )


class SettingsService:
    @staticmethod
    def change_model(model_choice):
        pass

    @staticmethod
    def update_generation_config():
        pass


models = {
    "meta-llama/Meta-Llama-3.1-8B (language)": "llama3_8b",
    "meta-llama/Meta-Llama-3.1-8B-Instruct (language)": "llama3_8b_instruct",
    "HuggingFaceM4/idefics2-8b-chatty (multimodal)": "idefics2_chat",
}


@contextmanager
def chat_tab():
    with gr.Tab("Chat") as chat_tab:
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    scale=2,
                )
                msg = gr.Textbox(
                    label="User input", placeholder="Type your message here"
                )
                msg.submit(answer, inputs=[msg, chatbot], outputs=[msg, chatbot])
                _ = gr.ClearButton([msg, chatbot])

            with gr.Column(scale=1):
                gr.Markdown("# Model selection")
                with gr.Column():
                    model_choice = gr.Dropdown(
                        choices=list(models.keys()),  # Todo: set the choices (models)
                        value=[],  # Todo: set the default value
                        label="Model",
                    )
                    btn = gr.Button("Select")
                    btn.click(
                        SettingsService.change_model, inputs=model_choice, outputs=None
                    )
                gr.Markdown("# Generation config")
                with gr.Column():
                    new_tokens_slider = gr.Slider(
                        minimum=50,
                        maximum=1000,
                        step=1.0,
                        label="Max. new tokens",
                        info="Test info",
                        value=service.generation_config.max_new_tokens,
                    )
                    n_gram_slider = gr.Slider(
                        minimum=0,
                        maximum=15,
                        step=1.0,
                        label="No repeat n-gram size",
                        value=service.generation_config.no_repeat_ngram_size,
                    )
                    temp_slider = gr.Slider(
                        minimum=0,
                        maximum=5,
                        step=0.1,
                        label="Temperature",
                        value=service.generation_config.temperature,
                    )
                    top_k_slider = gr.Slider(
                        minimum=0,
                        maximum=99,
                        step=1.0,
                        label="Top K",
                        value=service.generation_config.top_k,
                    )
                    num_beams_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1.0,
                        label="Number of beams",
                        value=service.generation_config.num_beams,
                    )
                    btn = gr.Button("Save")
                    btn.click(
                        update_generation_config,
                        inputs=[
                            new_tokens_slider,
                            n_gram_slider,
                            temp_slider,
                            top_k_slider,
                            num_beams_slider,
                        ],
                        outputs=None,
                    )

    yield chat_tab
