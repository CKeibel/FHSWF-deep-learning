import gradio as gr
from multimodal_rag.llm import LanguageModel
from multimodal_rag.llm_config import CausalLMConfig, read_model_config
from multimodal_rag.model_manager import ModelManager


models = read_model_config()

model_manager = ModelManager(list(models.values())[0])


settings = {
    "generation_config": {
        "temperature": 1.0,
        "max_new_tokens": 250,
        "pad_token_id": model_manager.model.tokenizer.pad_token_id,
        "no_repeat_ngram_size": 3,
    }
}


def response(message: str, history: list[list[str]]) -> tuple[str, list[list[str]]]:
    answer = model_manager.model.generate(message, **settings["generation_config"])
    history.append((message, answer))
    return message, history


def change_model(key: str) -> None:
    gr.Info("Loading model, this could take a while.")
    model_manager.change_model(models[key])
    settings["generation_config"]["pad_token_id"] = model_manager.model.tokenizer.pad_token_id
    gr.Info(f"Model {key} ready to use!")


def update_generation_config(
    max_new_tokens: float, temperature: float, no_repeat_ngram_size: float
) -> None:
    settings["generation_config"]["max_new_tokens"] = max_new_tokens
    settings["generation_config"]["temperature"] = temperature
    settings["generation_config"]["no_repeat_ngram_size"] = no_repeat_ngram_size
    gr.Info("Generation config saved.")
    return None


with gr.Blocks() as frontend:
    # Chat Tab
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(
            height=500,
        )
        msg = gr.Textbox(label="User input", placeholder="Type your message here")
        msg.submit(response, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear = gr.ClearButton([msg, chatbot])

    # Vector store Tab
    with gr.Tab("Storage"):
        pass

    # Settings Tab
    with gr.Tab("Settings"):
        gr.Markdown("# Model selection")
        with gr.Row():
            model_choice = gr.Dropdown(
                choices=[choice for choice in models.keys()],
                value=list(models.keys())[0],
                label="Model",
            )
            btn = gr.Button("Select")
            btn.click(change_model, inputs=model_choice, outputs=None)
        gr.Markdown("# Generation config")
        with gr.Column():
            new_tokens_slider = gr.Slider(
                minimum=50,
                maximum=1000,
                step=1.0,
                label="Max. new tokens",
                info="Test info",
                value=settings["generation_config"]["max_new_tokens"],
            )
            temp_slider = gr.Slider(
                minimum=0,
                maximum=99,
                step=1.0,
                label="Temperature",
                value=settings["generation_config"]["temperature"],
            )
            n_gram_slider = gr.Slider(
                minimum=0,
                maximum=15,
                step=1.0,
                label="No repeat n-gram size",
                value=settings["generation_config"]["no_repeat_ngram_size"],
            )
            btn = gr.Button("Save")
            btn.click(
                update_generation_config,
                inputs=[new_tokens_slider, temp_slider, n_gram_slider],
                outputs=None,
            )


if __name__ == "__main__":
    frontend.launch(share=True)
