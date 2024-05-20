import gradio as gr
from multimodal_rag.models import CausalLM
from multimodal_rag.model_config import ModelConfig, read_model_config


models = read_model_config()

model = CausalLM(
    ModelConfig(name=list(models.values())[0].name, path=list(models.values())[0].path)
)


def response(message: str, history: list[list[str]]) -> tuple[str, list[list[str]]]:
    answer = model.generate(message)
    history.append((message, answer))
    return message, history


def change_model(key: str) -> None:
    gr.Info("Loading model, this could take a while.")
    model = CausalLM(models[key])
    gr.Info(f"Model {key} ready to use!")


def update_generation_config(max_new_tokens, temperature) -> None:
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
                minimum=250,
                maximum=2500,
                step=1.0,
                label="Max. new tokens",
                info="Test info",
            )
            temp_slider = gr.Slider(
                minimum=0, maximum=99, step=1.0, label="Temperature"
            )
            btn = gr.Button("Save")
            btn.click(
                update_generation_config,
                inputs=[new_tokens_slider, temp_slider],
                outputs=None,
            )


if __name__ == "__main__":
    frontend.launch(share=True)
