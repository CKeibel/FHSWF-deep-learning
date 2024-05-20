import gradio as gr
from multimodal_rag.models import CausalLM
from multimodal_rag.model_config import ModelConfig

model = CausalLM(
    ModelConfig(
        name="HuggingFaceH4/zephyr-7b-beta", path="HuggingFaceH4/zephyr-7b-beta"
    )
)


def response(message: str, history: list[list[str]]) -> tuple[str, list[list[str]]]:
    answer = model.generate(message)
    history.append((message, answer))
    return message, history


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
        with gr.Row():
            gr.Dropdown(["model 1", "model 2", "model 3"], label="Model")


if __name__ == "__main__":
    frontend.launch(share=True)
