import gradio as gr
import time

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.
# https://www.gradio.app/docs/gradio/multimodaltextbox#demos




with gr.Blocks() as frontend:
    # Chat Tab
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(
            [],
            bubble_full_width=True
        )
        msg = gr.Textbox(
            label="User input",
        )
    
    # Vector store Tab
    with gr.Tab("Storage"):
        pass
    
    # Settings Tab
    with gr.Tab("Settings"):
        with gr.Row():
            gr.Dropdown(["model 1", "model 2", "model 3"], label="Model")


if __name__ == "__main__":
    frontend.launch(share=True)
