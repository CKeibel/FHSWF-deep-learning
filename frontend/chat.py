from contextlib import contextmanager

import gradio as gr

from backend.service import service


def answer(message: str, history: list[str]) -> tuple[str, list[list[str]]]:
    reply = service.inference(message)
    history.append([message, reply])
    return "", history


@contextmanager
def chat_tab():
    with gr.Tab("Chat") as chat_tab:
        chatbot = gr.Chatbot(
            height=500,
        )
        msg = gr.Textbox(label="User input", placeholder="Type your message here")
        msg.submit(answer, inputs=[msg, chatbot], outputs=[msg, chatbot])
        _ = gr.ClearButton([msg, chatbot])

    yield chat_tab
