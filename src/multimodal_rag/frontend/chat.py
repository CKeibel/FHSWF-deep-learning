from contextlib import contextmanager
import gradio as gr


class ChatService:
    @staticmethod
    def response():
        pass


@contextmanager
def chat_tab():
    with gr.Tab("Chat") as chat_tab:
        chatbot = gr.Chatbot(
            height=500,
        )
        msg = gr.Textbox(label="User input", placeholder="Type your message here")
        msg.submit(ChatService.response, inputs=[msg, chatbot], outputs=[msg, chatbot])
        _ = gr.ClearButton([msg, chatbot])

    yield chat_tab
