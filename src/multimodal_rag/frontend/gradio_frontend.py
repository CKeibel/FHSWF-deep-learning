from contextlib import contextmanager
import gradio as gr
from multimodal_rag.frontend.chat import chat_tab
from multimodal_rag.frontend.settings import settings_tab
from multimodal_rag.frontend.vector_store import vector_store_tab


@contextmanager
def gradio_frontend():
    with gr.Blocks() as frontend:
        # Chat Tab
        with chat_tab():
            pass

        # Vector store Tab
        with vector_store_tab():
            pass

        # Settings Tab
        with settings_tab():
            pass

    yield frontend
