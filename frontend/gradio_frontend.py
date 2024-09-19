from contextlib import contextmanager

import gradio as gr

from frontend.chat import chat_tab
from frontend.file_upload import file_upload_tab


@contextmanager
def gradio_frontend():
    with gr.Blocks(fill_width=True, fill_height=True) as frontend:
        # Chat Tab
        with chat_tab():
            pass

        # Vector store Tab
        with file_upload_tab():
            pass

    yield frontend
