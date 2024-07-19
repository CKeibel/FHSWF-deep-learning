from contextlib import contextmanager
import gradio as gr


@contextmanager
def vector_store_tab():
    with gr.Tab("Storage") as vector_store_tab:
        pass

    yield vector_store_tab
