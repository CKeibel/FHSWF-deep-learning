from multimodal_rag.services.file_service import file_upload_service
import gradio as gr
from contextlib import contextmanager


@contextmanager
def file_upload_tab():
    with gr.Tab("File Upload") as file_upload_tab:
        gr.Markdown(
            "First upload a file and and then you'll be able download it (but only once!)"
        )
        upload_button = gr.UploadButton("Upload", file_count="multiple")
        upload_button.upload(file_upload_service.process_files, upload_button)
        yield file_upload_tab
