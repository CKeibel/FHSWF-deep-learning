from backend.file_handling.service import FileService
import gradio as gr
from contextlib import contextmanager


@contextmanager
def file_upload_tab():
    with gr.Tab("File Upload") as file_upload_tab:
        gr.Markdown("Upload files.")
        upload_button = gr.UploadButton("Upload", file_count="multiple")
        upload_button.upload(FileService.process_files, upload_button)
        yield file_upload_tab
