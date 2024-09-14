from backend.file_handling.service import FileService
import gradio as gr
from gradio.utils import NamedString

from contextlib import contextmanager


file_service = FileService()


def process_files(files: list[NamedString]) -> None:
    file_service.process_files(files)


@contextmanager
def file_upload_tab():
    with gr.Tab("File Upload") as file_upload_tab:
        gr.Markdown("Upload files.")
        upload_button = gr.UploadButton("Upload", file_count="multiple")
        upload_button.upload(process_files, upload_button)
        yield file_upload_tab
