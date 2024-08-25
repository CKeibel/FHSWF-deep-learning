from multimodal_rag.file.file_processor import PdfProcessor
from multimodal_rag.enums.file_extentions import FileExtensions
from multimodal_rag.schemas.files import ExtractedFileContent
from gradio.utils import NamedString
from loguru import logger
from pathlib import Path


class FileService:
    @staticmethod
    def process_files(files: list[NamedString]) -> None:
        for i, p in enumerate(files):
            try:
                file_path = Path(p)
                logger.info(
                    f"{i + 1}/{len(files)}: Processing document: {file_path.name}"
                )
            except Exception as e:
                logger.error(f"Error file {p} not readable:\n{e}")
                continue

            # fmt: off
            if file_path.suffix == FileExtensions.PDF:
                extraced_content: list[ExtractedFileContent] = (
                    PdfProcessor.extract_content(
                        file_path
                    )
                )

            # chunk
