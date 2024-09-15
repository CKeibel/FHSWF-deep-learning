from backend.file_handling.extractors import PdfExtractor
from backend.file_handling.chunker import TextChunker
from backend.storage.vector_store import VectorStore
from backend.storage.factory import VectorStoreFactory
from backend.enums import FileExtensions
from backend.schemas import ExtractedFileContent
from gradio.utils import NamedString
from loguru import logger
from pathlib import Path
from dynaconf import settings


class FileService:
    def __init__(self) -> None:
        self.chunker = TextChunker()
        self.vector_store: VectorStore = VectorStoreFactory.create_vector_storage(
            settings.VECTOR_STORE
        )

    def insert_files(self, files: list[NamedString]) -> None:
        for i, path in enumerate(files):
            extraced_content: ExtractedFileContent | None = None
            try:
                file_path = Path(path)
                logger.info(
                    f"{i + 1}/{len(files)}: Processing document: {file_path.name}"
                )
            except Exception as e:
                logger.error(f"Error file {path} not readable:\n{e}")
                continue

            # fmt: off
            if file_path.suffix == FileExtensions.PDF:
                extraced_content = PdfExtractor.extract_content(file_path)

            # chunk text
            if extraced_content:
                logger.info("Chunking document...")
                chunked_content = [doc.page_content for doc in self.chunker.chunk_text(extraced_content.text)]
                logger.debug(f"Chunked '{file_path.name}' into {len(chunked_content)} parts.")
