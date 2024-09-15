from pathlib import Path

from dynaconf import settings
from gradio.utils import NamedString
from loguru import logger

from backend.enums import FileExtensions
from backend.file_handling.chunker import TextChunker
from backend.file_handling.extractors import PdfExtractor
from backend.retriever.factory import DenseRetrieverFactory
from backend.schemas import ExtractedFileContent, StoreEntry
from backend.storage.factory import VectorStoreFactory
from backend.storage.store_base import VectorStoreBase


class StoreService:
    def __init__(self) -> None:
        self.chunker = TextChunker()
        self.vector_store: VectorStoreBase = VectorStoreFactory.create_vector_storage(
            settings.VECTOR_STORE
        )
        self.dense_retriever = DenseRetrieverFactory.get_model(
            settings.DENSE_RETRIEVER_NAME
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

            if file_path.suffix == FileExtensions.PDF:
                extraced_content = PdfExtractor.extract_content(file_path)

            # chunk text
            if extraced_content:
                logger.info("Chunking document...")
                chunked_texts = [
                    doc.page_content
                    for doc in self.chunker.chunk_text(extraced_content.text)
                ]
                logger.debug(
                    f"Chunked '{file_path.name}' into {len(chunked_texts)} parts."
                )

                # Insert texts
                if len(chunked_texts) > 0:
                    dense_vectors = self.dense_retriever.vectorize(chunked_texts)
                    self.vector_store.insert(
                        StoreEntry(
                            type="text",
                            document_name=file_path.name,
                            content=chunked_texts,
                            vector=dense_vectors,
                        )
                    )
                # Images
                if len(extraced_content.images) > 0:
                    dense_vectors = self.dense_retriever.vectorize(
                        extraced_content.images
                    )
                    self.vector_store.insert(
                        StoreEntry(
                            type="image",
                            document_name=file_path.name,
                            content=extraced_content.images,
                            vector=dense_vectors,
                        )
                    )
