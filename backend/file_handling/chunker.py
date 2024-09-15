from dynaconf import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class TextChunker:
    def __init__(self) -> None:
        # Init chunk size
        try:
            self.chunk_size: int = settings.CHUNK_SIZE
            logger.debug(f"Using chunk size from settings: {self.chunk_size}")
        except AttributeError:
            self.chunk_size: int = 250
            logger.debug(f"Using default chunk size: {self.chunk_size}")

        # Init chunk overlap
        try:
            self.chunk_overlap: int = settings.CHUNK_OVERLAP
            logger.debug(f"Using chunk overlap from settings: {self.chunk_overlap}")
        except AttributeError:
            self.chunk_overlap: int = 50
            logger.debug(f"Using default chunk overlap: {self.chunk_overlap}")

    def chunk_text(self, text: str) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.create_documents([text])
