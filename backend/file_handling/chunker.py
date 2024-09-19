from abc import ABC, abstractmethod

import torch
from dynaconf import settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class TextChunker(ABC):
    @abstractmethod
    def chunk_text(self, text: str) -> list[str]:
        pass


class RecursiveTextChunker(TextChunker):
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
            is_separator_regex=True,
        )
        return text_splitter.create_documents([text])


class SemanticTextChunker(TextChunker):
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.model_kwargs = {"device": device}
        self.encode_kwargs = {"normalize_embeddings": True}

        model = HuggingFaceEmbeddings(
            model_name=self.model_id,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs,
        )
        self.text_splitter = SemanticChunker(model)

    def chunk_text(self, text: str) -> list[str]:
        return self.text_splitter.chunk_text(text)


class TextChunkerFactory:
    @staticmethod
    def create_text_chunker(chunker_type: str) -> TextChunker:
        if chunker_type == "recursive":
            return RecursiveTextChunker()
        elif chunker_type == "semantic":
            return SemanticTextChunker()
        else:
            raise ValueError(f"Invalid chunker type: {chunker_type}")
