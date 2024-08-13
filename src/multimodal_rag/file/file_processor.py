# Strategy pattern
from abc import ABC, abstractmethod
from pathlib import Path
from gradio.utils import NamedString
from loguru import logger
from multimodal_rag.enums.file_extentions import FileExtensions


class FileProcessor(ABC):
    @staticmethod
    @abstractmethod
    def process(filess) -> None:  # TODO
        pass

    @staticmethod
    @abstractmethod
    def chunk() -> None:  # TODO
        pass


class UnstructuredIOFileProcessor(FileProcessor):
    @staticmethod
    def process(files: list[NamedString]) -> None:  # TODO
        for file in files:
            try:
                file_path = Path(file)
            except Exception as e:
                logger.error(f"Error file not readable documents: {e}")
                continue
            
            # Standardize documents
            UnstructuredIOFileProcessor.__standardize_documents(file_path)

            # Process documents
            UnstructuredIOFileProcessor.process(file)

    @staticmethod
    def chunk() -> None:  # TODO
        pass

    @staticmethod
    def __standardize_documents(file_path: Path) -> None:
        with open(file_path, "rb") as f:
            pass

        # Get file extension (MIME type)
        file_extension = file_path.suffix

        if file_extension == FileExtensions.PDF:
            pass        


class PyPDFFileProcessor(FileProcessor):
    @staticmethod
    def process() -> None:  # TODO
        pass

    @staticmethod
    def chunk() -> None:  # TODO
        pass
