# Strategy pattern
from abc import ABC, abstractmethod
from pathlib import Path
from gradio.utils import NamedString
from loguru import logger
from multimodal_rag.enums.file_extentions import FileExtensions
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element


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
    def __standardize_documents(file_path: Path) -> list[Element]:
        # Get file extension (MIME type)
        file_extension = file_path.suffix

        logger.info(f"Partitioning document: {file_path.name}")

        if file_extension == FileExtensions.PDF:
            try:
                elements = partition_pdf(file_path)
            except Exception as e:
                logger.error(f"Error partitioning PDF: {e}")
                logger.info("Attempting to partition document using default method.")
                elements = UnstructuredIOFileProcessor.__default_partition(file_path)
                return elements
            return elements
        
        # Default partitioning method
        elements = UnstructuredIOFileProcessor.__default_partition(file_path)
            
    @staticmethod
    def __default_partition(file_path: Path) -> list[Element]:
        try:
            elements = partition(file_path)
        except Exception as e:
            logger.error(f"Error partitioning document ({file_path.name}) with default method: {e}")
            return []
        
        return elements

class PyPDFFileProcessor(FileProcessor):
    @staticmethod
    def process() -> None:  # TODO
        pass

    @staticmethod
    def chunk() -> None:  # TODO
        pass
