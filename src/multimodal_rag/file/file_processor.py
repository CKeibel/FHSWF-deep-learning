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
    def process(files: list[NamedString]) -> list[str]:
        pass


class UnstructuredIOFileProcessor(FileProcessor):
    @staticmethod
    def process(file_paths: list[NamedString]) -> list[str]:
        for i, p in enumerate(file_paths):
            try:
                file_path = Path(p)
                logger.info(
                    f"{i + 1}/{len(file_paths)}: Processing document: {file_path.name}"
                )
            except Exception as e:
                logger.error(f"Error file {p} not readable:\n{e}")
                continue

            # Standardize document
            elements = UnstructuredIOFileProcessor.__standardize_documents(file_path)

            # Chunk document
            chunks = UnstructuredIOFileProcessor.__chunk(elements)

    @staticmethod
    def __chunk(elements: list[Element]) -> list[str]:  # TODO
        pass

    @staticmethod
    def __standardize_documents(file_path: Path) -> list[Element]:
        # Get file extension (MIME type)
        file_extension = file_path.suffix

        if file_extension == FileExtensions.PDF:
            try:
                elements = partition_pdf(file_path, strategy="fast")

            except Exception as e:
                try:
                    elements = UnstructuredIOFileProcessor.__default_partition(
                        file_path
                    )
                    # elements = partition_pdf(file_path)
                    logger.error(f"Error partitioning PDF in high resolution: {e}")
                    logger.info("Attempting to partition PDF without high resolution.")
                except Exception as e:
                    logger.error(f"Error partitioning PDF: {e}")
                    logger.info(
                        "Attempting to partition document using default method."
                    )
                    elements = UnstructuredIOFileProcessor.__default_partition(
                        file_path
                    )
            logger.info(f"Finished processing document: {file_path.name}")
            return elements

        # Default partitioning method
        elements = UnstructuredIOFileProcessor.__default_partition(file_path)

    @staticmethod
    def __default_partition(file_path: Path) -> list[Element]:
        try:
            elements = partition(file_path)
        except Exception as e:
            logger.error(
                f"Error partitioning document ({file_path.name}) with default method: {e}"
            )
            return []

        return elements


class PyPDFFileProcessor(FileProcessor):
    @staticmethod
    def process(files: list[NamedString]) -> list[str]:  # TODO
        pass
