# Strategy pattern
from abc import ABC, abstractmethod
from pathlib import Path


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
    def process(files) -> None:  # TODO
        file_path = [file.name for file in files]

    @staticmethod
    def chunk() -> None:  # TODO
        pass

    @staticmethod
    def __standardize_documents(file) -> None:
        # Get file extension (MIME type)
        pass


class PyPDFFileProcessor(FileProcessor):
    @staticmethod
    def process() -> None:  # TODO
        pass

    @staticmethod
    def chunk() -> None:  # TODO
        pass
