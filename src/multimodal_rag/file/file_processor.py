# Strategy pattern
from abc import ABC, abstractmethod


class FileProcessor(ABC):
    @staticmethod
    @abstractmethod
    def process() -> None:  # TODO
        pass

    @staticmethod
    @abstractmethod
    def chunk() -> None:  # TODO
        pass


class UnstructuredIOFileProcessor(FileProcessor):
    def process() -> None:  # TODO
        pass

    def chunk() -> None:  # TODO
        pass


class PyPDFFileProcessor(FileProcessor):
    def process() -> None:  # TODO
        pass

    def chunk() -> None:  # TODO
        pass
