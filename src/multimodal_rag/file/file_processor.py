# Strategy pattern
from abc import ABC, abstractmethod
from pathlib import Path
from loguru import logger
from multimodal_rag.schemas.files import ExtractedFileContent
import pymupdf
from pymupdf import Document
from PIL import Image
import io


class FileProcessor(ABC):
    @staticmethod
    @abstractmethod
    def extract_content(file: Path) -> ExtractedFileContent:
        pass


class PdfProcessor(FileProcessor):
    @staticmethod
    def extract_content(file: Path) -> ExtractedFileContent:
        text: str = str()
        images: list[Image.Image] = list()

        document: Document = pymupdf.open(file)

        for page in document:
            # extracting text
            page_text = (
                page.get_text()
                .encode("utf8")
                .decode("utf-8", errors="replace")
                .replace("\n", " ")
            )
            text += page_text

            # extracting images
            image_list = page.get_images()
            for img in image_list:
                xref = img[0]
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]
                images.append(Image.open(io.BytesIO(image_bytes)))

        logger.info(f"Finished processing document: {file.name}")
        return ExtractedFileContent(text=text, images=images)
