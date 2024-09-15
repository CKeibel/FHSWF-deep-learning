# Strategy pattern
import io
from abc import ABC, abstractmethod
from pathlib import Path

import pymupdf
from loguru import logger
from PIL import Image
from pymupdf import Document

from backend.schemas import ExtractedFileContent


class ExtractorBase(ABC):
    @staticmethod
    @abstractmethod
    def extract_content(file: Path) -> ExtractedFileContent:
        pass


class PdfExtractor(ExtractorBase):
    @staticmethod
    def extract_content(file: Path) -> ExtractedFileContent:
        text: str = str()
        images: list[Image.Image] = list()

        document: Document = pymupdf.open(file)
        logger.info("Extracting content...")
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

        logger.info("Finished content extraction.")
        return ExtractedFileContent(name=file.name, text=text, images=images)
