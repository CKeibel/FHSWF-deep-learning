# Strategy pattern
import io
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import pymupdf
from loguru import logger
from PIL import Image
from pymupdf import Document

from backend.schemas import ExtractedContent, ExtractedImage


class ExtractorBase(ABC):
    @staticmethod
    @abstractmethod
    def extract_content(file: Path) -> ExtractedContent:
        pass


class PdfExtractor(ExtractorBase):
    @staticmethod
    def extract_content(file: Path) -> ExtractedContent:
        text: str = str()
        images: list[ExtractedImage] = list()

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
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]
                # Get the bounding box of the image
                img_rect = page.get_image_bbox(img)

                # Expand the bounding box to get nearby text
                expanded_rect = pymupdf.Rect(
                    0, img_rect.y0 + 70, 1440, img_rect.y1 + 70
                )

                # Get the text near the image
                nearby_text = page.get_text("text", clip=expanded_rect)
                images.append(
                    ExtractedImage(
                        id=f"{uuid.uuid4()}",
                        image=Image.open(io.BytesIO(image_bytes)),
                        caption=nearby_text,
                        document_name=file.name,
                    )
                )

        logger.info("Finished content extraction.")
        return ExtractedContent(document_name=file.name, full_text=text, images=images)
