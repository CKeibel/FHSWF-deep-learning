import uuid
from pathlib import Path

import chromadb
import numpy as np
from dynaconf import settings
from loguru import logger
from PIL import Image

from backend.schemas import ExtractedImage, SearchResult, StoreEntry
from backend.storage.store_base import VectorStoreBase


class ChromaDB(VectorStoreBase):
    def __init__(self) -> None:
        self.database_path = Path(settings.CHROMA_DB_PATH)
        self.client = chromadb.PersistentClient(path=str(self.database_path))
        self.store = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME
        )

    def query(self, query_vector: np.ndarray, k=10) -> list[SearchResult]:
        store_results = self.store.query(query_vector, n_results=k)
        results = []
        if "documents" in store_results:
            for text, metadata in zip(
                store_results["documents"][0], store_results["metadatas"][0]
            ):
                # Load image if available
                img = None
                if "image_path" in metadata:
                    img = Image.open(metadata["image_path"])

                # Append result
                results.append(
                    SearchResult(
                        text=text, document_name=metadata["document_name"], image=img
                    )
                )

        return results

    def insert(self, entry: StoreEntry) -> None:  # TODO: Document type

        # Image handling
        if entry.type == "image":
            # Convert Image to numpy array
            images = [np.array(img.image) for img in entry.content]
            self.store.add(
                ids=[f"{uuid.uuid4()}" for _ in entry.content],
                images=images,
                metadatas=[
                    {"document_name": entry.document_name, "caption": img.caption}
                    for img in entry.content
                ],
                embeddings=entry.vector,
            )
            logger.info("Inserted image embeddings into ChromaDB")

        elif entry.type == "text":
            self.store.add(
                ids=[f"{uuid.uuid4()}" for _ in entry.content],
                documents=entry.content,
                metadatas=[
                    {"document_name": entry.document_name} for _ in entry.content
                ],
                embeddings=entry.vector,
            )
            logger.info("Inserted text embeddings into ChromaDB")
        elif entry.type == "caption":
            self.store.add(
                ids=[f"{uuid.uuid4()}" for _ in entry.content],
                documents=[img.caption for img in entry.content],
                metadatas=[
                    {
                        "document_name": entry.document_name,
                        "image_path": self._save_image_to_disk(img),
                    }
                    for img in entry.content
                ],
                embeddings=entry.vector,
            )
            logger.info("Inserted caption embeddings into ChromaDB")
        else:
            logger.error(f"Invalid entry type: {entry.type}")

        logger.info("Finished inserting content.")

    def delete(self, document_ids) -> None:  # TODO: param
        pass

    def _save_image_to_disk(self, img: ExtractedImage) -> str:
        image_root_path = self.database_path / Path(f"images/{img.document_name}")
        image_root_path.mkdir(parents=True, exist_ok=True)
        image_path = image_root_path / f"{img.id}.png"
        img.image.save(image_path)
        return str(image_path)
