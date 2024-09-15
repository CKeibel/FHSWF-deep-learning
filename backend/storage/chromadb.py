import uuid

import chromadb
import numpy as np
from dynaconf import settings
from loguru import logger

from backend.schemas import StoreEntry
from backend.storage.store_base import VectorStoreBase


class ChromaDB(VectorStoreBase):
    def __init__(self) -> None:
        self.client = chromadb.Client()  # TODO: persistent client
        self.store = self.client.create_collection(name=settings.CHROMA_COLLECTION_NAME)

    def query(self, query_vector: np.ndarray, k=1):  # TODO: define return type
        pass

    def insert(self, entry: StoreEntry) -> None:  # TODO: Document type

        # Image handling
        if entry.type == "image":
            # Convert Image to numpy array
            images = [np.array(img) for img in entry.content]
            self.store.add(
                ids=[f"{uuid.uuid4()}" for _ in entry.content],
                images=images,
                metadatas=[
                    {"document_name": entry.document_name} for _ in entry.content
                ],
                embeddings=[[0.0, 0.0] for _ in entry.content],  # TODO
            )
        else:
            self.store.add(
                ids=[f"{uuid.uuid4()}" for _ in entry.content],
                documents=entry.content,
                metadatas=[
                    {"document_name": entry.document_name} for _ in entry.content
                ],
                embeddings=[[0.0, 0.0] for _ in entry.content],  # TODO
            )
        logger.info("Inserted documents into ChromaDB")

    def delete(self, document_ids) -> None:  # TODO: param
        pass
