import numpy as np
from lancedb.pydantic import LanceModel, Vector

from backend.schemas import StoreEntry
from backend.storage.store_base import VectorStoreBase


class MySchema(LanceModel):
    vector: Vector(128)
    text: str
    category: str


class LanceDB(VectorStoreBase):
    def __init__(self) -> None:
        pass

    def query(self, query_vector: np.ndarray, k=1):  # TODO: define return type
        pass

    def insert(self, entry: StoreEntry) -> None:  # TODO: Document type
        pass

    def delete(self, document_ids) -> None:  # TODO: param
        pass
