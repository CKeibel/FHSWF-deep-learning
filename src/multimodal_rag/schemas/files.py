from pydantic import BaseModel


class ExtractedFileContent(BaseModel):
    text: str
    images: list
