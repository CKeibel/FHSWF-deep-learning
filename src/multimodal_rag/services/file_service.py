from multimodal_rag.file.file_processor import FileProcessor

class FileService:
    def __init__(self, file_processor: FileProcessor) -> None:
        self.file_processor = file_processor
    
    def process_files(self, files) -> None:
        self.file_processor.process(files)