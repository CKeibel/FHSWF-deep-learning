from pathlib import Path

import huggingface_hub
import torch
from dynaconf import settings
from gradio.utils import NamedString
from loguru import logger
from unstructured.cleaners.core import clean

from backend.causal_models.factory import CausalLMFactory
from backend.enums import FileExtensions
from backend.file_handling.chunker import TextChunkerFactory
from backend.file_handling.extractors import PdfExtractor
from backend.retriever.factory import (DenseRetrieverFactory,
                                       SparseRetrieverFactory)
from backend.schemas import (ExtractedContent, GenerationConfig, SearchResult,
                             StoreEntry)
from backend.storage.factory import VectorStoreFactory
from backend.storage.store_base import VectorStoreBase


class Service:
    def __init__(self) -> None:
        Service.hf_login()
        self.chunker = TextChunkerFactory.create_text_chunker(settings.CHUNKER)
        self.vector_store: VectorStoreBase = VectorStoreFactory.create_vector_storage(
            settings.VECTOR_STORE
        )
        self.retrieve_n = settings.RETRIEVE_N
        self.dense_retriever = DenseRetrieverFactory.get_model(
            settings.DENSE_RETRIEVER_NAME
        )
        self.causal_model = CausalLMFactory.get_model(settings.CAUSAL_MODEL_NAME)
        self.generation_config = GenerationConfig(
            max_new_tokens=250,
            no_repeat_ngram_size=3,
            temperature=1.0,
            top_k=85,
            num_beams=1,
            do_sample=True,
            length_penalty=-0.7,
        )
        self.sparse_retriever = SparseRetrieverFactory.get_model("bm25")

    def insert_files(self, files: list[NamedString]) -> None:
        for i, path in enumerate(files):
            extraced_content: ExtractedContent | None = None
            try:
                file_path = Path(path)
                logger.info(
                    f"{i + 1}/{len(files)}: Processing document: {file_path.name}"
                )
            except Exception as e:
                logger.error(f"Error file {path} not readable:\n{e}")
                continue

            if file_path.suffix == FileExtensions.PDF:
                extraced_content = PdfExtractor.extract_content(file_path)

            # chunk text
            if extraced_content:
                # Clean text
                cleaned_text = clean(
                    extraced_content.full_text,
                    bullets=True,
                    extra_whitespace=True,
                    dashes=True,
                    trailing_punctuation=True,
                )
                logger.info("Chunking document...")
                chunked_texts = self.chunker.chunk_text(cleaned_text)
                logger.debug(
                    f"Chunked '{file_path.name}' into {len(chunked_texts)} parts."
                )

                # Insert texts
                if len(chunked_texts) > 0:
                    dense_text_vectors = self.dense_retriever.vectorize(chunked_texts)
                    # Dense embeddings
                    self.vector_store.insert(
                        StoreEntry(
                            type="text",
                            document_name=file_path.name,
                            content=chunked_texts,
                            vector=dense_text_vectors,
                        )
                    )
                    # Sparse embeddings
                    self.sparse_retriever.add_documents(chunked_texts)

                # Images
                if len(extraced_content.images) > 0:

                    # Image embeddings
                    if self.dense_retriever.is_multimodal():
                        dense_image_vectors = self.dense_retriever.vectorize(
                            extraced_content.images
                        )
                        if dense_image_vectors is not None:
                            self.vector_store.insert(
                                StoreEntry(
                                    type="image",
                                    document_name=file_path.name,
                                    content=extraced_content.images,
                                    vector=dense_image_vectors,
                                )
                            )

                    # Caption embeddings
                    dense_caption_vectors = self.dense_retriever.vectorize(
                        [img.caption for img in extraced_content.images]
                    )
                    if dense_caption_vectors is not None:
                        self.vector_store.insert(
                            StoreEntry(
                                type="caption",
                                document_name=file_path.name,
                                content=extraced_content.images,
                                vector=dense_caption_vectors,
                            )
                        )
        logger.info("Finished processing all documents.")

    def inference(self, query: str) -> str:
        logger.info(f"Got following user query: {query}")
        logger.info(f"Trying to find the {self.retrieve_n} best matching documents...")

        # Dense search
        query_vector = self.dense_retriever.vectorize([query])
        dense_result = self.vector_store.query(query_vector, self.retrieve_n)
        # Sparse search
        sparse_results = self.sparse_retriever.search(query)

        combined_results = self.reciprocal_rank_fusion(dense_result, sparse_results)

        return self.causal_model.generate(
            query, combined_results, **self.generation_config.dict()
        )

    # Reciprocal Rank Fusion
    def reciprocal_rank_fusion(
        self, dense_results: list[SearchResult], sparse_results: list[str], k: int = 60
    ) -> list[SearchResult]:
        # Store text to image mapping
        image_mapping = dict()
        dense_texts: list[str] = list()
        for result in dense_results:
            if result.image:
                image_mapping[result.text] = result.image
            dense_texts.append(result.text)

        # Combine found docs
        all_docs = set(sparse_results + dense_texts)

        # Reciprocal Rank Fusion
        scores = dict()
        for doc in all_docs:
            score = 0
            # Calculate score contribution from sparse results
            if doc in sparse_results:
                rank = sparse_results.index(doc) + 1
            else:
                rank = len(sparse_results) + 1
            score += 1 / (k + rank)

            # Calculate score contribution from dense results
            if doc in dense_results:
                rank = dense_results.index(doc) + 1
            else:
                rank = len(dense_results) + 1
            score += 1 / (k + rank)

            scores[doc] = score

        # Rank results
        ranked_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            : self.retrieve_n
        ]
        return [
            SearchResult(text=doc[0], image=image_mapping.get(doc[0]))
            for doc in ranked_documents
        ]

    @staticmethod
    def hf_login(secret: str | None = None) -> None:
        if secret is None:
            try:
                secret = settings.HF_SECRET
            except:
                logger.warning(
                    "Couldn't login into huggingface hub. Check if 'HF_SECRET environment variable is set.'"
                )
                return

        if secret:
            huggingface_hub.login(secret)
            logger.info("Logged in to Hugging Face.")
        else:
            logger.warning(
                "No Hugging Face token found. Please set 'HF_SECRET' environment variable."
            )

    def update_generation_config(self, generation_config: GenerationConfig) -> None:
        self.generation_config = generation_config
        logger.info(f"Updated generation config: {self.generation_config.dict()}")

    def change_model(self, model_name: str) -> None:
        del self.causal_model
        torch.cuda.empty_cache()
        logger.info(f"Changing model to: {model_name}")
        self.causal_model = CausalLMFactory.get_model(model_name)
        logger.info(f"Loading {model_name} complete!")


service = Service()
