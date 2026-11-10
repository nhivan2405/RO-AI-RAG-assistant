"""Điều phối index RAG và duy trì các API tương thích cũ."""
from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv

from src.chunking.chunker import Chunk, make_chunks
from src.config import load_config
from src.embeddings.embedder import Embedder
from src.ingestion.loader import load_documents
from src.llm.gemini import build_grounded_prompt, generate_grounded_answer
from src.prompts.context_builder import build_context
from src.retrieval.retriever import Retriever
from src.utils.helpers import get_logger
from src.vectordb.vector_store import PersistentVectorStore

load_dotenv()
logger = get_logger("pipeline")


def cosine_manual(a, b):
    """Tính cosine để phục vụ ví dụ và kiểm thử tương thích."""
    first_vector = np.asarray(a, dtype=float)
    second_vector = np.asarray(b, dtype=float)
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    if denominator == 0:
        raise ValueError("Zero vector")
    return float(np.dot(first_vector, second_vector) / denominator)


def _remote_metadata_path(vectordb_config, backend):
    # Tách manifest remote để collection Cloud không dùng metadata Docker.
    if backend == "cloud":
        return vectordb_config.get("cloud_metadata_path", "qdrant_cloud_metadata")
    return vectordb_config.get("metadata_path", "qdrant_docker_metadata")


class RagIndex:
    """Tương thích ngược cho index RAG dùng persistent vector store."""

    def __init__(self, model_name=None, collection=None, storage_path=None):
        config = load_config()
        self.model_name = model_name or os.getenv(
            "EMBED_MODEL", config["models"]["embedding"]
        )
        self.embedder = Embedder(self.model_name)
        vectordb_config = config["vectordb"]
        backend = vectordb_config.get("backend", "local")
        default_collection = (
            vectordb_config.get("cloud_collection_name", "rag_cloud")
            if backend == "cloud"
            else vectordb_config["collection_name"]
        )
        self.collection = collection or default_collection
        configured_storage = storage_path or vectordb_config["storage_path"]
        # Local giữ manifest cạnh storage cũ để không làm thay đổi dữ liệu hiện hữu.
        metadata_path = (
            configured_storage
            if backend == "local"
            else _remote_metadata_path(vectordb_config, backend)
        )
        self.vector_store = PersistentVectorStore(
            storage_path=configured_storage,
            collection_name=self.collection,
            model_name=self.model_name,
            vector_dimension=self.embedder.expected_dimension,
            backend=backend,
            url=os.getenv(
                "QDRANT_URL",
                vectordb_config.get("url", "http://localhost:6333"),
            ),
            api_key=os.getenv("QDRANT_API_KEY"),
            metadata_path=metadata_path,
        )
        self.client = self.vector_store.client
        self.chunks = []

    def close(self):
        """Đóng Qdrant client sau khi hoàn tất pipeline độc lập."""
        self.vector_store.close()

    def build(self, chunks):
        if not chunks:
            raise ValueError("No usable chunks")
        self.chunks = chunks
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        dimension = self.embedder.dimension
        if dimension != self.vector_store.vector_dimension:
            raise ValueError("Embedding dimension does not match vector store dimension")
        self.vector_store.index_chunks(chunks, vectors)
        return dimension

    def search(
        self,
        question,
        top_k=5,
        source=None,
        section=None,
        page=None,
        min_score=None,
    ):
        query = self.embedder.embed_query(question)
        if source is None and section is None and page is None:
            hits = self.vector_store.search(query, top_k)
        else:
            hits = self.vector_store.search(
                query,
                top_k,
                source=source,
                section=section,
                page=page,
            )
        # Giữ raw hits cho evaluator; chỉ lọc khi caller yêu cầu threshold.
        if min_score is None:
            return hits
        return Retriever(min_score).filter_hits(hits, top_k)

def make_prompt(question, hits):
    """Giữ API prompt cũ nhưng dùng Context Builder hiện tại."""
    context = build_context(hits)
    return build_grounded_prompt(question, context)


def answer_with_gemini(question, hits):
    """Giữ API Gemini cũ và ủy quyền cho tầng LLM grounded."""
    config = load_config()
    context = build_context(
        hits,
        token_budget=config.get("prompts", {}).get("context_token_budget", 1200),
    )
    try:
        return generate_grounded_answer(
            question,
            context,
            model_name=os.getenv("GEMINI_MODEL", config["models"]["gemini"]),
            timeout_seconds=config.get("llm", {}).get("timeout_seconds", 30),
        )
    except RuntimeError as error:
        logger.error("legacy_gemini_request_failed error_type=%s", type(error).__name__)
        raise
