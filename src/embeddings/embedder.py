"""Tạo embedding bằng một SentenceTransformer nhất quán."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.helpers import get_logger


MODEL_DIMENSIONS = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "BAAI/bge-m3": 1024,
}
logger = get_logger("embedding")


class Embedder:
    """Bao bọc một model embedding cho cả index và query."""

    def __init__(self, model_name: str, model: SentenceTransformer | None = None) -> None:
        self.model_name = model_name
        self.expected_dimension = _get_expected_dimension(model_name)
        self._model = model or self._initialize_model(model_name)
        self.dimension: int | None = None

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Tạo embedding chuẩn hóa cho danh sách văn bản."""
        if not texts:
            return []

        try:
            vectors = self._model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except (OSError, RuntimeError, ValueError) as error:
            logger.error("embedding_generation_failed error_type=%s", type(error).__name__)
            raise RuntimeError(
                "Unable to generate embeddings. Verify the embedding model and runtime."
            ) from error
        return self._validate_vectors(vectors, len(texts))

    def embed_query(self, text: str) -> list[float]:
        """Tạo embedding chuẩn hóa bằng chính model của tài liệu."""
        vectors = self.embed_texts([text])
        return vectors[0]

    def _validate_vectors(self, vectors: object, expected_count: int) -> list[list[float]]:
        array = np.asarray(vectors, dtype=float)
        if array.ndim != 2 or array.shape[0] != expected_count:
            raise ValueError("Embedding model returned an unexpected vector shape")
        if array.shape[1] != self.expected_dimension:
            logger.error(
                "embedding_dimension_mismatch expected=%s actual=%s",
                self.expected_dimension,
                array.shape[1],
            )
            raise ValueError(
                f"Model '{self.model_name}' returned dimension {array.shape[1]}; "
                f"expected {self.expected_dimension}"
            )
        if not np.isfinite(array).all():
            raise ValueError("Embedding model returned non-finite vector values")

        self.dimension = int(array.shape[1])
        return array.tolist()

    @staticmethod
    def _initialize_model(model_name: str) -> SentenceTransformer:
        try:
            return SentenceTransformer(model_name)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            logger.error("embedding_initialization_failed error_type=%s", type(error).__name__)
            raise RuntimeError(
                "Unable to initialize embedding model. Verify the model configuration and runtime."
            ) from error


def _get_expected_dimension(model_name: str) -> int:
    try:
        return MODEL_DIMENSIONS[model_name]
    except KeyError as error:
        supported_models = ", ".join(MODEL_DIMENSIONS)
        raise ValueError(
            f"Unsupported embedding model '{model_name}'. Supported models: {supported_models}"
        ) from error
