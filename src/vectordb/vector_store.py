"""Lưu trữ Qdrant local persistent với kiểm tra tính tương thích collection."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from src.chunking.chunker import Chunk
from src.utils.helpers import get_logger


MANIFEST_NAME = ".rag_collection_metadata.json"
POINT_NAMESPACE = uuid.UUID("6a94ac74-c5ea-4aa5-b2e4-c2820fe1191a")
logger = get_logger("vectordb")


class PersistentVectorStore:
    """Quản lý một collection Qdrant local cho một model embedding."""

    def __init__(
        self,
        storage_path: str | Path,
        collection_name: str,
        model_name: str,
        vector_dimension: int,
    ) -> None:
        if vector_dimension <= 0:
            raise ValueError("vector_dimension must be positive")

        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        self.model_name = model_name
        self.vector_dimension = vector_dimension
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.storage_path))
            self._ensure_compatible_collection()
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
            logger.error("qdrant_initialization_failed error_type=%s", type(error).__name__)
            raise

        logger.info("qdrant_initialized collection=%s", self.collection_name)

    def index_chunks(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Chỉ ghi các chunk mới hoặc đã thay đổi và trả về số chunk được ghi."""
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks must match number of vectors")
        if not chunks:
            return 0

        try:
            point_data = [
                self._build_point_data(chunk, vector)
                for chunk, vector in zip(chunks, vectors)
            ]
            existing_hashes = self._get_existing_hashes(
                [item["id"] for item in point_data]
            )
            points = [
                models.PointStruct(
                    id=item["id"], vector=item["vector"], payload=item["payload"]
                )
                for item in point_data
                if existing_hashes.get(item["id"]) != item["payload"]["content_hash"]
            ]
            if points:
                self.client.upsert(collection_name=self.collection_name, points=points)
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
            logger.error("qdrant_indexing_failed error_type=%s", type(error).__name__)
            raise

        logger.info("qdrant_indexing_completed indexed_count=%s", len(points))
        return len(points)

    def search(
        self,
        vector: Sequence[float],
        top_k: int,
        source: str | None = None,
        section: str | None = None,
        page: int | None = None,
    ) -> list[dict[str, Any]]:
        """Truy vấn cosine có Top-K và metadata filter tùy chọn."""
        self._validate_vector(vector)
        if top_k <= 0:
            return []
        try:
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=list(vector),
                limit=top_k,
                with_payload=True,
                query_filter=_build_metadata_filter(source, section, page),
            )
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
            logger.error("qdrant_search_failed error_type=%s", type(error).__name__)
            raise
        return _rank_unique_hits(result.points)

    def close(self) -> None:
        """Đóng client để dữ liệu local được giải phóng trước khi mở lại."""
        self.client.close()

    def _ensure_compatible_collection(self) -> None:
        manifest = self._read_manifest()
        metadata = manifest.get(self.collection_name)
        if self.client.collection_exists(self.collection_name):
            self._validate_existing_collection(metadata)
            return

        if metadata is not None:
            raise ValueError(
                f"Collection metadata exists but Qdrant collection '{self.collection_name}' is missing"
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        manifest[self.collection_name] = self._collection_metadata()
        self._write_manifest(manifest)

    def _validate_existing_collection(self, metadata: dict[str, Any] | None) -> None:
        if metadata is None:
            raise ValueError(
                f"Collection '{self.collection_name}' has no managed model metadata; re-index into a new collection"
            )
        if metadata != self._collection_metadata():
            raise ValueError(
                f"Collection '{self.collection_name}' is incompatible with model '{self.model_name}' "
                f"or vector dimension {self.vector_dimension}"
            )

        collection = self.client.get_collection(self.collection_name)
        configured_dimension = _get_collection_dimension(collection)
        if configured_dimension != self.vector_dimension:
            raise ValueError(
                f"Collection '{self.collection_name}' uses dimension {configured_dimension}, "
                f"not {self.vector_dimension}"
            )

    def _build_point_data(self, chunk: Chunk, vector: Sequence[float]) -> dict[str, Any]:
        self._validate_vector(vector)
        content_hash = _content_hash(chunk)
        point_id = str(uuid.uuid5(POINT_NAMESPACE, chunk.chunk_id))
        return {
            "id": point_id,
            "vector": list(vector),
            "payload": {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
                "section": chunk.section,
                "split_reason": chunk.split_reason,
                "content_hash": content_hash,
            },
        }

    def _get_existing_hashes(self, point_ids: list[str]) -> dict[str, str]:
        existing_points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        return {
            str(point.id): str(point.payload.get("content_hash", ""))
            for point in existing_points
            if point.payload is not None
        }

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.vector_dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match {self.vector_dimension}"
            )

    def _collection_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "vector_dimension": self.vector_dimension,
        }

    def _manifest_path(self) -> Path:
        return self.storage_path / MANIFEST_NAME

    def _read_manifest(self) -> dict[str, dict[str, Any]]:
        path = self._manifest_path()
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as manifest_file:
            contents = json.load(manifest_file)
        if not isinstance(contents, dict):
            raise ValueError("Vector store metadata manifest must be a JSON object")
        return contents

    def _write_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        with self._manifest_path().open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)


def _get_collection_dimension(collection: Any) -> int:
    vectors = collection.config.params.vectors
    if not isinstance(vectors, models.VectorParams):
        raise ValueError("Named vectors are not supported by this vector store")
    return vectors.size


def _content_hash(chunk: Chunk) -> str:
    content = "\x1f".join(
        (
            chunk.chunk_id,
            chunk.source,
            str(chunk.page),
            str(chunk.section),
            str(chunk.split_reason),
            chunk.text,
        )
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _build_metadata_filter(
    source: str | None,
    section: str | None,
    page: int | None,
) -> models.Filter | None:
    conditions = []
    for key, value in (("source", source), ("section", section), ("page", page)):
        if value is not None:
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
            )
    return models.Filter(must=conditions) if conditions else None


def _rank_unique_hits(points: Sequence[Any]) -> list[dict[str, Any]]:
    ranked = sorted(
        points,
        key=lambda hit: (
            -float(hit.score),
            str((hit.payload or {}).get("chunk_id", "")),
            str(hit.id),
        ),
    )
    seen_chunk_ids: set[str] = set()
    hits: list[dict[str, Any]] = []
    for hit in ranked:
        payload = dict(hit.payload or {})
        chunk_id = str(payload.get("chunk_id", hit.id))
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        hits.append({"score": float(hit.score), **payload})
    return hits
