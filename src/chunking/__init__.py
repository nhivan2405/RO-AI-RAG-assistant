"""Các chiến lược chia đoạn tài liệu."""

from .chunker import Chunk, chunk_documents, make_chunks

__all__ = ["Chunk", "chunk_documents", "make_chunks"]
