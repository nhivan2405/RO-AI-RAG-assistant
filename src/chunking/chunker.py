"""Chia tài liệu theo số ký tự và giữ lại metadata nguồn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loader import DocumentRecord

ChunkingStrategy = Literal["fixed", "recursive"]


@dataclass
class Chunk:
    """Một đoạn văn bản kèm metadata phục vụ truy xuất."""

    text: str
    source: str
    page: int | None
    chunk_id: str


def make_chunks(
    documents: Sequence[DocumentRecord],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Tương thích ngược: chia đệ quy mặc định theo số ký tự."""
    return chunk_documents(documents, chunk_size, chunk_overlap, strategy="recursive")


def chunk_documents(
    documents: Sequence[DocumentRecord],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    strategy: ChunkingStrategy = "recursive",
) -> list[Chunk]:
    """Chia theo ``fixed`` hoặc ``recursive``; mọi kích thước là ký tự."""
    _validate_chunk_options(chunk_size, chunk_overlap, strategy)
    split_text = _build_splitter(chunk_size, chunk_overlap, strategy)

    chunks: list[Chunk] = []
    for source, page, text in documents:
        chunks.extend(_chunk_document(source, page, text, split_text))
    return chunks


def _validate_chunk_options(
    chunk_size: int,
    chunk_overlap: int,
    strategy: str,
) -> None:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise ValueError("chunk_overlap must be an integer")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require 0 <= chunk_overlap < chunk_size")
    if strategy not in {"fixed", "recursive"}:
        raise ValueError("strategy must be 'fixed' or 'recursive'")


def _build_splitter(
    chunk_size: int,
    chunk_overlap: int,
    strategy: ChunkingStrategy,
):
    if strategy == "fixed":
        return lambda text: _split_fixed_size(text, chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text


def _split_fixed_size(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    step = chunk_size - chunk_overlap
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


def _chunk_document(
    source: str,
    page: int | None,
    text: str,
    split_text: Callable[[str], list[str]],
) -> list[Chunk]:
    return [
        Chunk(part, source, page, f"{source}:p{page}:c{index}")
        for index, part in enumerate(split_text(text))
        if part.strip()
    ]
