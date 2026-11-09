"""Chia tài liệu theo số ký tự và giữ lại metadata nguồn."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Literal, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loader import DocumentRecord
from src.utils.helpers import get_logger

ChunkingStrategy = Literal["fixed", "recursive", "structure_aware"]
logger = get_logger("chunking")


@dataclass
class Chunk:
    """Một đoạn văn bản kèm metadata phục vụ truy xuất."""

    text: str
    source: str
    page: int | None
    chunk_id: str
    section: str | None = None
    split_reason: str | None = None


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
    """Chia theo chiến lược chọn trước; mọi kích thước là ký tự."""
    _validate_chunk_options(chunk_size, chunk_overlap, strategy)
    chunks: list[Chunk] = []
    for source, page, text in documents:
        if strategy == "structure_aware":
            chunks.extend(_chunk_structure_aware(source, page, text, chunk_size, chunk_overlap))
        else:
            split_text = _build_splitter(chunk_size, chunk_overlap, strategy)
            chunks.extend(_chunk_document(source, page, text, split_text))
    logger.info("chunking_completed strategy=%s chunk_count=%s", strategy, len(chunks))
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
    if strategy not in {"fixed", "recursive", "structure_aware"}:
        raise ValueError("strategy must be 'fixed', 'recursive', or 'structure_aware'")


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


def _chunk_structure_aware(
    source: str,
    page: int | None,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    parts = _structure_aware_parts(text, chunk_size, chunk_overlap)
    chunks = [
        Chunk(
            text=part_text,
            source=source,
            page=page,
            chunk_id=f"{source}:p{page}:c{index}",
            section=section,
            split_reason=split_reason,
        )
        for index, (part_text, section, split_reason) in enumerate(parts)
        if part_text.strip()
    ]
    split_count = sum(chunk.split_reason is not None for chunk in chunks)
    if split_count:
        logger.warning("structure_aware_unavoidable_splits count=%s", split_count)
    return chunks


def _structure_aware_parts(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, str | None, str | None]]:
    parts: list[tuple[str, str | None, str | None]] = []
    for section, section_text in _split_sections(text):
        parts.extend(_pack_section(section_text, section, chunk_size, chunk_overlap))
    return parts


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        if _is_heading(line):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return sections


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    return bool(
        re.match(r"^(#{1,6}\s+|\d+(?:\.\d+)*[.)]?\s+)", stripped)
        or re.match(r"^[A-ZÀ-Ỵ][A-ZÀ-Ỵ0-9\s]{3,}:?$", stripped)
    )


def _pack_section(
    section_text: str,
    section: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, str | None, str | None]]:
    units = _combine_formula_with_definitions(_paragraph_units(section_text))
    parts: list[tuple[str, str | None, str | None]] = []
    current: list[str] = []
    current_length = 0
    prefix = f"{section}\n\n" if section else ""
    available_size = chunk_size - len(prefix)
    if available_size <= 0:
        raise ValueError("chunk_size must be larger than a section heading")
    for unit in units:
        if _is_markdown_table(unit) and len(unit) > available_size:
            if current:
                parts.append((prefix + "\n\n".join(current), section, None))
                current, current_length = [], 0
            parts.extend(_split_oversized_table(unit, section, chunk_size, chunk_overlap))
            continue
        if len(unit) > available_size:
            if current:
                parts.append((prefix + "\n\n".join(current), section, None))
                current, current_length = [], 0
            parts.extend(_split_oversized_unit(unit, section, available_size, chunk_overlap))
            continue
        added_length = len(unit) if not current else len(unit) + 2
        if current and current_length + added_length > available_size:
            parts.append((prefix + "\n\n".join(current), section, None))
            current = _overlap_units(current, chunk_overlap)
            current_length = len("\n\n".join(current))
            if current and current_length + len(unit) + 2 > available_size:
                current, current_length = [], 0
            current.append(unit)
            current_length = len("\n\n".join(current))
        else:
            current.append(unit)
            current_length += added_length
    if current:
        parts.append((prefix + "\n\n".join(current), section, None))
    return parts


def _paragraph_units(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _combine_formula_with_definitions(units: list[str]) -> list[str]:
    combined: list[str] = []
    index = 0
    while index < len(units):
        unit = units[index]
        if (
            _contains_formula(unit)
            and index + 1 < len(units)
            and _looks_like_definition(units[index + 1])
        ):
            combined.append(f"{unit}\n\n{units[index + 1]}")
            index += 2
        else:
            combined.append(unit)
            index += 1
    return combined


def _contains_formula(text: str) -> bool:
    return "=" in text or bool(re.search(r"\b[A-Za-z]\s*[+*/^]\s*[A-Za-z0-9]", text))


def _looks_like_definition(text: str) -> bool:
    return bool(re.match(r"^(where|trong đó|với|ký hiệu|gọi)\b", text.strip(), re.I))


def _is_markdown_table(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) >= 2 and all(line.strip().startswith("|") for line in lines)


def _overlap_units(units: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return []
    retained: list[str] = []
    for unit in reversed(units):
        candidate = [unit, *retained]
        if len("\n\n".join(candidate)) > overlap:
            break
        retained = candidate
    return retained


def _split_oversized_table(
    table: str,
    section: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, str | None, str | None]]:
    lines = [line for line in table.splitlines() if line.strip()]
    header = lines[:2]
    rows = lines[2:]
    prefix = f"{section}\n\n" if section else ""
    parts: list[tuple[str, str | None, str | None]] = []
    current = list(header)
    for row in rows:
        candidate = "\n".join([*current, row])
        if len(prefix + candidate) > chunk_size and len(current) == len(header):
            available_size = chunk_size - len(prefix + "\n".join(header)) - 1
            if available_size <= 0:
                raise ValueError("chunk_size must be larger than a table header")
            row_parts = _build_splitter(available_size, chunk_overlap, "recursive")(row)
            parts.extend(
                (
                    prefix + "\n".join([*header, row_part]),
                    section,
                    "oversized_table_row",
                )
                for row_part in row_parts
                if row_part.strip()
            )
            current = list(header)
            continue
        if len(prefix + candidate) > chunk_size and len(current) > len(header):
            parts.append((prefix + "\n".join(current), section, "oversized_table"))
            current = [*header, row]
        else:
            current.append(row)
    if current:
        parts.append((prefix + "\n".join(current), section, "oversized_table"))
    return parts


def _split_oversized_unit(
    unit: str,
    section: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, str | None, str | None]]:
    splitter = _build_splitter(chunk_size, chunk_overlap, "recursive")
    return [
        (part, section, "oversized_structure")
        for part in splitter(unit)
        if part.strip()
    ]
