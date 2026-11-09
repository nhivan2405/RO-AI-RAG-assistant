"""Nạp nội dung văn bản từ các định dạng tài liệu được hỗ trợ."""
from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile

from docx import Document
from pypdf.errors import PdfReadError
from pypdf import PdfReader

from src.utils.helpers import get_logger

DocumentRecord = tuple[str, int | None, str]
logger = get_logger("ingestion")


def load_documents(folder: str = "data") -> list[DocumentRecord]:
    """Nạp TXT, PDF có văn bản và DOCX trong một thư mục.

    Kết quả lần lượt là ``(source, page, text)``; kích thước chunk dùng ký tự.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        logger.error("document_folder_missing")
        raise FileNotFoundError(f"Document folder not found: {folder_path}")

    try:
        paths = sorted(folder_path.iterdir())
    except OSError as error:
        logger.error("document_folder_unreadable error_type=%s", type(error).__name__)
        raise RuntimeError("Document folder cannot be read. Verify folder permissions.") from error

    documents: list[DocumentRecord] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            documents.extend(_load_document(path))
        except (OSError, UnicodeError, ValueError, BadZipFile, PdfReadError) as error:
            logger.error(
                "document_load_failed extension=%s error_type=%s",
                path.suffix.lower(),
                type(error).__name__,
            )
            raise RuntimeError(
                f"Unable to read {path.suffix.lower()} document. "
                "Verify that the file is valid and readable."
            ) from error

    logger.info("documents_loaded record_count=%s", len(documents))
    return documents


def _load_document(path: Path) -> list[DocumentRecord]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _load_text_file(path)
    if suffix == ".pdf":
        return _load_pdf_file(path)
    if suffix == ".docx":
        return _load_docx_file(path)
    return []


def _load_text_file(path: Path) -> list[DocumentRecord]:
    text = path.read_text(encoding="utf-8")
    return _record_if_not_blank(path.name, None, text)


def _load_pdf_file(path: Path) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
        documents.extend(_record_if_not_blank(path.name, page_number, page.extract_text() or ""))
    return documents


def _load_docx_file(path: Path) -> list[DocumentRecord]:
    document = Document(path)
    text = "\n\n".join(_iter_docx_blocks(document))
    return _record_if_not_blank(path.name, None, text)


def _iter_docx_blocks(document: Document) -> list[str]:
    """Giữ thứ tự paragraph và bảng như trong body DOCX."""
    paragraphs = {paragraph._p: paragraph for paragraph in document.paragraphs}
    tables = {table._tbl: table for table in document.tables}
    blocks: list[str] = []
    for element in document.element.body.iterchildren():
        if element in paragraphs:
            text = _format_docx_paragraph(paragraphs[element])
        elif element in tables:
            text = _format_docx_table(tables[element])
        else:
            continue
        if text.strip():
            blocks.append(text)
    return blocks


def _format_docx_paragraph(paragraph) -> str:
    """Đánh dấu heading DOCX để chunking giữ được cấu trúc nguồn."""
    style_name = paragraph.style.name if paragraph.style is not None else ""
    if style_name.lower().startswith("heading"):
        return f"# {paragraph.text}"
    return paragraph.text


def _format_docx_table(table) -> str:
    """Chuẩn hóa bảng DOCX thành Markdown để giữ header và các dòng."""
    rows = [
        [cell.text.strip().replace("\n", " ") for cell in row.cells]
        for row in table.rows
    ]
    if not rows or not rows[0]:
        return ""
    width = len(rows[0])
    normalized_rows = [row[:width] + [""] * max(0, width - len(row)) for row in rows]
    header = normalized_rows[0]
    separator = ["---"] * width
    markdown_rows = [header, separator, *normalized_rows[1:]]
    return "\n".join("| " + " | ".join(row) + " |" for row in markdown_rows)


def _record_if_not_blank(source: str, page: int | None, text: str) -> list[DocumentRecord]:
    if not text.strip():
        return []
    return [(source, page, text)]
