"""Nạp nội dung văn bản từ các định dạng tài liệu được hỗ trợ."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from pypdf import PdfReader

DocumentRecord = tuple[str, int | None, str]


def load_documents(folder: str = "data") -> list[DocumentRecord]:
    """Nạp TXT, PDF có văn bản và DOCX trong một thư mục.

    Kết quả lần lượt là ``(source, page, text)``; kích thước chunk dùng ký tự.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Document folder not found: {folder_path}")

    documents: list[DocumentRecord] = []
    for path in sorted(folder_path.iterdir()):
        if not path.is_file():
            continue
        documents.extend(_load_document(path))
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
    text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    return _record_if_not_blank(path.name, None, text)


def _record_if_not_blank(source: str, page: int | None, text: str) -> list[DocumentRecord]:
    if not text.strip():
        return []
    return [(source, page, text)]
