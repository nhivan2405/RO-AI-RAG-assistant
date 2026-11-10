"""Tầng gọi LLM có kiểm soát evidence."""

from .rag_service import RagAnswer, answer_with_rag

__all__ = ["RagAnswer", "answer_with_rag"]
