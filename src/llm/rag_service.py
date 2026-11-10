"""Điều phối retrieval, context và Gemini cho RAG V1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm.gemini import generate_grounded_answer
from src.prompts.context_builder import Citation, build_context
from src.retrieval.retriever import FALLBACK_MESSAGE


@dataclass(frozen=True)
class RagAnswer:
    """Kết quả RAG có citation mapping đã kiểm tra."""

    answer: str
    citations: tuple[Citation, ...]
    is_sufficient: bool


def answer_with_rag(
    question: str,
    index: Any,
    top_k: int,
    min_score: float,
    token_budget: int,
    model_name: str,
    api_key: str | None = None,
    client: Any | None = None,
    hits: list[dict[str, Any]] | None = None,
    timeout_seconds: int = 30,
) -> RagAnswer:
    """Tạo câu trả lời grounded hoặc fallback khi evidence không đủ."""
    if hits is None:
        hits = index.search(question, top_k=top_k, min_score=min_score)
    context = build_context(hits, token_budget=token_budget)
    if not context.text:
        return RagAnswer(FALLBACK_MESSAGE, (), False)
    answer = generate_grounded_answer(
        question,
        context,
        model_name=model_name,
        api_key=api_key,
        client=client,
        timeout_seconds=timeout_seconds,
    )
    if answer == FALLBACK_MESSAGE:
        return RagAnswer(FALLBACK_MESSAGE, (), False)
    return RagAnswer(answer, context.citations, True)
