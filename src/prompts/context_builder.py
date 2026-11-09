"""Chọn evidence nguyên vẹn và tạo citation mapping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


METADATA_KEYS = ("source", "page", "section", "chunk_id")


@dataclass(frozen=True)
class Citation:
    """Citation gắn với một evidence đã chọn."""

    citation_id: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ContextBundle:
    """Context và citation mapping phục vụ tầng sinh câu trả lời."""

    text: str
    citations: tuple[Citation, ...]
    token_count: int


def build_context(
    hits: Sequence[dict[str, Any]],
    token_budget: int | None = None,
) -> ContextBundle:
    """Chọn evidence theo score mà không cắt nội dung hoặc citation."""
    if token_budget is None:
        from src.config import load_config

        token_budget = load_config()["prompts"]["context_token_budget"]
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")

    selected_parts: list[str] = []
    citations: list[Citation] = []
    used_tokens = 0
    for hit in _rank_unique_evidence(hits):
        citation_id = len(citations) + 1
        evidence = f"[{citation_id}] {hit['text']}"
        evidence_tokens = _count_tokens(evidence)
        if used_tokens + evidence_tokens > token_budget:
            continue
        selected_parts.append(evidence)
        citations.append(Citation(citation_id, _extract_metadata(hit)))
        used_tokens += evidence_tokens

    return ContextBundle("\n\n".join(selected_parts), tuple(citations), used_tokens)


def _rank_unique_evidence(hits: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for position, hit in enumerate(hits):
        text = hit.get("text")
        score = hit.get("score")
        if not isinstance(text, str) or not text.strip() or not isinstance(score, (int, float)):
            continue
        ranked.append((float(score), position, hit))

    seen_keys: set[str] = set()
    unique_hits: list[dict[str, Any]] = []
    for _, _, hit in sorted(ranked, key=lambda item: (-item[0], item[1])):
        key = str(hit.get("chunk_id") or _normalize_text(hit["text"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_hits.append(hit)
    return unique_hits


def _extract_metadata(hit: dict[str, Any]) -> dict[str, Any]:
    return {key: hit[key] for key in METADATA_KEYS if key in hit and hit[key] is not None}


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _count_tokens(text: str) -> int:
    return len(text.split())
