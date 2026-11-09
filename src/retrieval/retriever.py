"""Đánh giá kết quả truy xuất trước khi tạo câu trả lời."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.utils.helpers import get_logger


FALLBACK_MESSAGE = "Không tìm thấy đủ thông tin trong tài liệu được cung cấp."
logger = get_logger("retrieval")


class Retriever:
    """Áp dụng ngưỡng score lên hit đứng đầu mà không thay đổi danh sách hit."""

    def __init__(self, min_score: float) -> None:
        self.min_score = min_score

    def is_answerable(self, hits: Sequence[dict[str, Any]]) -> bool:
        """Trả về liệu hit tốt nhất có đạt ngưỡng để trả lời hay không."""
        if not hits:
            logger.info("retrieval_decision status=REJECTED top_score=none min_score=%s", self.min_score)
            return False
        top_score = float(hits[0]["score"])
        is_accepted = top_score >= self.min_score
        status = "ACCEPTED" if is_accepted else "REJECTED"
        logger.info(
            "retrieval_decision status=%s top_score=%.4f min_score=%.4f",
            status,
            top_score,
            self.min_score,
        )
        return is_accepted

    def filter_hits(
        self,
        hits: Sequence[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Lọc score, ổn định thứ hạng và loại chunk trùng lặp."""
        if top_k is not None and top_k <= 0:
            return []
        ranked = sorted(
            hits,
            key=lambda hit: (-float(hit["score"]), str(hit.get("chunk_id", ""))),
        )
        unique_hits: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for hit in ranked:
            chunk_id = str(hit.get("chunk_id", ""))
            if chunk_id in seen_chunk_ids or float(hit["score"]) < self.min_score:
                continue
            seen_chunk_ids.add(chunk_id)
            unique_hits.append(hit)
            if top_k is not None and len(unique_hits) == top_k:
                break
        return unique_hits
