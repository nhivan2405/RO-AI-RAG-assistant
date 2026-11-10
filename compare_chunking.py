"""So sánh fixed-size và recursive chunking theo số ký tự."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.chunking.chunker import Chunk, chunk_documents
from src.ingestion.loader import load_documents


CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
STRATEGIES = ("fixed", "recursive", "structure_aware")
RESULTS_PATH = Path("results/chunk_comparison.csv")


def calculate_metrics(chunks: list[Chunk]) -> dict[str, float | int]:
    """Tính các chỉ số chiều dài chunk theo ký tự."""
    lengths = [len(chunk.text) for chunk in chunks]
    if not lengths:
        return {"total_chunks": 0, "average_length": 0, "maximum_length": 0}

    return {
        "total_chunks": len(chunks),
        "average_length": round(sum(lengths) / len(lengths), 1),
        "maximum_length": max(lengths),
    }


def calculate_integrity_checks(chunks: list[Chunk]) -> dict[str, int]:
    """Đếm tín hiệu toàn vẹn để so sánh chiến lược mà không suy đoán chất lượng."""
    return {
        "chunks_with_section": sum(chunk.section is not None for chunk in chunks),
        "unavoidable_splits": sum(chunk.split_reason is not None for chunk in chunks),
        "empty_chunks": sum(not chunk.text.strip() for chunk in chunks),
        "missing_source": sum(not chunk.source for chunk in chunks),
    }


def print_strategy_result(strategy: str, chunks: list[Chunk]) -> None:
    """In chỉ số và hai chunk đầu với ranh giới văn bản rõ ràng."""
    metrics = calculate_metrics(chunks)
    integrity = calculate_integrity_checks(chunks)
    print(f"\nStrategy: {strategy}")
    print(f"Total chunk count: {metrics['total_chunks']}")
    print(f"Average chunk length: {metrics['average_length']} characters")
    print(f"Maximum chunk length: {metrics['maximum_length']} characters")
    print(f"Chunks with section metadata: {integrity['chunks_with_section']}")
    print(f"Unavoidable structure splits: {integrity['unavoidable_splits']}")

    for position, chunk in enumerate(chunks[:2], start=1):
        print(f"--- Chunk {position} start ({chunk.chunk_id}) ---")
        print(chunk.text)
        print(f"--- Chunk {position} end ({chunk.chunk_id}) ---")


def build_csv_row(strategy: str, chunks: list[Chunk]) -> dict[str, Any]:
    """Tạo một dòng CSV cho mỗi chiến lược chunking."""
    metrics = calculate_metrics(chunks)
    integrity = calculate_integrity_checks(chunks)
    return {
        "strategy": strategy,
        "chunk_size_chars": CHUNK_SIZE,
        "chunk_overlap_chars": CHUNK_OVERLAP,
        "total_chunks": metrics["total_chunks"],
        "average_chunk_length_chars": metrics["average_length"],
        "maximum_chunk_length_chars": metrics["maximum_length"],
        "chunks_with_section": integrity["chunks_with_section"],
        "unavoidable_splits": integrity["unavoidable_splits"],
        "empty_chunks": integrity["empty_chunks"],
        "missing_source": integrity["missing_source"],
        "first_chunk": chunks[0].text if chunks else "",
        "second_chunk": chunks[1].text if len(chunks) > 1 else "",
    }


def export_results(rows: list[dict[str, Any]]) -> None:
    """Tạo thư mục kết quả trước khi ghi tệp CSV."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    documents = load_documents("data")
    rows: list[dict[str, Any]] = []

    # Dùng cùng corpus và tham số để so sánh cấu trúc giữa các chiến lược.
    for strategy in STRATEGIES:
        chunks = chunk_documents(
            documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            strategy=strategy,
        )
        print_strategy_result(strategy, chunks)
        rows.append(build_csv_row(strategy, chunks))

    export_results(rows)
    print(f"\nExported comparison to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
