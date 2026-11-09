"""Đánh giá retrieval baseline theo anchor văn bản đã được gán nhãn."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from src.retrieval.retriever import Retriever


MINILM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def first_relevant_rank(
    hits: Sequence[dict[str, Any]], expected_anchor: str
) -> int | None:
    """Trả về thứ hạng hit đầu tiên chứa anchor đã gán nhãn."""
    for rank, hit in enumerate(hits, start=1):
        if expected_anchor in str(hit.get("text", "")):
            return rank
    return None


def evaluate_queries(
    index: Any,
    questions: Sequence[dict[str, Any]],
    top_k: int,
    min_score: float,
) -> dict[str, Any]:
    """Tính Hit@K, MRR và quyết định ngưỡng cho câu hỏi ngoài phạm vi."""
    retriever = Retriever(min_score)
    ground_truth_results: list[dict[str, Any]] = []
    out_of_scope_results: list[dict[str, Any]] = []

    for item in questions:
        hits = index.search(item["question"], top_k)
        if item["type"] == "ground_truth":
            rank = first_relevant_rank(hits, item["expected_anchor"])
            ground_truth_results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "expected_anchor": item["expected_anchor"],
                    "first_relevant_rank": rank,
                    "hit_at_k": rank is not None,
                    "reciprocal_rank": 1 / rank if rank is not None else 0.0,
                    "top_chunk_ids": [hit.get("chunk_id") for hit in hits],
                }
            )
            continue

        accepted = retriever.is_answerable(hits)
        out_of_scope_results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "decision": "ACCEPTED" if accepted else "REJECTED",
                "top_score": float(hits[0]["score"]) if hits else None,
                "top_chunk_ids": [hit.get("chunk_id") for hit in hits],
            }
        )

    query_count = len(ground_truth_results)
    if not query_count:
        raise ValueError("Evaluation dataset must contain ground-truth questions")

    hit_at_k = sum(result["hit_at_k"] for result in ground_truth_results) / query_count
    mrr = sum(result["reciprocal_rank"] for result in ground_truth_results) / query_count
    return {
        "metrics": {f"hit_at_{top_k}": hit_at_k, "mrr": mrr},
        "ground_truth_results": ground_truth_results,
        "out_of_scope_results": out_of_scope_results,
    }


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Đọc và kiểm tra dataset để tránh đánh giá với nhãn không đầy đủ."""
    with path.open(encoding="utf-8") as question_file:
        questions = json.load(question_file)

    if not isinstance(questions, list):
        raise ValueError("Evaluation dataset must be a JSON list")
    for item in questions:
        if not isinstance(item, dict) or item.get("type") not in {
            "ground_truth",
            "out_of_scope",
        }:
            raise ValueError("Each item must have type 'ground_truth' or 'out_of_scope'")
        if not isinstance(item.get("id"), str) or not isinstance(item.get("question"), str):
            raise ValueError("Each item must have string id and question")
        if item["type"] == "ground_truth" and not isinstance(
            item.get("expected_anchor"), str
        ):
            raise ValueError("Ground-truth items must have expected_anchor")
    return questions


def build_baseline_index(data_path: str, collection: str) -> tuple[Any, dict[str, Any]]:
    """Tạo index bằng cấu hình chunking hiện tại và MiniLM 384 chiều."""
    from src.chunking.chunker import chunk_documents
    from src.config import load_config
    from src.pipeline import RagIndex, load_documents

    config = load_config()
    if config["models"]["embedding"] != MINILM_MODEL:
        raise ValueError("Baseline evaluation requires the configured MiniLM embedding model")

    chunking = config["chunking"]
    documents = load_documents(data_path)
    chunks = chunk_documents(
        documents,
        chunk_size=chunking["chunk_size"],
        chunk_overlap=chunking["chunk_overlap"],
        strategy=chunking["strategy"],
    )
    index = RagIndex(model_name=MINILM_MODEL, collection=collection)
    dimension = index.build(chunks)
    if dimension != 384:
        raise ValueError("Baseline evaluation requires MiniLM vectors with dimension 384")
    return index, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG retrieval baseline")
    parser.add_argument("--data", default="data")
    parser.add_argument("--questions", default="data/evaluation_questions.json")
    parser.add_argument("--collection", default="rag_evaluation_baseline")
    parser.add_argument("--output", default="results/retrieval_baseline_results.json")
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    index, config = build_baseline_index(args.data, args.collection)
    result = evaluate_queries(
        index,
        questions,
        top_k=config["retrieval"]["top_k"],
        min_score=config["retrieval"]["min_score"],
    )
    result["run_configuration"] = {
        "data_path": args.data,
        "collection": args.collection,
        "embedding_model": MINILM_MODEL,
        "embedding_dimension": 384,
        "chunking": config["chunking"],
        "top_k": config["retrieval"]["top_k"],
        "min_score": config["retrieval"]["min_score"],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
    print(json.dumps(result["metrics"], ensure_ascii=False))
    print(f"Saved evaluation details to {output_path}")


if __name__ == "__main__":
    main()
