"""Đánh giá retrieval V2 giữa các embedding model được cấu hình."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

from src.retrieval.retriever import Retriever


GROUND_TRUTH = "ground_truth"
OUT_OF_SCOPE = "out_of_scope"
EVALUATION_CATEGORIES = (
    "regulations",
    "numerical_values",
    "tables",
    "formulas",
    "multi_step_reasoning",
)


def relevant_hit_ids(
    hits: Sequence[dict[str, Any]], item: dict[str, Any]
) -> set[str]:
    """Trả về các chunk liên quan theo ID, hoặc anchor cho định dạng cũ."""
    expected_ids = set(item.get("relevant_chunk_ids", []))
    if expected_ids:
        return {
            str(hit["chunk_id"])
            for hit in hits
            if str(hit.get("chunk_id")) in expected_ids
        }

    anchor = item.get("expected_anchor")
    if not anchor:
        return set()
    return {
        str(hit["chunk_id"])
        for hit in hits
        if anchor in str(hit.get("text", ""))
    }


def relevant_rank(hits: Sequence[dict[str, Any]], item: dict[str, Any]) -> int | None:
    """Trả về thứ hạng kết quả liên quan đầu tiên."""
    expected_ids = set(item.get("relevant_chunk_ids", []))
    anchor = item.get("expected_anchor")
    for rank, hit in enumerate(hits, start=1):
        if expected_ids and str(hit.get("chunk_id")) in expected_ids:
            return rank
        if not expected_ids and anchor and anchor in str(hit.get("text", "")):
            return rank
    return None


def first_relevant_rank(
    hits: Sequence[dict[str, Any]], expected_anchor: str
) -> int | None:
    """Tương thích ngược với evaluation anchor của baseline."""
    return relevant_rank(hits, {"expected_anchor": expected_anchor})


def expected_relevance_count(item: dict[str, Any]) -> int:
    """Đếm số chunk liên quan được gán nhãn cho một truy vấn."""
    chunk_ids = item.get("relevant_chunk_ids", [])
    return len(set(chunk_ids)) if chunk_ids else 1


def calculate_metrics(query_results: Sequence[dict[str, Any]], top_k: int) -> dict[str, float]:
    """Tính metric binary relevance trên tập truy vấn có ground truth."""
    if not query_results:
        return {}

    recall_sum = 0.0
    precision_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    hit_sum = 0.0
    for result in query_results:
        relevant_retrieved = len(result["relevant_retrieved_ids"])
        relevant_total = result["relevant_total"]
        rank = result["first_relevant_rank"]
        recall_sum += relevant_retrieved / relevant_total
        precision_sum += relevant_retrieved / top_k
        mrr_sum += 1 / rank if rank is not None else 0.0
        hit_sum += float(rank is not None)
        dcg = sum(1 / math.log2(position + 1) for position in result["relevant_ranks"])
        ideal_count = min(relevant_total, top_k)
        idcg = sum(1 / math.log2(position + 1) for position in range(1, ideal_count + 1))
        ndcg_sum += dcg / idcg if idcg else 0.0

    count = len(query_results)
    return {
        f"recall_at_{top_k}": recall_sum / count,
        f"precision_at_{top_k}": precision_sum / count,
        "mrr": mrr_sum / count,
        f"ndcg_at_{top_k}": ndcg_sum / count,
        f"hit_at_{top_k}": hit_sum / count,
    }


def evaluate_queries(
    index: Any,
    questions: Sequence[dict[str, Any]],
    top_k: int,
    min_score: float,
) -> dict[str, Any]:
    """Đánh giá metric và chẩn đoán retrieval cho từng truy vấn."""
    retriever = Retriever(min_score)
    ground_truth_results: list[dict[str, Any]] = []
    out_of_scope_results: list[dict[str, Any]] = []

    # Giữ out-of-scope tách khỏi metric relevance để không làm sai Recall/MRR.
    for item in questions:
        hits = index.search(item["question"], top_k)
        hit_ids = [str(hit.get("chunk_id")) for hit in hits]
        if item["type"] == OUT_OF_SCOPE:
            accepted = retriever.is_answerable(hits)
            out_of_scope_results.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "question": item["question"],
                    "decision": "ACCEPTED" if accepted else "REJECTED",
                    "top_score": float(hits[0]["score"]) if hits else None,
                    "retrieved_chunk_ids": hit_ids,
                    "retrieved_scores": [float(hit["score"]) for hit in hits],
                }
            )
            continue

        retrieved_relevant_ids = relevant_hit_ids(hits, item)
        expected_ids = set(item.get("relevant_chunk_ids", []))
        relevant_ranks = [
            rank
            for rank, hit in enumerate(hits, start=1)
            if str(hit.get("chunk_id")) in retrieved_relevant_ids
        ]
        ground_truth_results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "expected_anchor": item.get("expected_anchor"),
                "relevant_chunk_ids": sorted(expected_ids),
                "relevant_total": expected_relevance_count(item),
                "relevant_retrieved_ids": sorted(retrieved_relevant_ids),
                "relevant_ranks": relevant_ranks,
                "first_relevant_rank": relevant_rank(hits, item),
                "top_score": float(hits[0]["score"]) if hits else None,
                "retrieved_chunk_ids": hit_ids,
                "retrieved_scores": [float(hit["score"]) for hit in hits],
            }
        )

    by_category: dict[str, dict[str, Any]] = {}
    for category in EVALUATION_CATEGORIES:
        category_results = [
            result for result in ground_truth_results if result["category"] == category
        ]
        by_category[category] = {
            "query_count": len(category_results),
            "metrics": calculate_metrics(category_results, top_k),
        }
    out_scope_by_category = {
        "query_count": len(out_of_scope_results),
        "accepted_count": sum(result["decision"] == "ACCEPTED" for result in out_of_scope_results),
        "rejected_count": sum(result["decision"] == "REJECTED" for result in out_of_scope_results),
    }
    by_category[OUT_OF_SCOPE] = out_scope_by_category
    return {
        "metrics": calculate_metrics(ground_truth_results, top_k),
        "by_category": by_category,
        "ground_truth_results": ground_truth_results,
        "out_of_scope_results": out_of_scope_results,
    }


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Đọc dataset và kiểm tra nhãn ground truth đầy đủ."""
    with path.open(encoding="utf-8") as question_file:
        questions = json.load(question_file)
    if not isinstance(questions, list):
        raise ValueError("Evaluation dataset must be a JSON list")

    for item in questions:
        if not isinstance(item, dict) or item.get("type") not in {
            GROUND_TRUTH,
            OUT_OF_SCOPE,
        }:
            raise ValueError("Each item must have a valid type")
        if not all(isinstance(item.get(key), str) and item[key] for key in ("id", "question", "category")):
            raise ValueError("Each item must have non-empty id, question, and category")
        valid_categories = set(EVALUATION_CATEGORIES) | {OUT_OF_SCOPE}
        if item["category"] not in valid_categories:
            raise ValueError("Each item must use a supported evaluation category")
        labels = item.get("relevant_chunk_ids")
        if labels is not None and (
            not isinstance(labels, list) or not all(isinstance(label, str) for label in labels)
        ):
            raise ValueError("relevant_chunk_ids must be a list of strings")
        if item["type"] == GROUND_TRUTH and not item.get("expected_anchor") and not labels:
            raise ValueError("Ground-truth items require expected_anchor or relevant_chunk_ids")
    return questions


def _collection_name(prefix: str, run_id: str, model_name: str) -> str:
    model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return f"{prefix}_{run_id}_{model_slug}"


def _evaluation_storage_path(
    storage_root: Path, run_id: str, model_name: str
) -> Path:
    """Tạo đường dẫn Qdrant riêng cho từng model của một benchmark run."""
    model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return storage_root / run_id / model_slug


def build_evaluation_index(
    data_path: str,
    model_name: str,
    collection: str,
    storage_path: Path,
    chunking_strategy: str | None = None,
) -> tuple[Any, dict[str, Any], int]:
    """Tạo index benchmark và lấy dimension thực tế từ Embedder."""
    from src.chunking.chunker import chunk_documents
    from src.config import load_config
    from src.pipeline import RagIndex, load_documents

    config = load_config()
    if storage_path.exists():
        raise FileExistsError(f"Evaluation storage already exists: {storage_path}")
    if chunking_strategy is not None:
        config = deepcopy(config)
        config["chunking"]["strategy"] = chunking_strategy
    chunking = config["chunking"]
    chunks = chunk_documents(
        load_documents(data_path),
        chunk_size=chunking["chunk_size"],
        chunk_overlap=chunking["chunk_overlap"],
        strategy=chunking["strategy"],
    )
    index = RagIndex(
        model_name=model_name,
        collection=collection,
        storage_path=str(storage_path),
    )
    dimension = index.build(chunks)
    return index, config, dimension


def run_model_evaluation(
    data_path: str,
    model_name: str,
    collection: str,
    storage_path: Path,
    questions: Sequence[dict[str, Any]],
    chunking_strategy: str | None = None,
) -> dict[str, Any]:
    """Đánh giá một model và luôn giải phóng Qdrant Local client."""
    index, config, dimension = build_evaluation_index(
        data_path, model_name, collection, storage_path, chunking_strategy
    )
    try:
        result = evaluate_queries(
            index,
            questions,
            top_k=config["retrieval"]["top_k"],
            min_score=config["retrieval"]["min_score"],
        )
    finally:
        index.close()

    chunks = index.chunks
    integrity = {
        "chunk_count": len(chunks),
        "chunks_with_section": sum(chunk.section is not None for chunk in chunks),
        "unavoidable_splits": sum(chunk.split_reason is not None for chunk in chunks),
        "maximum_chunk_length": max((len(chunk.text) for chunk in chunks), default=0),
    }

    result["run_configuration"] = {
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "collection": collection,
        "storage_path": str(storage_path),
        "data_path": data_path,
        "chunking": config["chunking"],
        "chunking_integrity": integrity,
        "top_k": config["retrieval"]["top_k"],
        "min_score": config["retrieval"]["min_score"],
    }
    return result


def _csv_rows(model_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for category, details in model_result["by_category"].items():
        row = {
            "model": model_result["run_configuration"]["embedding_model"],
            "embedding_dimension": model_result["run_configuration"]["embedding_dimension"],
            "category": category,
            "query_count": details["query_count"],
            **details.get("metrics", {}),
            "accepted_count": details.get("accepted_count", ""),
            "rejected_count": details.get("rejected_count", ""),
        }
        rows.append(row)
    return rows


def export_results(output_dir: Path, run_id: str, results: dict[str, Any]) -> None:
    """Xuất chi tiết JSON và bảng so sánh CSV mà không ghi đè artifact cũ."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"embedding_evaluation_{run_id}.json"
    csv_path = output_dir / f"embedding_evaluation_{run_id}.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("Evaluation output already exists; choose a new --run-id")

    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=False, indent=2)
    rows = [row for result in results["models"] for row in _csv_rows(result)]
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare configured embedding models")
    parser.add_argument("--data", default="data")
    parser.add_argument("--questions", default="data/evaluation_questions.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--storage-path", default="results/qdrant_evaluation")
    parser.add_argument("--collection-prefix", default="rag_evaluation_v2")
    parser.add_argument("--models", nargs="+")
    parser.add_argument(
        "--chunking-strategy",
        choices=("fixed", "recursive", "structure_aware"),
    )
    args = parser.parse_args()

    from src.config import load_config

    config = load_config()
    models = args.models or config["evaluation"]["embedding_models"]
    questions = load_questions(Path(args.questions))
    results: dict[str, Any] = {"run_id": args.run_id, "models": []}
    for model_name in models:
        collection = _collection_name(args.collection_prefix, args.run_id, model_name)
        # Mỗi model dùng storage riêng để không khóa hoặc trộn vector benchmark.
        storage_path = _evaluation_storage_path(
            Path(args.storage_path), args.run_id, model_name
        )
        results["models"].append(
            run_model_evaluation(
                args.data,
                model_name,
                collection,
                storage_path,
                questions,
                args.chunking_strategy,
            )
        )

    export_results(Path(args.output_dir), args.run_id, results)
    print(json.dumps([result["metrics"] for result in results["models"]], ensure_ascii=False))


if __name__ == "__main__":
    main()
