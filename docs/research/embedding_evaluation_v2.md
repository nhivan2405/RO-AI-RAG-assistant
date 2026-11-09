# Embedding Evaluation V2

## Phạm vi công bằng

Mỗi model trong `evaluation.embedding_models` được index từ cùng `--data`, cùng cấu hình chunking, cùng dataset câu hỏi và cùng `top_k`/`min_score` trong `config.yaml`. Benchmark không gọi Gemini, fine-tuning, hybrid search hoặc reranking.

Mỗi model dùng collection và Qdrant Local storage tách biệt theo `run_id` và model tại `results/qdrant_evaluation/` mặc định. Evaluator từ chối storage hoặc file output đã tồn tại, vì vậy không ghi đè collection production hay artifact benchmark cũ.

## Relevance labels

Ground truth dùng một trong hai cách:

- `expected_anchor`: tương thích dataset cũ; chunk liên quan khi payload text chứa anchor.
- `relevant_chunk_ids`: danh sách một hoặc nhiều `chunk_id` liên quan; được ưu tiên khi có mặt.

Relevance là binary. Với `relevant_chunk_ids`, mỗi ID duy nhất là một relevant item; với anchor, có một relevant item. `out_of_scope` chỉ báo cáo ACCEPTED/REJECTED, không tính vào metric relevance.

## Metrics

- Recall@K = số relevant chunk truy xuất được / tổng relevant chunk đã gán nhãn.
- Precision@K = số relevant chunk truy xuất được / K.
- MRR = `1 / rank` của relevant chunk đầu tiên, hoặc 0.
- nDCG@K dùng binary gain `1 / log2(rank + 1)`, chuẩn hóa bằng ideal DCG.
- Hit@K = 1 khi có ít nhất một relevant chunk trong Top-K, ngược lại 0.

Metric tổng hợp và theo category: `regulations`, `numerical_values`, `tables`, `formulas`, `multi_step_reasoning`; category không có query được xuất với `query_count: 0`. `out_of_scope` báo số ACCEPTED/REJECTED.

## Chạy benchmark

```powershell
python -m pip install -r requirements.txt
python evaluate.py --run-id baseline-v2-001
```

Có thể giới hạn model mà không thay đổi production defaults:

```powershell
python evaluate.py --run-id minilm-v2-001 --models sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## Output schema

`results/embedding_evaluation_<run_id>.json` chứa `models[]`; mỗi model gồm `run_configuration`, tổng `metrics`, `by_category`, chẩn đoán từng ground-truth query (`retrieved_chunk_ids`, scores, ranks, relevant IDs) và out-of-scope diagnostics.

`results/embedding_evaluation_<run_id>.csv` có một dòng cho mỗi cặp model/category: `model`, `embedding_dimension`, `category`, `query_count`, các metric có áp dụng, `accepted_count`, `rejected_count`.
