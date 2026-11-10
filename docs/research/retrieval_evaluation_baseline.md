# Baseline Evaluation: RAG Retrieval

## Mục đích

Baseline đánh giá retrieval trên `data/sample.txt`, tài liệu thư viện giả lập duy nhất hiện được index. Nó không phải quy định của HCMUTE.

## Cấu hình được cố định

- Chunking `recursive`, `chunk_size=700`, `chunk_overlap=100` ký tự.
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 chiều).
- Qdrant local persistent, Top-5 và `min_score=0.5`.
- Không gọi Gemini khi đánh giá retrieval.

Lệnh dùng collection `rag_evaluation_baseline`, tách biệt với collection ứng dụng mặc định.

## Dataset và ground truth

`data/evaluation_questions.json` có sáu câu hỏi ground truth và ba câu ngoài phạm vi. Ground truth được gán bằng `expected_anchor`: câu/đoạn nguyên văn, duy nhất trong `data/sample.txt`, trực tiếp chứa câu trả lời. Evaluator xem chunk là liên quan nếu payload text chứa anchor. Đây là tham chiếu ổn định theo nội dung, không dùng `source` vì corpus chỉ có một tệp.

Câu ngoài phạm vi không tính vào metric. Chúng được báo cáo riêng với quyết định thực tế `ACCEPTED`/`REJECTED` ở ngưỡng 0.5; không giả định mọi câu ngoài phạm vi đều bị từ chối.

## Chỉ số

- **Hit@5**: tỷ lệ có chunk chứa anchor trong năm kết quả đầu.
- **MRR**: trung bình `1/r`, với `r` là thứ hạng chunk liên quan đầu tiên; bằng 0 khi không có trong Top-5.

Các chỉ số không đo chất lượng Gemini, tính đúng đắn học thuật hoặc khả năng tổng quát sang corpus khác.

## Chạy và lưu kết quả

```powershell
& "C:\Users\Lenovo\anaconda3\envs\ro-ai-rag\python.exe" evaluate.py --run-id baseline-v2-004
```

Lệnh in `hit_at_5`, `mrr` và ghi thứ hạng, `chunk_id`, quyết định ngoài phạm vi vào `results/retrieval_baseline_results.json`.

## Kết quả đã lưu

`results/retrieval_baseline_results.json` lưu baseline trên corpus minh họa: `Hit@5 = 1.0`, `MRR = 0.9167`. Đây là kết quả của sáu câu ground truth trong `data/sample.txt`, không phải đánh giá trên Student Handbook thực tế.

## Giới hạn

- Corpus chỉ có một tài liệu giả lập, không đại diện cho tài liệu học tập FIT.
- Anchor xác nhận chunk chứa thông tin, không đánh giá mọi chunk có thể liên quan.
- Ngưỡng 0.5 là bộ lọc ban đầu; câu ngoài phạm vi vẫn có thể `ACCEPTED` nếu tương đồng ngữ nghĩa với corpus.
