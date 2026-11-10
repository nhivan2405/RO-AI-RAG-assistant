# Thí nghiệm

Hiện các script có sẵn nằm tại **thư mục gốc** để dễ chạy theo hướng dẫn: `cosine_demo.py`, `compare_chunking.py`, `evaluate.py`.

- `cosine_demo.py`: minh họa cosine similarity.
- `compare_chunking.py`: so sánh số lượng/độ dài chunks (không phải Recall@K).
- `evaluate.py`: benchmark retrieval với ground truth, metrics và output JSON/CSV tách biệt theo run ID.

Corpus hiện tại là dữ liệu minh họa. Khi có ground truth đại diện hơn, giữ nguyên corpus, query set, chunking và retrieval parameters cho mỗi lần so sánh.
