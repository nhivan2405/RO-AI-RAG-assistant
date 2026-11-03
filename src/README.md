# Source

`pipeline.py` điều phối embedding, retrieval Qdrant in-memory và Gemini tùy chọn, đồng thời tái xuất `load_documents` và `make_chunks` để tương thích với mã gọi cũ.

- `config.py`: đọc và kiểm tra `config.yaml`.
- `ingestion/loader.py`: nạp TXT, PDF có extractable text và DOCX.
- `chunking/chunker.py`: chia đoạn fixed-size hoặc recursive theo ký tự; recursive là mặc định.

FastAPI/OCR chưa tích hợp.
