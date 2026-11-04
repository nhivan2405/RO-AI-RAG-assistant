# Source

`pipeline.py` điều phối embedding, retrieval Qdrant local persistent và Gemini tùy chọn, đồng thời tái xuất `load_documents` và `make_chunks` để tương thích với mã gọi cũ.

- `config.py`: đọc và kiểm tra `config.yaml`.
- `ingestion/loader.py`: nạp TXT, PDF có extractable text và DOCX.
- `chunking/chunker.py`: chia đoạn fixed-size hoặc recursive theo ký tự; recursive là mặc định.
- `embeddings/embedder.py`: tạo embedding MiniLM hoặc BGE-M3 và kiểm tra dimension model.
- `vectordb/vector_store.py`: quản lý Qdrant persistent, metadata collection và chống index chunk không đổi.

FastAPI/OCR chưa tích hợp.
