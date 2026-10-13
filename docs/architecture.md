# Kiến trúc đề xuất

![Kiến trúc RAG](diagrams/architecture.png)

## Luồng indexing

Upload PDF/DOCX/TXT → kiểm tra định dạng và quyền truy cập → trích xuất → chuẩn hóa → Chunking → Embedding → Qdrant.

![Luồng indexing](diagrams/ingestion.png)

## Luồng query

Câu hỏi → Embedding câu hỏi → Retrieval Top-K → cung cấp context cho Gemini → câu trả lời và tham chiếu tài liệu.

![Indexing và query](diagrams/index_query.png)

## Hướng nghiên cứu mở rộng

![OCR](diagrams/ingestion_ocr.png)

OCR cho PDF scan/ảnh, Reranking, Hybrid retrieval và FastAPI là **định hướng**, chưa được tính là chức năng đã hoàn thành.
