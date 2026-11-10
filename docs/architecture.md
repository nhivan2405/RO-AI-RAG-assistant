# Kiến trúc Rô AI RAG

![Kiến trúc RAG](diagrams/architecture.png)

## Luồng indexing

TXT/PDF/DOCX → kiểm tra định dạng → trích xuất metadata có sẵn → Chunking → Embedding → Qdrant Local/Docker/Cloud.

![Luồng indexing](diagrams/ingestion.png)

## Luồng query

Câu hỏi → Embedding → Dense Retrieval Top-K → Context Builder → Gemini → câu trả lời có citation mapping.

![Indexing và query](diagrams/index_query.png)

## Hướng nghiên cứu mở rộng

![OCR](diagrams/ingestion_ocr.png)

OCR cho PDF scan/ảnh, reranking, hybrid retrieval, UI và FastAPI là **định hướng**, chưa được tính là chức năng đã hoàn thành.
