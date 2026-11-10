# Bảng so sánh và trạng thái công nghệ

| Thành phần | Đề xuất nghiên cứu | Lý do | Trạng thái |
|---|---|---|---|
| Chunking | Fixed, recursive, structure-aware | So sánh trade-off kích thước và toàn vẹn cấu trúc | Đã triển khai, có regression tests |
| Embedding | MiniLM và BGE-M3 | Đa ngôn ngữ, dense 384/1024 chiều | Đã benchmark trên corpus minh họa |
| Vector DB | Qdrant | Dense similarity và metadata payload | Local; Docker/Cloud có backend cấu hình rõ ràng |
| Retrieval | Dense Top-K | Baseline đánh giá được | Đã có metrics trên corpus minh họa |
| Generation | Gemini API | Sinh trả lời có căn cứ và citation | Cần API key; không đánh giá chất lượng học thuật |
| OCR | Tesseract/PaddleOCR | Tiềm năng đọc PDF scan và ảnh | Chưa tích hợp |
| Backend | FastAPI | API upload/chat | Thiết kế dự kiến, chưa triển khai |

**Lưu ý:** Kết quả benchmark hiện có không khẳng định công nghệ tối ưu hoặc khả năng tổng quát sang tài liệu học tập thực tế.
