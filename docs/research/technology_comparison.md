# Bảng so sánh và định hướng công nghệ (tháng 9/2026)

| Thành phần | Đề xuất nghiên cứu | Lý do | Trạng thái |
|---|---|---|---|
| Chunking | RecursiveCharacterTextSplitter | Đơn giản, có overlap và dễ so sánh với fixed-size | Đang nghiên cứu/thử nghiệm |
| Embedding | BGE-M3 | Đa ngôn ngữ, biểu diễn dense 1024 chiều | Chưa benchmark BGE-M3 |
| Model chạy prototype nhẹ | multilingual MiniLM | Tải nhẹ hơn để thử trên CPU | Có ví dụ cấu hình |
| Vector DB | Qdrant | Dense similarity và metadata payload | Code demo in-memory |
| Retrieval | Dense Top-K | Baseline đánh giá được | Demo chưa benchmark nghiêm ngặt |
| Generation | Gemini API | Sinh trả lời trên nguồn truy xuất | Tùy API key và model hợp lệ |
| OCR | Tesseract/PaddleOCR | Tiềm năng đọc PDF scan và ảnh | Chưa tích hợp |
| Backend | FastAPI | API upload/chat | Thiết kế dự kiến, chưa triển khai |

**Lưu ý:** Bảng là lựa chọn nghiên cứu, không phải kết quả thực nghiệm hoặc khẳng định công nghệ tối ưu.
