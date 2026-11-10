<div align="center">
  <img src="assets/fie-hcmute-logo.png" alt="Logo khoa FIE HCMUTE" width="160">
  <h1>Rô AI RAG Assistant</h1>
  <p>Hệ thống tra cứu tài liệu học tập có căn cứ cho sinh viên.</p>
</div>

## Mục tiêu

Sinh viên thường phải tìm lại quy định, công thức, bảng số liệu và hướng dẫn trong nhiều tài liệu, đặc biệt khi ôn tập hoặc làm bài thi mở tài liệu nhưng thời gian có hạn. Rô AI dùng Retrieval-Augmented Generation (RAG) để tìm các đoạn liên quan trước, sau đó chỉ tạo câu trả lời dựa trên evidence được chọn.

Đây là hệ thống RAG theo tài liệu, không phải chatbot trả lời theo kiến thức không kiểm soát. Nếu evidence không đủ, hệ thống trả về thông báo an toàn thay vì suy đoán.

## Thành phần đã triển khai

- Nạp TXT, PDF có text extractable và DOCX; giữ metadata nguồn khi có.
- Chunking fixed, recursive và structure-aware với chunk ID ổn định.
- Embedding MiniLM/BGE-M3, kiểm tra dimension và persistence Qdrant.
- Dense retrieval có Top-K, threshold, lọc metadata, xếp hạng ổn định và khử trùng lặp.
- Context Builder chọn evidence theo token budget, không cắt nội dung làm hỏng citation.
- Gemini RAG có prompt tiếng Việt grounded, citation mapping và xử lý lỗi API.
- Qdrant Local mặc định; Docker và Cloud chỉ được dùng khi cấu hình backend rõ ràng.
- Evaluation có Recall@K, Precision@K, MRR, nDCG@K, Hit@K và artifact đã lưu trong `results/`.

OCR, hybrid search, reranking, UI và API web chưa được triển khai.

## Kiến trúc

`Ingestion → Chunking → Embedding → Qdrant → Dense Retrieval → Context Builder → Gemini`

Chi tiết sơ đồ và các giới hạn nghiên cứu: [docs/architecture.md](docs/architecture.md).

## Cài đặt

Môi trường phát triển hiện dùng Conda `ro-ai-rag` trên Windows:

```powershell
& "C:\Users\Lenovo\anaconda3\envs\ro-ai-rag\python.exe" -m pip check
Copy-Item .env.example .env
```

Không commit `.env`, API key, Qdrant Local storage hay dữ liệu học tập riêng tư. Chỉ cài dependency khi môi trường hiện có thiếu gói cần thiết:

```powershell
& "C:\Users\Lenovo\anaconda3\envs\ro-ai-rag\python.exe" -m pip install -r requirements.txt
```

## Cấu hình

`config.yaml` là nguồn cấu hình không nhạy cảm. Biến bí mật đọc từ `.env` hoặc environment variables:

- `GEMINI_API_KEY`, `GEMINI_MODEL`
- `QDRANT_URL`, `QDRANT_API_KEY`
- `EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`

Mặc định dùng Qdrant Local tại `qdrant_storage/`, collection `rag_demo`. Không đổi model trong một collection đã index.

### Qdrant Docker

```powershell
docker compose up -d
$env:QDRANT_URL = "http://localhost:6335"
```

Đặt `vectordb.backend: docker` trong `config.yaml`. Docker Compose dùng named volume `qdrant_docker_storage`; không tự di chuyển Local storage.

### Qdrant Cloud

Đặt endpoint HTTPS và API key trong `.env`, sau đó chọn backend Cloud:

```env
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-api-key
```

```yaml
vectordb:
  backend: cloud
  cloud_collection_name: ro_ai_handbook_cloud
```

Cloud dùng collection và metadata path riêng. Không upload tài liệu riêng tư nếu chưa được phép.

## Sử dụng

Đặt tài liệu được phép sử dụng vào `data/`, rồi chạy retrieval không dùng LLM:

```powershell
& "C:\Users\Lenovo\anaconda3\envs\ro-ai-rag\python.exe" main.py --no-llm --question "Sinh viên được mượn bao nhiêu quyển sách?"
```

Chỉ gọi Gemini khi thêm `--gemini` và đã có `GEMINI_API_KEY`:

```powershell
& "C:\Users\Lenovo\anaconda3\envs\ro-ai-rag\python.exe" main.py --gemini --question "Sinh viên được mượn bao nhiêu quyển sách?"
```

## Đánh giá và kiểm thử

Chạy toàn bộ regression tests:

```powershell
& "C:\Users\Lenovo\anaconda3\envs\ro-ai-rag\python.exe" -m unittest discover -s tests -v
```

Benchmark dùng run ID mới để không ghi đè artifact:

```powershell
& "C:\Users\Lenovo\anaconda3\envs\ro-ai-rag\python.exe" evaluate.py --run-id phase3-check-001
```

Kết quả benchmark hiện có chỉ phản ánh corpus minh họa và ground truth đã lưu; không đại diện cho quy định chính thức hay mọi tài liệu học tập. Xem methodology tại `docs/research/`.

## An toàn Git và dữ liệu

- Kiểm tra `git status` trước khi stage thay đổi.
- Không stage `.env`, Qdrant storage, log, temporary storage hoặc tài liệu không được phép chia sẻ.
- Chỉ công bố repository/tài liệu khi đã kiểm tra quyền truy cập và thông tin cá nhân.
- Không dùng force push, reset, prune hoặc xóa collection để xử lý xung đột lịch sử.

## Tài liệu nghiên cứu

- [Kiến trúc](docs/architecture.md)
- [Embedding Evaluation V2](docs/research/embedding_evaluation_v2.md)
- [Retrieval baseline](docs/research/retrieval_evaluation_baseline.md)
- [So sánh công nghệ](docs/research/technology_comparison.md)
