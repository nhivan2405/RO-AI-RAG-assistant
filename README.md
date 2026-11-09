<div align="center">
  <img src="assets/fie-hcmute-logo.png" alt="Logo Khoa Đào tạo quốc tế - HCMUTE" width="160">

  <h1>Rô AI Assistant</h1>
  <p><strong>Hệ thống hỗ trợ tra cứu và ôn tập tài liệu học tập cho sinh viên FIE HCMUTE</strong></p>
</div>

## Vấn đề

Sinh viên khoa FIE thường phải đọc lại nhiều slide, giáo trình, tài liệu môn học và quy định trước kỳ thi. Với các kỳ thi mở tài liệu, việc được phép mang tài liệu không đồng nghĩa với việc có đủ thời gian tìm đúng phần cần dùng. Khi câu hỏi xuất hiện, việc lật nhiều tệp PDF, DOCX hoặc TXT trong thời gian ngắn dễ làm mất nhịp ôn tập và ảnh hưởng đến khả năng tổng hợp kiến thức.

**Rô AI Assistant** hỗ trợ tra cứu trên chính bộ tài liệu người học cung cấp. Hệ thống được xây dựng theo hướng Retrieval-Augmented Generation (RAG): tìm các đoạn liên quan trong tài liệu trước, sau đó mới dùng chúng làm ngữ cảnh để tạo câu trả lời. Vì vậy, Rô AI là một hệ thống học tập dựa trên tài liệu, không phải chatbot hoặc chatbox trả lời tự do.

## Mục tiêu

- Giúp sinh viên tìm nhanh nội dung trong tài liệu học tập.
- Hỗ trợ ôn tập bằng câu trả lời bám sát các đoạn tài liệu đã truy xuất.
- Hiển thị thông tin nguồn và trang khi dữ liệu có sẵn để người học tự kiểm tra.
- Trả về thông báo rõ ràng khi tài liệu không có đủ căn cứ để trả lời.

## Tính năng đã triển khai

- Đọc tài liệu `.txt`, `.pdf` có văn bản trích xuất được và `.docx` từ thư mục dữ liệu.
- Giữ metadata nguồn, số trang (với PDF) và mã đoạn văn bản.
- Chia tài liệu theo ký tự với hai chiến lược: `recursive` và `fixed`.
- Tạo embedding bằng SentenceTransformers; hỗ trợ `paraphrase-multilingual-MiniLM-L12-v2` và `BAAI/bge-m3`.
- Lưu và tìm kiếm vector bằng Qdrant local persistent.
- Kiểm tra tính tương thích của collection với embedding model và số chiều vector.
- Hạn chế index lại các đoạn không thay đổi.
- Truy xuất Top-K, áp dụng ngưỡng điểm cấu hình được và trả về thông báo khi độ liên quan không đủ.
- Có thể dùng Gemini API để tạo câu trả lời chỉ dựa trên ngữ cảnh đã truy xuất và trích dẫn theo chỉ số nguồn.
- Ghi log ứng dụng vào `logs/app.log`; khóa API được đọc từ `.env`.

> Ngưỡng điểm truy xuất chỉ là bước lọc ban đầu, không bảo đảm câu trả lời luôn đúng. Người học cần đối chiếu lại nguồn tài liệu, đặc biệt trước khi sử dụng trong bài thi hoặc bài tập quan trọng.

## Quy trình RAG

```text
Tài liệu TXT / PDF / DOCX
            ↓
Nạp nội dung và metadata nguồn
            ↓
Chia đoạn văn bản (chunking theo ký tự)
            ↓
Tạo embedding và lưu Qdrant local
            ↓
Câu hỏi → embedding truy vấn → tìm Top-K đoạn liên quan
            ↓
Kiểm tra ngưỡng điểm
            ↓
Gemini tạo câu trả lời có căn cứ, hoặc thông báo thiếu thông tin
```

## Công nghệ sử dụng

| Thành phần | Công nghệ |
| --- | --- |
| Ngôn ngữ | Python |
| Tải tài liệu | pypdf, python-docx |
| Chia đoạn | LangChain Text Splitters |
| Embedding | SentenceTransformers (MiniLM / BGE-M3) |
| Cơ sở dữ liệu vector | Qdrant local persistent |
| Mô hình sinh câu trả lời | Gemini API (tùy chọn) |
| Cấu hình và biến môi trường | PyYAML, python-dotenv |

## Cấu trúc dự án

```text
.
├── data/                   # Tài liệu cần tra cứu
├── src/
│   ├── ingestion/          # Nạp TXT, PDF, DOCX và metadata
│   ├── chunking/           # Chia đoạn theo ký tự
│   ├── embeddings/         # Tạo và kiểm tra embedding
│   ├── vectordb/           # Qdrant local persistent
│   ├── retrieval/          # Truy xuất và kiểm tra ngưỡng điểm
│   ├── utils/              # Logging và tiện ích dùng chung
│   ├── config.py           # Đọc, hợp nhất và kiểm tra config.yaml
│   └── pipeline.py         # Điều phối pipeline RAG
├── tests/                  # Kiểm thử đơn vị
├── config.yaml             # Cấu hình không nhạy cảm
├── .env.example            # Mẫu biến môi trường
├── main.py                 # Điểm chạy ứng dụng dòng lệnh
└── requirements.txt        # Thư viện phụ thuộc
```

## Cài đặt

Yêu cầu: Python 3.11. Chạy tại thư mục gốc dự án trên Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Lần chạy đầu tiên có thể cần kết nối Internet để tải embedding model. Không commit tệp `.env`.

### Cấu hình

`config.yaml` là nguồn cấu hình không nhạy cảm cho chunking, truy xuất, model, Qdrant và logging. Mặc định hệ thống dùng:

- Chunking `recursive`, kích thước `700` ký tự và overlap `100` ký tự.
- Embedding `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Top-K là `5`, ngưỡng điểm là `0.5`.
- Qdrant local tại `qdrant_storage/`, collection `rag_demo`.

Các biến `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `EMBED_MODEL` và `GEMINI_MODEL` trong `.env` có thể ghi đè giá trị tương ứng khi chạy CLI. Khi đổi embedding model, hãy dùng collection mới và index lại để tránh trộn vector khác số chiều hoặc khác model.

## Sử dụng

1. Đặt tài liệu `.txt`, `.pdf` có text hoặc `.docx` vào thư mục `data/`.
2. Chạy tra cứu không dùng LLM:

```powershell
python main.py --no-llm --question "Sinh viên được mượn bao nhiêu quyển sách?"
```

Lệnh hiển thị model, số chiều embedding, các đoạn tìm thấy cùng điểm số, nguồn và trang (nếu có). Nếu kết quả không đạt ngưỡng, hệ thống sẽ báo không tìm thấy đủ thông tin trong tài liệu được cung cấp.

Để nhận câu trả lời từ Gemini dựa trên ngữ cảnh truy xuất, điền `GEMINI_API_KEY` hợp lệ trong `.env`, sau đó chạy:

```powershell
python main.py --question "Sinh viên được mượn bao nhiêu quyển sách?"
```

## Kiểm thử

Chạy toàn bộ kiểm thử đơn vị:

```powershell
python -m unittest discover -s tests -v
```

Các kiểm thử hiện có bao gồm cấu hình, nạp/chia tài liệu, embedding, retriever, vector store persistent, logging và khả năng tương thích của pipeline.

## Hướng nghiên cứu và cải tiến tiếp theo

Những nội dung sau **chưa được tích hợp vào prototype hiện tại**:

- OCR cho PDF scan và ảnh.
- Hybrid Search và reranking.
- Mở rộng đánh giá với bộ ground truth và quy trình đánh giá có kiểm soát.
- Phát triển giao diện hoặc API phục vụ trải nghiệm sử dụng thực tế.

## Lưu ý về dữ liệu và độ tin cậy

- Chỉ PDF có thể trích xuất văn bản mới được xử lý; PDF scan chưa được OCR.
- Tài liệu đầu vào là dữ liệu không tin cậy; không nên xem câu trả lời như nguồn thay thế tài liệu gốc.
- `data/sample.txt` là dữ liệu minh họa, không phải quy định chính thức của nhà trường.
- Không tự tạo nguồn hoặc metadata khi tài liệu không cung cấp thông tin đó.
