# Hướng dẫn đưa project lên GitHub (Windows PowerShell)

1. Giải nén `ro-ai-rag-assistant.zip` vào `E:\NewTech\ro-ai-rag-assistant`.
2. Vào https://github.com/new , đặt tên `ro-ai-rag-assistant`, chọn **Private** nếu báo cáo/tài liệu không được phép công khai, **không** chọn tạo README/.gitignore/license trên GitHub. Nhấn **Create repository**.
3. Mở Terminal ở thư mục dự án và chạy các lệnh sau, thay `USERNAME` bằng GitHub username.

```powershell
cd E:\NewTech\ro-ai-rag-assistant
git init
git branch -M main
git add .
git status
git commit -m "docs: initialize Ro AI RAG research project"
git remote add origin https://github.com/USERNAME/ro-ai-rag-assistant.git
git push -u origin main
```

Trước khi `git add .`, kiểm tra tài liệu có chứa thông tin cá nhân hoặc tài liệu không được phép chia sẻ và lựa chọn quyền truy cập repo phù hợp. Không đưa `.env` hay API key thật lên GitHub. Với repository đã có commit, không được chạy push cưỡng bức (`--force`); nên clone repo đó trước hoặc đồng bộ lịch sử.

## Các lần cập nhật sau

```powershell
git add .
git commit -m "experiment: add chunking comparison results"
git push
```

Chỉ commit nội dung bạn thực sự viết/chạy và kết quả đã kiểm chứng.
