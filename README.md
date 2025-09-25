# Digital_Hunman

### CÁC BƯỚC:

Sao chép kho lưu trữ

```bash
git clone
```
### BƯỚC 01 - Tạo môi trường conda sau khi mở kho lưu trữ

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

### BƯỚC 02 - Cài đặt các yêu cầu
```bash
pip install -r requirements.txt
```

### Tạo tệp `.env` trong thư mục gốc và thêm Pinecone của bạn

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"

```

```bash
# Chạy lệnh sau để lưu trữ các nhúng vào pinecone
python store_index.py
```

```bash
# Cuối cùng chạy lệnh sau
python app.py
```

```bash
mở localhost:
```

### Techstack đã sử dụng:

- Python
- LangChain
- Flask
- Pinecone