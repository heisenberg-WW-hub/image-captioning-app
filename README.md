# 🖼️ Image Captioning - So sánh các mô hình sinh mô tả ảnh

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Ứng dụng web cho phép sinh mô tả ảnh tự động sử dụng các mô hình AI tiên tiến**

[Tính năng](#-tính-năng) •
[Demo](#-demo) •
[Cài đặt](#-cài-đặt) •
[Sử dụng](#-sử-dụng) •
[Mô hình](#-các-mô-hình)

</div>

---

## ✨ Tính năng

- 🖼️ **Tải ảnh linh hoạt**: Từ file, URL, hoặc webcam
- 🤖 **3 mô hình AI**: ViT-GPT2, BLIP-Large, Microsoft GIT
- ⚙️ **Điều chỉnh tham số**: Temperature, Top-K, Top-P, Beam Search...
- 🎛️ **Preset sẵn có**: Creative, Balanced, Precise, Custom
- 📊 **So sánh mô hình**: Chạy song song và so sánh kết quả
- 🇻🇳 **Dịch tiếng Việt**: Tích hợp Google Translate

## 🎬 Demo

<div align="center">

| Ảnh đầu vào | Mô tả được sinh |
|:-----------:|:---------------:|
| 🖼️ Ảnh của bạn | "A dog playing with a ball in the park" |

</div>

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- CUDA (khuyến nghị cho GPU acceleration)
- RAM: 8GB+ (16GB khuyến nghị)

### Bước 1: Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/image-captioning-app.git
cd image-captioning-app
```

### Bước 2: Tạo môi trường ảo

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 4: Chạy ứng dụng

```bash
streamlit run app.py
```

Truy cập `http://localhost:8501` để sử dụng.

## 📖 Sử dụng

### 1️⃣ Tải ảnh
Chọn một trong 3 cách:
- **📤 Tải ảnh lên**: Upload từ máy tính
- **🌐 Từ URL**: Nhập link ảnh
- **📷 Webcam**: Chụp trực tiếp

### 2️⃣ Chọn mô hình
Trong sidebar, chọn model và nhấn **"Tải mô hình"**.

### 3️⃣ Chọn cấu hình
| Preset | Mô tả | Khi nào dùng |
|--------|-------|--------------|
| 🎨 Creative | Đa dạng, sáng tạo | Muốn kết quả độc đáo |
| ⚖️ Balanced | Cân bằng | Mặc định, phù hợp đa số |
| 🎯 Precise | Chính xác | Cần kết quả ổn định |
| 🔧 Custom | Tùy chỉnh | Pro users |

### 4️⃣ Sinh mô tả
Nhấn **"🚀 Tạo mô tả"** và chờ kết quả!

## 🤖 Các mô hình

| Model | Kiến trúc | Ưu điểm | Nhược điểm |
|-------|-----------|---------|------------|
| **ViT-GPT2** | Vision Transformer + GPT-2 | Nhanh, nhẹ (~1-2s) | Độ chính xác trung bình |
| **BLIP-Large** | Multimodal Transformer | Chính xác cao | Tốn tài nguyên (~3-4s) |
| **GIT** | Generative Image-to-text | Kiến trúc đơn giản, hiệu quả | - |

## ⚙️ Tham số điều chỉnh

### Tham số cơ bản

| Tham số | Mô tả | Phạm vi |
|---------|-------|---------|
| `max_length` | Độ dài tối đa caption | 10 - 100 |
| `num_beams` | Số beam (beam search) | 1 - 10 |

### Tham số Sampling

| Tham số | Mô tả | Phạm vi |
|---------|-------|---------|
| `temperature` | Độ ngẫu nhiên | 0.1 - 2.0 |
| `top_k` | Số từ xem xét mỗi bước | 0 - 100 |
| `top_p` | Nucleus sampling | 0.1 - 1.0 |
| `repetition_penalty` | Phạt lặp từ | 1.0 - 2.0 |

## 📁 Cấu trúc dự án

```
├── app.py              # Ứng dụng Streamlit chính
├── utils.py            # Class quản lý các mô hình AI
├── requirements.txt    # Thư viện cần thiết
├── README.md           # Tài liệu này
├── LICENSE             # Giấy phép MIT
└── .gitignore          # Files bị bỏ qua bởi Git
```

## 🔧 Xử lý sự cố

### CUDA out of memory
```
Nhấn "Xóa cache và giải phóng bộ nhớ" trong sidebar
Hoặc chỉ load 1 model tại một thời điểm
```

### Lỗi tải model
```
Kiểm tra kết nối internet
Thử lại sau vài phút (Hugging Face có thể đang bảo trì)
```

## 🌐 Deploy lên Streamlit Cloud

1. Push code lên GitHub
2. Truy cập [share.streamlit.io](https://share.streamlit.io)
3. Chọn repository và deploy

> ⚠️ **Lưu ý**: Streamlit Cloud miễn phí có giới hạn RAM (~1GB). Nếu gặp lỗi, hãy thử [Hugging Face Spaces](https://huggingface.co/spaces).

## 📄 License

Dự án được phát hành theo giấy phép [MIT License](LICENSE).

## 👤 Tác giả

**Trần Anh Tùng - 20227164**

**HUST-FaMI**

**Đồ án 2 - 2024**

---

<div align="center">

⭐ **Nếu dự án hữu ích, hãy cho một Star!** ⭐

</div>
