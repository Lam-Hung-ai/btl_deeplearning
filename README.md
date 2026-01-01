# Đề tài môn Deep Learning
## 1. Thông tin sinh viên
- Họ tên: Nguyễn Văn Lâm Hùng
- MSV: 2351250653
## 2. Hướng dẫn thực hiện lại dự án
- Tải các thư viện cần thiết
```cmd
pip install -r requirements.txt
```
- Chạy chương trình tải bộ dữ liệu VOC PASCAL
```cmd
python -m data.download
```
- Kiểm tra bộ dữ liệu đã tải hết chưa
```cmd
python check_dataset.py
```
- Huấn luyện mô hình
```cmd
python -m tools.train
```
- Chạy đánh giá
```cmd
python -m tools.infer --evaluate False --infer_samples True
```