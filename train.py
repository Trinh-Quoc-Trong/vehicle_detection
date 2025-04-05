from ultralytics import YOLO

# Tải mô hình YOLOv8n đã được huấn luyện trước trên COCO
# 'yolov8n.pt' chứa cả kiến trúc và trọng số đã huấn luyện
# Nếu bạn thực sự muốn huấn luyện từ đầu (không khuyến khích), hãy sử dụng:
# model = YOLO('yolov8n.yaml') # Chỉ tải kiến trúc
# model = YOLO('yolov8n.pt') # Sử dụng mô hình nano
model = YOLO('yolov8s.pt') # Sử dụng mô hình small

# Đường dẫn đến tệp cấu hình dữ liệu (tương đối từ vị trí chạy script)
data_config_path = 'trafic_data/data_1.yaml'

# Bọc mã chính trong if __name__ == '__main__'
if __name__ == '__main__':
    # Huấn luyện mô hình
    results = model.train(
        data=data_config_path,
        epochs=10,          # Số lượng epochs (có thể điều chỉnh)
        imgsz=640,           # Kích thước ảnh đầu vào
        batch=32,            # Kích thước batch (thử nghiệm cho yolov8s trên 12GB VRAM)
        device=0,            # Sử dụng GPU đầu tiên (CUDA device 0). Nếu không có GPU, dùng 'cpu'
        project='runs/train',# Thư mục lưu kết quả
        name='traffic_detect_yolov8s', # Cập nhật tên lần chạy
        exist_ok=True        # Cho phép ghi đè lên thư mục kết quả nếu đã tồn tại
    )

    print("Hoàn tất huấn luyện!")
    print(f"Kết quả được lưu tại: {results.save_dir}") 