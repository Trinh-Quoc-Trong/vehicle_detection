import torch
from ultralytics import YOLO
import yaml
import os
import logging

# Cấu hình logging cơ bản
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    """Hàm chính để huấn luyện mô hình YOLOv8."""

    # 1. Kiểm tra và thiết lập thiết bị (CUDA hoặc CPU)
    try:
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_count = torch.cuda.device_count()
            logging.info(f"Phát hiện {gpu_count} GPU CUDA.")
            for i in range(gpu_count):
                logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            # YOLO tự động chọn GPU tốt nhất hoặc sử dụng nhiều GPU nếu được cấu hình
            # device = 0 # Hoặc chỉ định cụ thể GPU ID nếu muốn
        else:
            device = 'cpu'
            logging.info("Không tìm thấy GPU CUDA, sử dụng CPU.")
    except Exception as e:
        logging.error(f"Lỗi khi kiểm tra CUDA: {e}. Sử dụng CPU.")
        device = 'cpu'

    # 2. Đường dẫn đến tệp cấu hình dữ liệu
    # Sử dụng đường dẫn tương đối từ gốc workspace
    data_config_path = 'trafic_data/data_1.yaml'
    logging.info(f"Đường dẫn tệp cấu hình dữ liệu: {data_config_path}")

    # Kiểm tra sự tồn tại của file cấu hình
    if not os.path.exists(data_config_path):
        logging.error(f"Lỗi: Không tìm thấy tệp cấu hình dữ liệu tại: {data_config_path}")
        return

    # Đảm bảo đường dẫn trong file YAML là tuyệt đối hoặc đúng tương đối
    # YOLO sẽ xử lý đường dẫn tương đối so với vị trí file YAML
    try:
        with open(data_config_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
            logging.info(f"Đã đọc thành công tệp YAML: {data_config_path}")
            # Kiểm tra các đường dẫn trong YAML (tùy chọn nhưng hữu ích)
            for key in ['train', 'val']:
                if key in data_yaml:
                    path_in_yaml = data_yaml[key]
                    # YOLO xử lý đường dẫn tương đối từ thư mục chứa file yaml
                    # Hoặc bạn có thể chuyển thành đường dẫn tuyệt đối nếu cần
                    # absolute_path = os.path.abspath(os.path.join(os.path.dirname(data_config_path), path_in_yaml))
                    # logging.info(f"  Đường dẫn '{key}': {path_in_yaml} (Resolved: {absolute_path})")
                    # if not os.path.exists(absolute_path):
                    #     logging.warning(f"  Cảnh báo: Đường dẫn '{key}' trong YAML không tồn tại: {absolute_path}")
                    logging.info(f"  Đường dẫn '{key}' trong YAML: {path_in_yaml}")
                else:
                    logging.warning(f"  Cảnh báo: Thiếu khóa '{key}' trong tệp YAML.")

    except FileNotFoundError:
        logging.error(f"Lỗi: Không thể mở tệp cấu hình YAML tại: {data_config_path}")
        return
    except yaml.YAMLError as e:
        logging.error(f"Lỗi khi đọc tệp YAML: {e}")
        return
    except Exception as e:
        logging.error(f"Lỗi không xác định khi xử lý tệp YAML: {e}")
        return


    # 3. Chọn kiến trúc YOLOv8 (ví dụ: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
    # 'n' (nano) nhanh nhất, nhẹ nhất; 'x' (extra large) chính xác nhất, nặng nhất
    model_variant = 'yolov8s' # Bắt đầu với 's' (small) là một lựa chọn cân bằng
    model_weights_file = f'{model_variant}.pt' # File này chứa cả kiến trúc và trọng số pre-trained (nếu có)

    logging.info(f"Khởi tạo kiến trúc mô hình từ: {model_weights_file} (Sẽ không sử dụng trọng số pre-trained)")
    # Khởi tạo mô hình YOLO từ file .pt (chỉ lấy kiến trúc)
    try:
        model = YOLO(model_weights_file)
    except Exception as e:
        logging.error(f"Lỗi khi khởi tạo mô hình YOLO từ '{model_weights_file}': {e}")
        logging.error("Hãy đảm bảo bạn đã cài đặt 'ultralytics' và tệp trọng số tồn tại (nó sẽ tự động tải về nếu chưa có).")
        return

    # 4. Cấu hình huấn luyện
    epochs = 15       # Tăng số lượng epochs vì huấn luyện từ đầu
    batch_size = 32  # Giảm nếu gặp lỗi CUDA out of memory, tăng nếu còn dư VRAM
    img_size = 640   # Kích thước ảnh đầu vào tiêu chuẩn cho YOLOv8
    project_name = 'Vehicle_Detection_From_Scratch'
    exp_name = f'{model_variant}_scratch_{epochs}epochs'
    num_workers = 8 # Số process để tải dữ liệu (điều chỉnh theo CPU và RAM)

    logging.info("Bắt đầu quá trình huấn luyện...")
    logging.info(f"  Model: {model_variant}")
    logging.info(f"  Epochs: {epochs}")
    logging.info(f"  Batch Size: {batch_size}")
    logging.info(f"  Image Size: {img_size}")
    logging.info(f"  Device: {device}")
    logging.info(f"  Project: {project_name}")
    logging.info(f"  Experiment Name: {exp_name}")
    logging.info(f"  Data Config: {data_config_path}")
    logging.info(f"  Training from scratch: True")
    logging.info(f"  Augmentations: Enabled (Mosaic, MixUp, CopyPaste, etc.)")


    try:
        results = model.train(
            data=data_config_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,       # Sử dụng thiết bị đã xác định (CUDA hoặc CPU)
            project=project_name,
            name=exp_name,
            exist_ok=True,      # Ghi đè lên lần chạy trước nếu tên thí nghiệm trùng
            pretrained=False,    # *** QUAN TRỌNG: Huấn luyện từ đầu ***
            optimizer='AdamW',   # Bộ tối ưu hóa hiện đại
            lr0=0.005,          # Learning rate ban đầu (có thể cần tinh chỉnh)
            lrf=0.01,           # Learning rate cuối cùng = lr0 * lrf
            workers=num_workers,
            # Bật các augmentation mạnh mẽ tích hợp sẵn:
            augment=True,       # Bật tất cả các augmentation mặc định
            mosaic=1.0,         # Xác suất áp dụng Mosaic (0.0 = tắt, 1.0 = luôn bật nếu có thể)
            mixup=0.1,          # Xác suất áp dụng MixUp (0.0 = tắt)
            copy_paste=0.1,     # Xác suất áp dụng CopyPaste (0.0 = tắt)
            # Các tham số augmentation khác có thể tinh chỉnh (xem docs của YOLOv8):
            # degrees=10.0, translate=0.1, scale=0.5, shear=2.0, perspective=0.0,
            # flipud=0.0, fliplr=0.5,
            # hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        )
        logging.info("Quá trình huấn luyện hoàn tất.")
        logging.info(f"Kết quả và mô hình đã được lưu tại: {results.save_dir}")

        # 5. Đánh giá mô hình trên tập validation (tùy chọn)
        logging.info("Bắt đầu đánh giá mô hình trên tập validation...")
        # Tìm đường dẫn đến trọng số tốt nhất được lưu trong quá trình huấn luyện
        best_weights_path = os.path.join(results.save_dir, 'weights', 'best.pt')

        if os.path.exists(best_weights_path):
            logging.info(f"Tải trọng số tốt nhất từ: {best_weights_path}")
            # Tải mô hình với trọng số tốt nhất
            model_trained = YOLO(best_weights_path)

            metrics = model_trained.val(
                data=data_config_path,
                imgsz=img_size,
                batch=batch_size, # Có thể dùng batch size lớn hơn khi đánh giá nếu VRAM cho phép
                device=device,
                project=project_name,
                name=f'{exp_name}_validation', # Lưu kết quả đánh giá vào thư mục riêng
                exist_ok=True,
                split='val' # Chỉ định đánh giá trên tập 'val'
            )
            logging.info("Đánh giá hoàn tất.")
            # In ra các chỉ số quan trọng
            logging.info("Kết quả đánh giá (Validation Metrics):")
            # Các chỉ số chính thường nằm trong metrics.box object
            map50_95 = metrics.box.map    # mAP50-95
            map50 = metrics.box.map50    # mAP50
            logging.info(f"  mAP50-95: {map50_95:.4f}")
            logging.info(f"  mAP50:    {map50:.4f}")
            # Xem thêm các chỉ số khác trong metrics.results_dict hoặc metrics.box
            # logging.info(f"  Chi tiết: {metrics.results_dict}")

        else:
            logging.warning(f"Không tìm thấy tệp trọng số tốt nhất tại: {best_weights_path}. Bỏ qua bước đánh giá.")

    except Exception as e:
        logging.error(f"Đã xảy ra lỗi trong quá trình huấn luyện hoặc đánh giá: {e}", exc_info=True) # In traceback

if __name__ == '__main__':
    logging.info("Chạy script huấn luyện...")
    train_model()
    logging.info("Script huấn luyện đã kết thúc.") 