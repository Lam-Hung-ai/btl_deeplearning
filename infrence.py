import torch
import cv2
import yaml
import numpy as np
from model.detr import DETR # Đảm bảo file model/detr.py nằm đúng vị trí
import torchvision.transforms.v2 as transforms
from torchvision.io import read_image

# 1. Cấu hình thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

def load_inference_model(config_path, checkpoint_path):
    # Đọc file config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model_params']
    dataset_config = config['dataset_params']
    
    # Khởi tạo model
    model = DETR(
        config=model_config,
        num_classes=dataset_config['num_classes'],
        bg_class_idx=dataset_config['bg_class_idx']
    )
    
    # Load trọng số (weights)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Danh sách nhãn (Labels) - Theo thứ tự của VOCDataset trong code bạn
    classes = [
        'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
        'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
    ]
    idx2label = {i + 1: label for i, label in enumerate(sorted(classes))}
    idx2label[0] = 'background'
    
    return model, config, idx2label

def inference_single_image(model, config, idx2label, image_path, output_path='result.jpg'):
    # 2. Tiền xử lý ảnh (giống hệt lúc Test)
    im_size = config['dataset_params']['im_size']
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Đọc ảnh gốc bằng OpenCV để vẽ
    original_img = cv2.imread(image_path)
    h, w = original_img.shape[:2]
    
    # Transform ảnh cho model
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(size=(im_size, im_size)),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Chuyển đổi định dạng ảnh
    input_tensor = read_image(image_path)
    input_tensor = transform(input_tensor).unsqueeze(0).to(device)
    
    # 3. Chạy Model
    with torch.no_grad():
        output = model(
            input_tensor, 
            score_thresh=0.5, # Bạn có thể điều chỉnh ngưỡng này
            use_nms=True
        )
    
    detections = output['detections'][0]
    boxes = detections['boxes']
    labels = detections['labels']
    scores = detections['scores']
    
    # 4. Vẽ kết quả
    overlay = original_img.copy()
    for i in range(len(boxes)):
        # Box từ model đang ở hệ [0, 1], cần scale lại theo kích thước ảnh gốc
        x1, y1, x2, y2 = boxes[i].cpu().numpy()
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        
        label_idx = labels[i].item()
        label_name = idx2label.get(label_idx, "Unknown")
        score = scores[i].item()
        
        # Vẽ hình chữ nhật
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Vẽ nhãn và background cho nhãn
        text = f"{label_name}: {score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(original_img, (x1, y1 - 20), (x1 + text_w, y1), (0, 0, 255), -1)
        cv2.putText(original_img, text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    # Tạo hiệu ứng trong suốt cho box (tùy chọn)
    cv2.addWeighted(overlay, 0.3, original_img, 0.7, 0, original_img)
    
    # Lưu ảnh
    cv2.imwrite(output_path, original_img)
    print(f"Đã lưu kết quả tại: {output_path}")

if __name__ == '__main__':
    # THAY ĐỔI ĐƯỜNG DẪN DƯỚI ĐÂY
    CONFIG_FILE = 'config/voc.yaml'
    CHECKPOINT = 'detr_34.pth' # Đường dẫn file .pth của bạn
    IMAGE_INPUT = 'images/2008_000011.jpg' # File ảnh bạn muốn test
    
    # Thực hiện
    model, config, idx2label = load_inference_model(CONFIG_FILE, CHECKPOINT)
    inference_single_image(model, config, idx2label, IMAGE_INPUT)