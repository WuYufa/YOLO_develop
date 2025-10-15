import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import cv2

def save_detections_as_txt(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, result in enumerate(results):
        img_name = os.path.splitext(os.path.basename(result.path))[0]
        txt_path = os.path.join(output_dir, f"{img_name}.txt")
        
        with open(txt_path, 'w') as f:
            for box in result.boxes.cpu().numpy(): 
                cls = int(box.cls[0])  
                x_center, y_center, width, height = box.xywh[0]  
                img_width, img_height = result.orig_shape[1], result.orig_shape[0]
                x_center /= img_width
                y_center /= img_height
                width /= img_width
                height /= img_height
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == '__main__':
    model = YOLO('dataset/best_v8zhangaiwu.pt')
    results = model.predict(
        source='dataset/test',
        device='cpu',
        project='runs/detect',
        name='exp',
        save=True,       
        conf=0.5,
        save_txt=False  # 禁用默认 TXT 保存（采用自定义保存逻辑）
    )
    save_detections_as_txt(results, 'runs/detect/exp/labels')
    print("检测结果已保存到:runs/detect/exp/labels")