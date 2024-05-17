# Simple reference: https://medium.com/@achyutpaudel50/yolov8-train-and-inference-detection-or-segmentation-on-custom-data-using-roboflow-481c8d27445d
from ultralytics import YOLO
import os; os.chdir('src_app_host_model')

# model_file_dir = '../roboflow-yolo-experiments/models'
#model_file_name = 'yolov8n-cls.pt'
#model_file_name = 'yolov8n.pt'
# model_file_name = 'yolov8n-cls_original.pt'
model_file_name = 'best.pt'
# model_file_path = f'{model_file_dir}/{model_file_name}'


model = YOLO(model_file_name)
image_path = 'tmp/wcc_raw_image_data_prod_2024-May-09_street_works_and_obstructions_2216806_0.jpeg'
results = model(image_path, conf = 0.5, iou = 0.6)
r = results[0]

if r.probs.top1conf > 0.5:
    inference_category_str = r.names[r.probs.top1]
else:
    inference_category_str = 'Unknown (Low confidence)'
