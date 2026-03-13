'''
Trains YOLOv8 segmentation model for pill detection.
'''

from ultralytics import YOLO

from config import *


def train(model_path, yaml_file):
    model = YOLO(model_path) 

    results = model.train(
        data=str(yaml_file),
        epochs=30, 
        imgsz=640,
        batch=16,
        project='runs/pills',
        name='yolov8n'
    )

if __name__ == "__main__":
    train(MODEL, YAML_FILE)