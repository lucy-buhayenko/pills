'''
Trains YOLOv8 segmentation model for pill detection.
'''

from ultralytics import YOLO

from config import *


def train(model_path, yaml_file):
    model = YOLO(model_path) 

    results = model.train(
        data=str(yaml_file),
        epochs=EPOCHS, 
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=RESULTS_DIR,
        name=MODEL_NAME
    )

if __name__ == "__main__":
    train(MODEL, YAML_FILE)