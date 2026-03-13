'''
Global variables.
'''

from pathlib import Path


DATA_DIR = Path("../../data/pill_dataset")

YOLO_DIR = DATA_DIR / "yolo_dataset"
YOLO_IMAGES_DIR = YOLO_DIR / "images"
YOLO_LABELS_DIR = YOLO_DIR / "labels"
YOLO_CLASSES_FILE = YOLO_DIR / "classes.txt"
YOLO_TEST_DIR = YOLO_IMAGES_DIR / "test"
YAML_FILE = YOLO_DIR / "pills.yaml"

MODEL = "yolov8n-seg.pt"
MODEL_PATH = "runs/pills/yolov8n/weights/best.pt"

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10