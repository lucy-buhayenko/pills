'''
Inference of YOLOv8-seg model on test set.
'''

from ultralytics import YOLO

from config import *


def infer(model_path, yaml_file, test_images_dir):
    model = YOLO(model_path)
    
    metrics = model.val(data=yaml_file, split='test')
    
    print(f"mAP50-95: {metrics.seg.map:.4f}")
    
    model.predict(
        source=test_images_dir,
        conf=0.5,
        save=True,
        project='runs/pills',
        name='test_preds'
    )
    
if __name__ == "__main__":
    infer(MODEL_PATH, YAML_FILE, YOLO_TEST_DIR)