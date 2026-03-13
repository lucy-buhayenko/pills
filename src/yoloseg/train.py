import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO


YOLO_DIR = Path("../../data/pill_dataset/yolo_dataset")
IMAGES_DIR = YOLO_DIR / "images"
LABELS_DIR = YOLO_DIR / "labels"
CLASSES_FILE = YOLO_DIR / "classes.txt"
YAML_FILE = YOLO_DIR / "dataset.yaml"

def split_dataset(split_ratio=0.8):
    for split in ['train', 'val']:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

    all_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.png'))]
    random.shuffle(all_images)
    
    split_index = int(len(all_images) * split_ratio)
    train_imgs = all_images[:split_index]
    val_imgs = all_images[split_index:]

    def move_files(file_list, split_name):
        for img_name in file_list:
            label_name = img_name.rsplit('.', 1)[0] + '.txt'
            
            src_img = IMAGES_DIR / img_name
            dst_img = IMAGES_DIR / split_name / img_name
            if src_img.exists() and not dst_img.exists():
                shutil.move(str(src_img), str(dst_img))
                
            src_label = LABELS_DIR / label_name
            dst_label = LABELS_DIR / split_name / label_name
            if src_label.exists() and not dst_label.exists():
                shutil.move(str(src_label), str(dst_label))

    move_files(train_imgs, 'train')
    move_files(val_imgs, 'val')
    print(f"Split complete: {len(train_imgs)} train, {len(val_imgs)} val.")

def create_yaml():
    """Generates the dataset.yaml file required by YOLO."""
    with open(CLASSES_FILE, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    yaml_content = {
        'path': str(YOLO_DIR.absolute()), # Absolute path is safest for YOLO
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(classes)}
    }

    with open(YAML_FILE, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    print(f"Created YOLO config at {YAML_FILE}")

def train_model():
    """Loads a pre-trained segmentation model and starts training."""
    print("Initializing YOLOv8 Segmentation Model...")
    
    # We use the 'nano' segmentation model (yolov8n-seg.pt) as a starting point. 
    # It's lightweight, fast to train, and usually more than enough for clean objects like pills.
    model = YOLO('yolov8n-seg.pt') 

    # Start training
    # epochs: How many times to pass over the 15,000 images. 
    # imgsz: YOLO resizes images to a square. 640 is standard.
    results = model.train(
        data=str(YAML_FILE),
        epochs=30, 
        imgsz=640,
        batch=16, # Adjust this down (e.g., 8) if you run out of GPU/RAM memory
        project='runs/pill_segmentation',
        name='yolov8n_run1'
    )
    print("Training complete!")

if __name__ == "__main__":
    # 1. Restructure the folders
    split_dataset()
    
    # 2. Build the config
    create_yaml()
    
    # 3. Train
    train_model()