'''
Generates YAML for YOLO dataset.
'''

import yaml
from pathlib import Path
from config import *

def build_yaml(yolo_dir, classes_file, yaml_file):
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    yaml_content = {
        'path': str(Path(yolo_dir).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(classes)}
    }

    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        

if __name__ == "__main__":
    build_yaml(YOLO_DIR, CLASSES_FILE, YAML_FILE)