'''
Entrypoint.
'''

from ultralytics import YOLO

from config import *
from split import split_dataset, move_files
from build_yaml import build_yaml
from train import train
from infer import infer


train_imgs, val_imgs, test_imgs = split_dataset(TRAIN_RATIO, VAL_RATIO)

move_files(train_imgs, 'train')
move_files(val_imgs, 'val')
move_files(test_imgs, 'test')

build_yaml(YOLO_DIR, YOLO_CLASSES_FILE, YAML_FILE)

train(MODEL, YAML_FILE)

infer(MODEL_PATH, str(YAML_FILE), str(YOLO_TEST_DIR))