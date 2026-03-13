'''
Splits data into train/val/test.
'''

import os
import random
import shutil

from config import *

random.seed(49)


def split_dataset(train_ratio, val_ratio):
    for split in ['train', 'val', 'test']:
        (YOLO_IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (YOLO_LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

    all_images = [f for f in os.listdir(YOLO_IMAGES_DIR) if f.endswith(('.png'))]
    random.shuffle(all_images)
    
    total = len(all_images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_imgs = all_images[:train_end]
    val_imgs = all_images[train_end:val_end]
    test_imgs = all_images[val_end:]
    
    return train_imgs, val_imgs, test_imgs
    
def move_files(file_list, split_name):
    for img_name in file_list:
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        
        src_img = YOLO_IMAGES_DIR / img_name
        dst_img = YOLO_IMAGES_DIR / split_name / img_name
        if src_img.exists() and not dst_img.exists():
            shutil.move(str(src_img), str(dst_img))
            
        src_label = YOLO_LABELS_DIR / label_name
        dst_label = YOLO_LABELS_DIR / split_name / label_name
        if src_label.exists() and not dst_label.exists():
            shutil.move(str(src_label), str(dst_label))

if __name__ == "__main__":
    train_imgs, val_imgs, test_imgs = split_dataset(TRAIN_RATIO, VAL_RATIO)
    
    move_files(train_imgs, 'train')
    move_files(val_imgs, 'val')
    move_files(test_imgs, 'test')