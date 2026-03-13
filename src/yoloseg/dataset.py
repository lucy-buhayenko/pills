'''
Generates YOLOv8 segmentation labels.
'''

import cv2
import shutil
import pandas as pd
from sklearn.cluster import KMeans

from map import get_colors, build_map
from config import *


def gen_dataset(df, kmeans, mapping):
    for i, row in df.iterrows():
        if (i+1) % 100 == 0:
            print(f"Processing {i+1}/{len(df)}")

        img_name = row['image_name']
        contours, colors, shape = get_colors(img_name)
        
        if not contours:
            continue
            
        h, w = shape[:2]
        yolo_labels = []
        
        cluster_labels = kmeans.predict(colors)
        
        for contour, cluster_label in zip(contours, cluster_labels):
            class_idx = mapping[cluster_label]
            
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            polygon = []
            for point in approx:
                x_norm = point[0][0] / w
                y_norm = point[0][1] / h
                polygon.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
                
            yolo_labels.append(f"{class_idx} " + " ".join(polygon))
            
        label_filename = img_name.rsplit('.', 1)[0] + '.txt'
        with open(YOLO_LABELS_DIR / label_filename, 'w') as f:
            f.write("\n".join(yolo_labels))
            
        orig_img_path = IMAGES_DIR / img_name
        dest_img_path = YOLO_IMAGES_DIR / img_name
        if orig_img_path.exists() and not dest_img_path.exists():
            shutil.copy(orig_img_path, dest_img_path)

if __name__ == "__main__":
    YOLO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(YOLO_DIR / "classes.txt", "w") as f:
        f.write("\n".join(PILL_CLASSES))
        
    df = pd.read_csv(LABELS_CSV)
    
    kmeans, mapping = build_map(df, sample_size=50)
    
    gen_dataset(df, kmeans, mapping)