'''
Maps segmented pills to respective classes.
'''

import numpy as np
from pathlib import Path
import cv2
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as lsa

from confif import *

        
def get_colors(img_name):
    segmented_path = SEGMENTED_DIR / img_name
    if not segmented_path.exists():
        return [], []

    seg_img = cv2.imread(str(segmented_path))
    gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    mean_colors = []

    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue
            
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        mean_color = cv2.mean(seg_img, mask=mask)[:3]
        
        valid_contours.append(contour)
        mean_colors.append(mean_color)

    return valid_contours, mean_colors, seg_img.shape

def build_map(df, sample_size=50):    
    sample_df = df.head(sample_size).reset_index(drop=True)
    all_colors = []
    img_color_count = []

    for img_name in sample_df['image_name']:
        _, colors, _ = get_colors(img_name)
        all_colors.extend(colors)
        img_color_count.append(colors)

    kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
    kmeans.fit(all_colors)

    cluster_counts = np.zeros((sample_size, 9), dtype=int)
    for i, colors in enumerate(img_color_count):
        if len(colors) > 0:
            labels = kmeans.predict(colors)
            for label in labels:
                cluster_counts[i, label] += 1

    true_counts = sample_df[PILL_CLASSES].values

    cost_matrix = np.zeros((9, 9))
    for cluster_idx in range(9):
        for class_idx in range(9):
            diff = np.abs(cluster_counts[:, cluster_idx] - true_counts[:, class_idx])
            cost_matrix[cluster_idx, class_idx] = np.sum(diff)

    row_ind, col_ind = lsa(cost_matrix)
    
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    
    return kmeans, mapping