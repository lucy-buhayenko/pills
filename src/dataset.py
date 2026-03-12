import os
import cv2
import csv
import random
import numpy as np

random.seed(49)

#configuration

BASE_FOLDER = "../data/pill_dataset"
IMAGE_FOLDER = os.path.join(BASE_FOLDER, "images")
CSV_PATH = os.path.join(BASE_FOLDER, "labels.csv")

IMAGE_SIZE = 640
TOTAL_IMAGES = 15000

MIN_PILLS = 40
MAX_PILLS = 60

MAX_RETRIES = 300

PILL_FOLDER = "../data/pill_lib"

os.makedirs(IMAGE_FOLDER, exist_ok=True)

#load pills

pill_images = []
pill_names = []

for file in os.listdir(PILL_FOLDER):
    if file.lower().endswith(".png"):
        img = cv2.imread(os.path.join(PILL_FOLDER, file), cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 4:
            pill_images.append(img)
            pill_names.append(os.path.splitext(file)[0])

NUM_PILL_TYPES = len(pill_images)

print(f"loaded {NUM_PILL_TYPES} pill types")

#generate dataset

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)

    #write header
    header = ["image_name"] + pill_names + ["total"]
    writer.writerow(header)

    img_index = 0

    #loop until enough images made
    while img_index < TOTAL_IMAGES:

        #make background
        base_color = random.randint(225, 255)
        canvas = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * base_color
        occupancy_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        #random number of pills
        target_count = random.randint(MIN_PILLS, MAX_PILLS)

        pill_type_counts = [0] * NUM_PILL_TYPES

        placed = 0
        attempts = 0

        #place pills
        while placed < target_count and attempts < target_count * MAX_RETRIES:

            #pick random pill
            pill_idx = random.randint(0, NUM_PILL_TYPES - 1)
            pill = pill_images[pill_idx]

            #random rotate and scale
            angle = random.uniform(0, 360)
            scale = random.uniform(0.6, 1.0)

            h, w = pill.shape[:2]

            #rotation matrix
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)

            cos = abs(M[0,0])
            sin = abs(M[0,1])
            new_w = int(h*sin + w*cos)
            new_h = int(h*cos + w*sin)

            #fix center
            M[0,2] += (new_w/2) - w//2
            M[1,2] += (new_h/2) - h//2

            #apply rotation
            transformed = cv2.warpAffine(
                pill, M, (new_w, new_h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0,0,0,0)
            )

            #skip if too big
            if new_w >= IMAGE_SIZE or new_h >= IMAGE_SIZE:
                attempts += 1
                continue

            #get alpha mask
            alpha = transformed[:,:,3]
            pill_mask = (alpha > 0).astype(np.uint8) * 255

            #try random positions
            for _ in range(MAX_RETRIES):

                x = random.randint(0, IMAGE_SIZE - new_w)
                y = random.randint(0, IMAGE_SIZE - new_h)

                roi = occupancy_mask[y:y+new_h, x:x+new_w]

                #check overlap
                if not np.any(cv2.bitwise_and(roi, pill_mask)):

                    #blend pill
                    roi_color = canvas[y:y+new_h, x:x+new_w]
                    alpha_norm = alpha / 255.0

                    for c in range(3):
                        roi_color[:,:,c] = (
                            alpha_norm * transformed[:,:,c] +
                            (1 - alpha_norm) * roi_color[:,:,c]
                        )

                    canvas[y:y+new_h, x:x+new_w] = roi_color

                    #update mask
                    occupancy_mask[y:y+new_h, x:x+new_w] = cv2.bitwise_or(
                        occupancy_mask[y:y+new_h, x:x+new_w], pill_mask)

                    placed += 1
                    pill_type_counts[pill_idx] += 1
                    break

            attempts += 1

        #restart if not enough pills
        if placed < MIN_PILLS:
            print(f"regenerating image {img_index}")
            continue

        #add noise
        noise = np.random.normal(0, 3, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        #save image
        filename = f"img_{img_index:05d}.png"
        cv2.imwrite(os.path.join(IMAGE_FOLDER, filename), canvas)

        #write label
        writer.writerow([filename] + pill_type_counts + [placed])

        #progress print
        if (img_index+1) % 50 == 0:
            print(f"{img_index+1}/{TOTAL_IMAGES} done")

        img_index += 1

print("done")