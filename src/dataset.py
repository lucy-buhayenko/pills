import os #for folder and file
import cv2 #image processing
import csv
import random
import numpy as np

#configuration

#main dataset folder
BASE_FOLDER = "pill_dataset"

#folder for generated images
IMAGE_FOLDER = os.path.join(BASE_FOLDER, "images")

#csv file for labels
CSV_PATH = os.path.join(BASE_FOLDER, "labels.csv")


IMAGE_SIZE = 640
TOTAL_IMAGES = 10000

#min and max pills per image
MIN_PILLS = 40
MAX_PILLS = 60

#max placement tries
MAX_RETRIES = 500

#folder with pill png file
PILL_FOLDER = "Projects/LFC/CSCI-450/pill_library"

#create image folder if needed
os.makedirs(IMAGE_FOLDER, exist_ok=True)

#load pill images
pill_images = []

for file in os.listdir(PILL_FOLDER):
    if file.lower().endswith(".png"):
        img = cv2.imread(os.path.join(PILL_FOLDER, file), cv2.IMREAD_UNCHANGED)
        pill_images.append(img)

print(f"loaded {len(pill_images)} pills")

#generate dataset

with open(CSV_PATH, "w", newline="") as f:

    writer = csv.writer(f)

    #write header
    writer.writerow(["image_name", "pill_count"])

    img_index = 0

    #loop until enough images made
    while img_index < TOTAL_IMAGES:

        #make background
        base_color = random.randint(225, 255)
        canvas = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * base_color

        #mask to stop overlap
        occupancy_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        #random number of pills
        target_count = random.randint(MIN_PILLS, MAX_PILLS)

        placed = 0
        attempts = 0

        #place pills
        while placed < target_count and attempts < target_count * MAX_RETRIES:

            #pick random pill
            pill = random.choice(pill_images)

            #random rotate and scale
            angle = random.uniform(0, 360)
            scale = random.uniform(0.9, 1.1)

            h, w = pill.shape[:2]

            #rotation matrix
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)

            #new size after rotate
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

            #get alpha mask
            alpha = transformed[:,:,3]
            pill_mask = (alpha > 0).astype(np.uint8) * 255 #255 where pill exists, 0 where no pill to check overlap

            #skip if too big
            if new_w >= IMAGE_SIZE or new_h >= IMAGE_SIZE:
                attempts += 1
                continue

            #try random positions
            for _ in range(MAX_RETRIES):

                x = random.randint(0, IMAGE_SIZE - new_w)
                y = random.randint(0, IMAGE_SIZE - new_h)

                roi = occupancy_mask[y:y+new_h, x:x+new_w] #region of interst

                #check overlap
                if not np.any(cv2.bitwise_and(roi, pill_mask)): #if no overlap, then place 

                    #blend pill
                    roi_color = canvas[y:y+new_h, x:x+new_w] #blends pill with background, otherwise edges look harsh
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
                    break

            attempts += 1

        #restart if not enough pills
        if placed < MIN_PILLS:
            print(f"regenerating image {img_index}")
            continue

        #add noise
        #mean = 0; standard deviation = 3; shape = same shape as image
        noise = np.random.normal(0, 3, canvas.shape).astype(np.int16) #A normal distribution is the same thing as a Gaussian distribution.
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        #save image
        filename = f"img_{img_index:05d}.png"
        cv2.imwrite(os.path.join(IMAGE_FOLDER, filename), canvas)

        #write label
        writer.writerow([filename, placed])

        #progress print for every 50 images
        if (img_index+1) % 50 == 0:
            print(f"{img_index+1}/{TOTAL_IMAGES} done")

        img_index += 1

print("done")