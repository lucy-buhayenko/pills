'''
Segments pills.
'''

from skimage import io, color, filters, morphology, measure
import os
from skimage.filters import gaussian, median
import warnings
warnings.filterwarnings('ignore')


def segment_pills(image, sigma=1, min_size=150, hole_size=50):
    gray = color.rgb2gray(image)
    smooth = gaussian(gray, sigma=sigma)
    smooth = median(smooth)

    threshold = filters.threshold_otsu(smooth)

    mask = smooth < threshold

    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.remove_small_holes(mask, area_threshold=hole_size)
    
    labels = measure.label(mask)

    img_segmented = image.copy()
    img_segmented[~mask] = 0

    return img_segmented, mask, labels


if __name__ == "__main__":
    IMG_DIR = "../data/pill_dataset/images/"
    SEG_DIR = "../data/pill_dataset/segmented/"
    os.makedirs(SEG_DIR, exist_ok=True)
    
    for i, img_file in enumerate(os.listdir(IMG_DIR)):
        if img_file.lower().endswith((".png")):
            img_path = os.path.join(IMG_DIR, img_file)
            image = io.imread(img_path)
            segmented, mask, labels = segment_pills(image)

            seg_path = os.path.join(SEG_DIR, img_file)
            io.imsave(seg_path, segmented)
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{len(os.listdir(IMG_DIR))}")