'''
Entrypoint.
'''

from config import *
from dataset import gen_dataset
from split import split_dataset, move_files
from build_yaml import build_yaml
from train import train
from infer import infer

print("Generating YOLOv8 segmentation dataset...")
YOLO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
YOLO_LABELS_DIR.mkdir(parents=True, exist_ok=True)

with open(YOLO_DIR / "classes.txt", "w") as f:
    f.write("\n".join(PILL_CLASSES))
df = pd.read_csv(LABELS_CSV)

kmeans, mapping = build_map(df, sample_size=50)

gen_dataset(df, kmeans, mapping)

print("Splitting dataset...")
train_imgs, val_imgs, test_imgs = split_dataset(TRAIN_RATIO, VAL_RATIO)

move_files(train_imgs, 'train')
move_files(val_imgs, 'val')
move_files(test_imgs, 'test')

print("Building YAML file...")
build_yaml(YOLO_DIR, YOLO_CLASSES_FILE, YAML_FILE)

print("Training...")
train(MODEL, YAML_FILE)

print("Infering...")
infer(MODEL_PATH, str(YAML_FILE), str(YOLO_TEST_DIR))

print(f"Results saved to {RESULTS_DIR}/{MODEL_NAME}.")