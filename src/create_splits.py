import csv
import random
from pathlib import Path
from collections import defaultdict

SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# test gets the remainder

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CAPTIONS_FILE = RAW_DIR / "captions.txt"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- Load captions grouped by image ---
image_to_captions = defaultdict(list)
with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    next(f)  # skip header
    for line in f:
        line = line.strip()
        if not line:
            continue
        img_name, caption = line.split(",", 1)
        image_to_captions[img_name].append(caption)

all_images = sorted(image_to_captions.keys())  # sorted first for determinism
random.seed(SEED)
random.shuffle(all_images)

n = len(all_images)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_images = all_images[:n_train]
val_images = all_images[n_train:n_train + n_val]
test_images = all_images[n_train + n_val:]

print(f"Total images: {n}")
print(f"Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}")

def write_split(filename, image_list):
    path = PROCESSED_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "caption"])
        for img in image_list:
            for cap in image_to_captions[img]:
                writer.writerow([img, cap])
    print(f"Wrote {filename}: {len(image_list)} images, "
          f"{sum(len(image_to_captions[i]) for i in image_list)} caption rows")

write_split("train.csv", train_images)
write_split("val.csv", val_images)
write_split("test.csv", test_images)

# Save just the image lists too — useful for quick membership checks later
for name, imgs in [("train", train_images), ("val", val_images), ("test", test_images)]:
    with open(PROCESSED_DIR / f"{name}_images.txt", "w") as f:
        f.write("\n".join(imgs))

print("\nSeed used:", SEED)
print("Done. Split files saved to data/processed/")