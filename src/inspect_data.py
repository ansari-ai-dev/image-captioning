from pathlib import Path
from collections import defaultdict

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
CAPTIONS_FILE = RAW_DIR / "captions.txt"

# --- Load captions ---
image_to_captions = defaultdict(list)
malformed_lines = []

with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

header = lines[0]
print("First line (header check):", header.strip())

for i, line in enumerate(lines[1:], start=2):  # skip header
    line = line.strip()
    if not line:
        continue
    parts = line.split(",", 1)  # split on first comma only
    if len(parts) != 2:
        malformed_lines.append((i, line))
        continue
    img_name, caption = parts
    image_to_captions[img_name].append(caption)

# --- Count images on disk ---
disk_images = set(p.name for p in IMAGES_DIR.glob("*.jpg"))

# --- Stats ---
print("\n--- Dataset Summary ---")
print("Images on disk:", len(disk_images))
print("Unique images referenced in captions.txt:", len(image_to_captions))
print("Total caption lines:", sum(len(v) for v in image_to_captions.values()))
print("Malformed lines:", len(malformed_lines))

# --- Cross-check: images with no captions, captions with no image ---
captioned_images = set(image_to_captions.keys())
missing_images = captioned_images - disk_images
extra_images = disk_images - captioned_images

print("\nCaptioned but missing from disk:", len(missing_images))
if missing_images:
    print("  e.g.:", list(missing_images)[:5])

print("Images on disk with no captions:", len(extra_images))
if extra_images:
    print("  e.g.:", list(extra_images)[:5])

# --- Captions per image distribution ---
counts = [len(v) for v in image_to_captions.values()]
from collections import Counter
print("\nCaptions-per-image distribution:", Counter(counts))

# --- Duplicate caption check (same image, identical caption text twice) ---
dupes = 0
for img, caps in image_to_captions.items():
    if len(caps) != len(set(caps)):
        dupes += 1
print("\nImages with duplicate caption text:", dupes)

# --- Show a few samples ---
print("\n--- Sample entries ---")
for img_name in list(image_to_captions.keys())[:3]:
    print(f"\n{img_name}:")
    for c in image_to_captions[img_name]:
        print("  -", c)