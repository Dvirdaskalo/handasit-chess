import os
import random

DATA_DIR = "data/train"
SEED = 42

random.seed(SEED)

# Collect files per class
class_files = {}
for cls in os.listdir(DATA_DIR):
    cls_dir = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(cls_dir):
        continue
    files = [
        f for f in os.listdir(cls_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    class_files[cls] = files

# Show counts
print("Before balancing:")
for cls, files in class_files.items():
    print(f"{cls}: {len(files)}")

# Find minimum class size
min_count = min(len(files) for files in class_files.values())
print(f"\nBalancing to {min_count} samples per class\n")

# Delete extras
for cls, files in class_files.items():
    cls_dir = os.path.join(DATA_DIR, cls)

    if len(files) > min_count:
        remove = random.sample(files, len(files) - min_count)
        for f in remove:
            os.remove(os.path.join(cls_dir, f))

# Final counts
print("After balancing:")
for cls in os.listdir(DATA_DIR):
    cls_dir = os.path.join(DATA_DIR, cls)
    if os.path.isdir(cls_dir):
        print(cls, "→", len(os.listdir(cls_dir)))

print("\n✅ Dataset balanced.")
