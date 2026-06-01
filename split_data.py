import os
import random
import shutil

# Paths
data_dir = "data/train"
val_dir = "data/val"
split_ratio = 0.2  # 20% for validation

# Create validation directory structure
os.makedirs(val_dir, exist_ok=True)

# Get all classes
classes = [d for d in os.listdir(data_dir) 
           if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]

print(f"Found classes: {classes}")

# Split each class
for cls in classes:
    # Create class directory in validation set
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    
    # Get all images in this class
    class_path = os.path.join(data_dir, cls)
    images = [f for f in os.listdir(class_path) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    print(f"\nClass '{cls}': {len(images)} images")
    
    if len(images) == 0:
        print(f"  Warning: No images found in {class_path}")
        continue
    
    # Shuffle and split
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    val_images = images[:split_idx]
    
    # Move validation images
    moved = 0
    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_dir, cls, img)
        
        # Check if file exists and is valid
        if os.path.exists(src) and os.path.getsize(src) > 0:
            shutil.move(src, dst)
            moved += 1
        else:
            print(f"  Skipping invalid file: {src}")
    
    print(f"  Moved {moved} images to validation set")
    print(f"  Remaining {len(images) - moved} images in training set")

print("\nData split complete!")
print(f"Training data: {data_dir}")
print(f"Validation data: {val_dir}")