import torch
from torchvision import models, transforms
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load model and class info
with open("class_mapping.json") as f:
    class_data = json.load(f)
labels = class_data["classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
model.load_state_dict(torch.load("model_best.pth", map_location=device))
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print(f"Model loaded with {len(labels)} classes: {labels}")

# 1. Check training data distribution
print("\n1. Training Data Distribution:")
train_dir = "data/train"
for cls in labels:
    cls_dir = os.path.join(train_dir, cls)
    if os.path.exists(cls_dir):
        count = len([f for f in os.listdir(cls_dir) if f.endswith('.png')])
        print(f"  {cls}: {count} images")
    else:
        print(f"  {cls}: Directory not found!")

# 2. Test model on sample images
print("\n2. Testing model on sample images...")
test_dir = "test_samples"  # Create this directory with test images
if os.path.exists(test_dir):
    for img_file in os.listdir(test_dir):
        if img_file.endswith('.png'):
            img_path = os.path.join(test_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                probs = probabilities[0].cpu().numpy()
                
                pred_idx = probs.argmax()
                confidence = probs[pred_idx]
                
                print(f"\n  {img_file}:")
                print(f"    Prediction: {labels[pred_idx]} ({confidence:.3f})")
                print(f"    Top 3 predictions:")
                top_indices = np.argsort(probs)[-3:][::-1]
                for idx in top_indices:
                    print(f"      {labels[idx]}: {probs[idx]:.3f}")

# 3. Create a simple test to see if model can distinguish pieces
print("\n3. Testing with synthetic squares...")
# Create colored squares to see what the model thinks
test_images = {
    'white_rook': np.ones((100, 100, 3), dtype=np.uint8) * 255,
    'white_pawn': np.ones((100, 100, 3), dtype=np.uint8) * 200,
    'black_square': np.zeros((100, 100, 3), dtype=np.uint8),
}

for name, img_array in test_images.items():
    img = Image.fromarray(img_array)
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        probs = probabilities[0].cpu().numpy()
        
        pred_idx = probs.argmax()
        print(f"\n  {name}:")
        print(f"    Predicted as: {labels[pred_idx]} ({probs[pred_idx]:.3f})")
        
        # Show probabilities for all classes
        for i, label in enumerate(labels):
            if probs[i] > 0.1:
                print(f"      {label}: {probs[i]:.3f}")