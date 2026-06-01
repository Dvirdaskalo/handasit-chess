import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ColorJitter(
        brightness=0.25,  # Increased
        contrast=0.25,    # Increased
        saturation=0.25,  # Increased
        hue=0.1          # Increased
    ),
    transforms.RandomRotation(15),  # Increased
    transforms.RandomAffine(
        degrees=10,        # Increased
        translate=(0.1, 0.1),  # Increased
        scale=(0.9, 1.1)   # Increased range
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# Validation transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Loading datasets...")

# Check if validation directory exists and has data
train_dir = "data/train"
val_dir = "data/val"

if not os.path.exists(val_dir) or not os.listdir(val_dir):
    print("Warning: Validation directory is empty or doesn't exist.")
    print("Creating validation set from training data (80/20 split)...")
    
    # Create a combined dataset and split it
    from torch.utils.data import random_split
    
    # Load all data
    full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Split the dataset
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    print(f"Split dataset: {train_size} training, {val_size} validation samples")
    
else:
    # Load separate train and validation datasets
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

# Check class distribution
print(f"\nClasses: {train_ds.classes}")
print(f"Class to index: {train_ds.class_to_idx}")

# Count samples per class
class_counts = {}
for cls_idx in range(len(train_ds.classes)):
    if hasattr(train_ds, 'targets'):
        count = sum(1 for target in train_ds.targets if target == cls_idx)
    else:
        # For Subset objects from random_split
        count = sum(1 for _, label in train_ds if label == cls_idx)
    class_counts[train_ds.classes[cls_idx]] = count

print("\nTraining class distribution:")
for cls, count in class_counts.items():
    print(f"  {cls}: {count}")

# Save class mapping for inference
class_mapping = {
    "classes": train_ds.classes,
    "class_to_idx": train_ds.class_to_idx
}
with open("class_mapping.json", "w") as f:
    json.dump(class_mapping, f, indent=2)
print("\nSaved class mapping to class_mapping.json")

# Create data loaders
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

# Initialize model
print("\nInitializing model...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
model.to(DEVICE)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5)

# Training loop
print("\nStarting training...")
best_val_acc = 0

for epoch in range(30):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (x, y) in enumerate(train_dl):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        
        train_loss += loss.item()
        _, predicted = out.max(1)
        train_total += y.size(0)
        train_correct += predicted.eq(y).sum().item()
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_dl)}: Loss={loss.item():.4f}")
    
    train_acc = 100. * train_correct / train_total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            val_loss += loss.item()
            
            _, predicted = out.max(1)
            val_total += y.size(0)
            val_correct += predicted.eq(y).sum().item()
    
    val_acc = 100. * val_correct / val_total
    val_loss_avg = val_loss / len(val_dl)
    
    # Update learning rate
    scheduler.step(val_loss_avg)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model_best.pth")
        print(f"  ✓ Saved best model with val_acc={val_acc:.2f}%")
    
    # Print epoch summary
    print(f"\nEpoch {epoch+1}/30:")
    print(f"  Train Loss: {train_loss/len(train_dl):.4f}, Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print("-" * 50)

# Save final model
torch.save(model.state_dict(), "model_final.pth")
print("\nTraining complete!")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print("Models saved: model_best.pth, model_final.pth")