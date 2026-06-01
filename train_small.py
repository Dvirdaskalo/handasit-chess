import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='collect/labeled', help='Labeled data dir (subfolders per class)')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--out', type=str, default='model_finetuned.pth')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100,100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Use ImageFolder; expects subfolders for each class
data_path = Path(args.data)
if not data_path.exists():
    raise SystemExit(f"Data path {data_path} does not exist")

dataset = datasets.ImageFolder(str(data_path), transform=transform)
loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=2)

# Build model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print(f"Classes: {dataset.classes}")

for epoch in range(args.epochs):
    model.train()
    running = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
    avg = running / len(dataset)
    print(f"Epoch {epoch+1}/{args.epochs} loss={avg:.4f}")

# Save
torch.save(model.state_dict(), args.out)
print(f"Saved fine-tuned model to {args.out}")
