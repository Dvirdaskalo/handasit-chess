import torch

ckpt = torch.load("model_best.pth", map_location="cpu")

print(type(ckpt))

if isinstance(ckpt, dict):
    print("\nTop-level keys:")
    print(ckpt.keys())
