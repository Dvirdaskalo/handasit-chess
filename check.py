from collections import Counter
from torchvision import datasets

ds = datasets.ImageFolder("data/train")
print(Counter(ds.targets))
print(ds.classes)
