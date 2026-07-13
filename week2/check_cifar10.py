import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class KaggleCIFAR10Dataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(label_csv)
        self.transform = transform

        self.classes = sorted(self.labels["label"].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_id = self.labels.iloc[idx]["id"]
        label_name = self.labels.iloc[idx]["label"]

        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        image = Image.open(image_path).convert("RGB")

        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label
    
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = KaggleCIFAR10Dataset(
    image_dir="./data/train",
    label_csv="./data/trainLabels.csv",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

images, labels = next(iter(train_loader))

print("dataset size:", len(train_dataset))
print("classes:", train_dataset.classes)
print("images shape:", images.shape)
print("labels shape:", labels.shape)
print("labels:", labels[:10])