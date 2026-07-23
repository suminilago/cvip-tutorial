import os
import time

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models


class KaggleCIFAR10Dataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(label_csv)
        self.transform = transform

        self.classes = sorted(self.labels["label"].unique())
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_id = self.labels.iloc[idx]["id"]
        label_name = self.labels.iloc[idx]["label"]

        image_path = os.path.join(
            self.image_dir,
            f"{image_id}.png"
        )

        image = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# CIFAR-10은 원본 크기가 32x32이므로 224x224로 확대하지 않음
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])


dataset = KaggleCIFAR10Dataset(
    image_dir="./data/train",
    label_csv="./data/trainLabels.csv",
    transform=transform
)


# 실행할 때마다 같은 방식으로 train/validation 분할
generator = torch.Generator().manual_seed(42)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)


train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("device:", device)
print("dataset size:", len(dataset))
print("train size:", len(train_dataset))
print("validation size:", len(val_dataset))
print("classes:", dataset.classes)


# torchvision ResNet18을 CIFAR-10 입력 크기 32x32에 맞게 수정
model = models.resnet18(weights=None)

model.conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

# ImageNet용 초기 MaxPool 제거
model.maxpool = nn.Identity()

# CIFAR-10은 10개 클래스
model.fc = nn.Linear(
    model.fc.in_features,
    10
)

model = model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


def train_one_epoch():
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted = outputs.argmax(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            current_acc = 100.0 * correct / total
            elapsed = time.time() - start_time

            print(
                f"  Batch [{batch_idx + 1}/{len(train_loader)}] | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {current_acc:.2f}% | "
                f"Elapsed: {elapsed / 60:.1f} min"
            )

    average_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def evaluate():
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predicted = outputs.argmax(dim=1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# CPU 속도 확인을 위한 1 epoch 시험 실행
num_epochs = 1

best_val_acc = 0.0

for epoch in range(num_epochs):
    epoch_start = time.time()

    print(f"\nEpoch [{epoch + 1}/{num_epochs}] 시작")

    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate()

    epoch_time = time.time() - epoch_start

    print(
        f"\nEpoch [{epoch + 1}/{num_epochs}] 완료\n"
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}%\n"
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}%\n"
        f"Epoch Time: {epoch_time / 60:.1f} min"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "resnet18_cifar10_best.pth"
        )

        print(
            "best model saved: "
            "resnet18_cifar10_best.pth"
        )


print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")