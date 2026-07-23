import csv
import time
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# 1. 기본 설정
# ============================================================

SEED = 42

BATCH_SIZE = 64
MAX_EPOCHS = 23
LEARNING_RATE = 0.1
TARGET_VAL_ACC = 80.0

DATA_DIR = Path("./data/train")
LABEL_CSV = Path("./data/trainLabels.csv")

MODEL_PATH = Path("resnet18_scratch_best.pth")
HISTORY_PATH = Path("resnet18_scratch_history.csv")


# ============================================================
# 2. Seed 고정
# ============================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. Kaggle CIFAR-10 Dataset
# ============================================================

class KaggleCIFAR10Dataset(Dataset):
    def __init__(
        self,
        image_dir,
        label_csv,
        indices=None,
        transform=None
    ):
        self.image_dir = Path(image_dir)
        self.labels = pd.read_csv(label_csv)
        self.transform = transform

        self.classes = sorted(
            self.labels["label"].unique()
        )

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

        if indices is None:
            self.indices = list(
                range(len(self.labels))
            )
        else:
            self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        image_id = self.labels.iloc[real_idx]["id"]
        label_name = self.labels.iloc[real_idx]["label"]

        image_path = self.image_dir / f"{image_id}.png"

        if not image_path.exists():
            raise FileNotFoundError(
                f"이미지를 찾을 수 없습니다: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        label = self.class_to_idx[label_name]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# 4. Train / Validation / Internal Test 분리
#
# 전체 50,000장
# Train: 40,000장
# Validation: 5,000장
# Internal Test: 5,000장
# ============================================================

def make_splits(dataset_size: int):
    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        dataset_size,
        generator=generator
    ).tolist()

    train_end = 40000
    val_end = 45000

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return (
        train_indices,
        val_indices,
        test_indices
    )


# ============================================================
# 5. CIFAR-10용 ResNet-18 생성
# ============================================================

def create_model():
    model = models.resnet18(
        weights=None
    )

    # CIFAR-10의 32x32 입력에 맞게 수정
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    # 작은 입력에서는 초기 max pooling 제거
    model.maxpool = nn.Identity()

    # CIFAR-10 클래스 수는 10개
    model.fc = nn.Linear(
        model.fc.in_features,
        10
    )

    return model


# ============================================================
# 6. 한 epoch 학습
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions.eq(labels)
            .sum()
            .item()
        )

        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            elapsed = (
                time.time() - start_time
            )

            current_acc = (
                100.0 * correct / total
            )

            print(
                f"  Batch "
                f"[{batch_idx + 1}/{len(loader)}] | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {current_acc:.2f}% | "
                f"Elapsed: {elapsed / 60:.1f} min"
            )

    average_loss = (
        total_loss / total
    )

    accuracy = (
        100.0 * correct / total
    )

    return average_loss, accuracy


# ============================================================
# 7. Validation / Test 평가
# ============================================================

def evaluate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions.eq(labels)
                .sum()
                .item()
            )

            total += labels.size(0)

    average_loss = (
        total_loss / total
    )

    accuracy = (
        100.0 * correct / total
    )

    return average_loss, accuracy


# ============================================================
# 8. History 저장
# ============================================================

def save_history(history):
    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "learning_rate",
                "epoch_minutes"
            ]
        )

        writer.writeheader()
        writer.writerows(history)


# ============================================================
# 9. Main
# ============================================================

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    train_transform = transforms.Compose([
        transforms.RandomCrop(
            32,
            padding=4
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.4914,
                0.4822,
                0.4465
            ],
            std=[
                0.2470,
                0.2435,
                0.2616
            ]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.4914,
                0.4822,
                0.4465
            ],
            std=[
                0.2470,
                0.2435,
                0.2616
            ]
        )
    ])

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------

    labels = pd.read_csv(
        LABEL_CSV
    )

    dataset_size = len(labels)

    (
        train_indices,
        val_indices,
        test_indices
    ) = make_splits(dataset_size)

    train_dataset = KaggleCIFAR10Dataset(
        image_dir=DATA_DIR,
        label_csv=LABEL_CSV,
        indices=train_indices,
        transform=train_transform
    )

    val_dataset = KaggleCIFAR10Dataset(
        image_dir=DATA_DIR,
        label_csv=LABEL_CSV,
        indices=val_indices,
        transform=eval_transform
    )

    test_dataset = KaggleCIFAR10Dataset(
        image_dir=DATA_DIR,
        label_csv=LABEL_CSV,
        indices=test_indices,
        transform=eval_transform
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(
        "train size:",
        len(train_dataset)
    )

    print(
        "validation size:",
        len(val_dataset)
    )

    print(
        "internal test size:",
        len(test_dataset)
    )

    print(
        "classes:",
        train_dataset.classes
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model().to(
        device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[12, 18],
        gamma=0.1
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_acc = 0.0
    history = []

    total_start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print()
        print("=" * 70)

        print(
            f"Epoch "
            f"[{epoch + 1}/{MAX_EPOCHS}] 시작 | "
            f"LR: {current_lr:.6f}"
        )

        print("=" * 70)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        epoch_minutes = (
            time.time() - epoch_start
        ) / 60

        print()

        print(
            f"Epoch "
            f"[{epoch + 1}/{MAX_EPOCHS}] 완료"
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}%"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        print(
            f"Epoch Time: "
            f"{epoch_minutes:.1f} min"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": current_lr,
            "epoch_minutes": epoch_minutes
        })

        save_history(history)

        # Validation 성능이 좋아지면 best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                f"best model saved: "
                f"{MODEL_PATH}"
            )

        # 다음 epoch의 learning rate 갱신
        scheduler.step()

        # Validation Accuracy 80% 이상이면 종료
        if val_acc >= TARGET_VAL_ACC:
            print()
            print(
                f"목표 Validation Accuracy "
                f"{TARGET_VAL_ACC:.2f}% 이상 달성"
            )

            print(
                "Scratch 학습을 "
                "조기 종료합니다."
            )

            break

    # --------------------------------------------------------
    # Best model 결과 출력
    # --------------------------------------------------------

    total_minutes = (
        time.time() - total_start_time
    ) / 60

    print()
    print("=" * 70)

    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.2f}%"
    )

    print(
        f"Total Training Time: "
        f"{total_minutes:.1f} min"
    )

    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"저장된 모델을 찾을 수 없습니다: "
            f"{MODEL_PATH}"
        )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    # --------------------------------------------------------
    # Internal test는 학습 종료 후 한 번만 평가
    # --------------------------------------------------------

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print()
    print("Internal Test Result")

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_acc:.2f}%"
    )

    if test_acc >= 75.0:
        print()
        print(
            "Scratch ResNet-18 목표인 "
            "Test Accuracy 75% 이상을 "
            "달성했습니다."
        )
    else:
        print()
        print(
            "Scratch ResNet-18 목표인 "
            "Test Accuracy 75%에 "
            "도달하지 못했습니다."
        )


if __name__ == "__main__":
    main()