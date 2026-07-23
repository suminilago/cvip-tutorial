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
# 기본 설정
# ============================================================

SEED = 42
BATCH_SIZE = 64

STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 20
TOTAL_EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS

TARGET_VAL_ACC = 85.0
TARGET_TEST_ACC = 85.0

DATA_DIR = Path("./data/train")
LABEL_CSV = Path("./data/trainLabels.csv")

MODEL_PATH = Path("resnet18_transfer_best.pth")
HISTORY_PATH = Path("resnet18_transfer_history.csv")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class KaggleCIFAR10Dataset(Dataset):
    def __init__(self, image_dir, label_csv, indices, transform=None):
        self.image_dir = Path(image_dir)
        self.labels = pd.read_csv(label_csv)
        self.indices = list(indices)
        self.transform = transform

        self.classes = sorted(self.labels["label"].unique())
        self.class_to_idx = {
            name: idx for idx, name in enumerate(self.classes)
        }

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

        image = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def make_splits(dataset_size: int):
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(
        dataset_size,
        generator=generator
    ).tolist()

    return (
        indices[:40000],
        indices[40000:45000],
        indices[45000:]
    )


def create_model():
    print("ImageNet pretrained ResNet-18을 불러옵니다.")

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


def configure_stage1(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.conv1.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer = optim.AdamW(
        [
            {"params": model.conv1.parameters(), "lr": 1e-3},
            {"params": model.fc.parameters(), "lr": 1e-3}
        ],
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=STAGE1_EPOCHS
    )

    return optimizer, scheduler


def configure_stage2(model):
    for parameter in model.parameters():
        parameter.requires_grad = True

    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("conv1")
        and not name.startswith("fc")
    ]

    optimizer = optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 1e-4},
            {"params": model.conv1.parameters(), "lr": 3e-4},
            {"params": model.fc.parameters(), "lr": 5e-4}
        ],
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=STAGE2_EPOCHS,
        eta_min=1e-6
    )

    return optimizer, scheduler


def set_batchnorm_eval(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    freeze_batchnorm=False
):
    model.train()

    if freeze_batchnorm:
        set_batchnorm_eval(model)

    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += predictions.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            elapsed = (time.time() - start_time) / 60
            current_acc = 100.0 * correct / total

            print(
                f"  Batch [{batch_idx + 1}/{len(loader)}] | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {current_acc:.2f}% | "
                f"Elapsed: {elapsed:.1f} min"
            )

    return (
        total_loss / total,
        100.0 * correct / total
    )


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += predictions.eq(labels).sum().item()
            total += labels.size(0)

    return (
        total_loss / total,
        100.0 * correct / total
    )


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
                "stage",
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


def run_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    history,
    start_epoch,
    num_epochs,
    stage_name,
    best_val_acc,
    freeze_batchnorm=False
):
    for local_epoch in range(num_epochs):
        epoch = start_epoch + local_epoch + 1
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print()
        print("=" * 70)
        print(
            f"Epoch [{epoch}/{TOTAL_EPOCHS}] | "
            f"{stage_name} | LR: {current_lr:.6f}"
        )
        print("=" * 70)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            freeze_batchnorm
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        epoch_minutes = (time.time() - epoch_start) / 60

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}%"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        print(f"Epoch Time: {epoch_minutes:.1f} min")

        history.append({
            "epoch": epoch,
            "stage": stage_name,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": current_lr,
            "epoch_minutes": epoch_minutes
        })

        save_history(history)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_accuracy": best_val_acc,
                    "epoch": epoch,
                    "stage": stage_name
                },
                MODEL_PATH
            )

            print(f"Best model saved: {MODEL_PATH}")

        scheduler.step()

    

    return best_val_acc, False


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("CIFAR-10 Transfer Learning ResNet-18")
    print("=" * 70)
    print("device:", device)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"학습 이미지 폴더를 찾을 수 없습니다: "
            f"{DATA_DIR.resolve()}"
        )

    if not LABEL_CSV.exists():
        raise FileNotFoundError(
            f"라벨 CSV 파일을 찾을 수 없습니다: "
            f"{LABEL_CSV.resolve()}"
        )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    dataset_size = len(pd.read_csv(LABEL_CSV))

    train_indices, val_indices, test_indices = (
        make_splits(dataset_size)
    )

    train_dataset = KaggleCIFAR10Dataset(
        DATA_DIR,
        LABEL_CSV,
        train_indices,
        train_transform
    )

    val_dataset = KaggleCIFAR10Dataset(
        DATA_DIR,
        LABEL_CSV,
        val_indices,
        eval_transform
    )

    test_dataset = KaggleCIFAR10Dataset(
        DATA_DIR,
        LABEL_CSV,
        test_indices,
        eval_transform
    )

    loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available()
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_options
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_options
    )

    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))
    print("Internal test size:", len(test_dataset))
    print("Classes:", train_dataset.classes)

    model = create_model().to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    history = []
    best_val_acc = 0.0
    total_start = time.time()

    print()
    print("Stage 1: conv1 + fc 학습")

    optimizer, scheduler = configure_stage1(model)

    best_val_acc, early_stopped = run_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        history,
        start_epoch=0,
        num_epochs=STAGE1_EPOCHS,
        stage_name="Stage 1",
        best_val_acc=best_val_acc,
        freeze_batchnorm=True
    )

    if not early_stopped:
        print()
        print("Stage 2: 전체 모델 fine-tuning")

        optimizer, scheduler = configure_stage2(model)

        best_val_acc, _ = run_stage(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            history,
            start_epoch=STAGE1_EPOCHS,
            num_epochs=STAGE2_EPOCHS,
            stage_name="Stage 2",
            best_val_acc=best_val_acc
        )

    total_minutes = (time.time() - total_start) / 60

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

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print()
    print("Internal Test Result")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    if test_acc >= TARGET_TEST_ACC:
        print(
            "Transfer Learning ResNet-18 목표인 "
            "Test Accuracy 85% 이상을 달성했습니다."
        )
    else:
        print(
            "Test Accuracy 85%에 아직 도달하지 못했습니다."
        )


if __name__ == "__main__":
    main()