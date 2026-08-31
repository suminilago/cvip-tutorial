import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import ConvNeXt


# =========================
# 기본 설정
# =========================
SEED = 42

BATCH_SIZE = 32
NUM_EPOCHS = 20

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1

NUM_CLASSES = 30

DEPTHS = (2, 2, 3, 2)
DIMS = (64, 128, 256, 512)
KERNEL_SIZE = 3


BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
REPORT_DIR = BASE_DIR / "result" / "reports"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "convnext_kernel3_ls_best.pth"
)

HISTORY_PATH = (
    REPORT_DIR / "convnext_kernel3_ls_history.csv"
)


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("ConvNeXt Kernel 3x3 + Label Smoothing Training")
    print("=" * 70)
    print("Device:", device)
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Label smoothing:", LABEL_SMOOTHING)
    print("Depths:", DEPTHS)
    print("Dims:", DIMS)
    print("Kernel size:", KERNEL_SIZE)
    print()

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    model = ConvNeXt(
        num_classes=NUM_CLASSES,
        depths=DEPTHS,
        dims=DIMS,
        kernel_size=KERNEL_SIZE,
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {total_params:,}")
    print()

    # Train loss에는 label smoothing 적용
    train_criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    # Validation loss는 baseline과 동일한 일반 CE 사용
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    best_val_acc = 0.0
    best_epoch = 0

    history = []

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            train_criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            eval_criterion,
            device,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        epoch_time = time.time() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr,
            }
        )

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time / 60:.1f} min"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "depths": DEPTHS,
                    "dims": DIMS,
                    "kernel_size": KERNEL_SIZE,
                    "num_classes": NUM_CLASSES,
                    "label_smoothing": LABEL_SMOOTHING,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "batch_size": BATCH_SIZE,
                    "num_epochs": NUM_EPOCHS,
                },
                BEST_MODEL_PATH,
            )

            print(
                f"  → Best model saved "
                f"(Val Acc: {best_val_acc:.2f}%)"
            )

    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "learning_rate",
            ],
        )

        writer.writeheader()
        writer.writerows(history)

    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("Training completed")
    print("=" * 70)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Total Training Time: {total_time / 60:.1f} min")
    print("Best model:", BEST_MODEL_PATH)
    print("History:", HISTORY_PATH)


if __name__ == "__main__":
    main()