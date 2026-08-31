import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from dataset import get_dataloaders


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "convnext_tiny_transfer_best.pth"
HISTORY_PATH = BASE_DIR / "result" / "reports" / "convnext_tiny_transfer_history.csv"

SEED = 42
NUM_CLASSES = 30
BATCH_SIZE = 32

FREEZE_EPOCHS = 3
FINETUNE_EPOCHS = 7

HEAD_LR = 1e-3
FINETUNE_LR = 3e-5
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def set_trainable(model, phase):
    for p in model.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = True

    if phase == "finetune":
        for p in model.features[-1].parameters():
            p.requires_grad = True


def train_phase(
    model,
    train_loader,
    val_loader,
    device,
    start_epoch,
    epochs,
    lr,
    phase,
    best_val,
    best_epoch,
    history,
):
    set_trainable(model, phase)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    train_criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )
    val_criterion = nn.CrossEntropyLoss()

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"\n[{phase}] Trainable parameters: {trainable:,}")

    for i in range(epochs):
        epoch = start_epoch + i
        start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            train_criterion,
            device,
            optimizer,
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            val_criterion,
            device,
        )

        history.append([
            epoch,
            phase,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
        ])

        print(
            f"Epoch [{epoch:02d}/{FREEZE_EPOCHS + FINETUNE_EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.2f}% "
            f"| LR: {current_lr:.6f} "
            f"| Time: {(time.time() - start) / 60:.1f} min"
        )

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch

            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "epoch": epoch,
                    "phase": phase,
                    "val_acc": val_acc,
                    "model_state_dict": model.state_dict(),
                },
                CHECKPOINT_PATH,
            )

            print(f"  → Best model saved (Val Acc: {val_acc:.2f}%)")

        scheduler.step()

    return best_val, best_epoch


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("Pretrained ConvNeXt-Tiny Transfer Learning")
    print("=" * 70)
    print("Device:", device)
    print("Batch size:", BATCH_SIZE)
    print("Classifier epochs:", FREEZE_EPOCHS)
    print("Fine-tuning epochs:", FINETUNE_EPOCHS)
    print("Label smoothing:", LABEL_SMOOTHING)

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    model = convnext_tiny(
        weights=ConvNeXt_Tiny_Weights.DEFAULT
    )

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)

    print(
        "Total parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    history = []
    best_val = 0.0
    best_epoch = 0
    start = time.time()

    best_val, best_epoch = train_phase(
        model,
        train_loader,
        val_loader,
        device,
        start_epoch=1,
        epochs=FREEZE_EPOCHS,
        lr=HEAD_LR,
        phase="classifier",
        best_val=best_val,
        best_epoch=best_epoch,
        history=history,
    )

    best_val, best_epoch = train_phase(
        model,
        train_loader,
        val_loader,
        device,
        start_epoch=FREEZE_EPOCHS + 1,
        epochs=FINETUNE_EPOCHS,
        lr=FINETUNE_LR,
        phase="finetune",
        best_val=best_val,
        best_epoch=best_epoch,
        history=history,
    )

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "phase",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "learning_rate",
        ])
        writer.writerows(history)

    print("\n" + "=" * 70)
    print("Training completed")
    print("=" * 70)
    print("Best Epoch:", best_epoch)
    print(f"Best Validation Accuracy: {best_val:.2f}%")
    print(f"Total Training Time: {(time.time() - start) / 60:.1f} min")
    print("Best model:", CHECKPOINT_PATH)
    print("History:", HISTORY_PATH)


if __name__ == "__main__":
    main()