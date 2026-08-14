import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from model import build_vit


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULT_DIR = BASE_DIR / "result"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

MODEL_DEPTH = 2
NUM_CLASSES = 30

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_vit_depth2.pth"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_vit_depth2.pth"
HISTORY_PATH = RESULT_DIR / "history_depth2.csv"
CONFIG_PATH = RESULT_DIR / "training_config_depth2.json"


def calculate_accuracy(logits, labels):
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    start_time = time.time()

    for batch_index, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        running_correct += calculate_accuracy(logits, labels)
        total_samples += batch_size

        if batch_index % 100 == 0 or batch_index == len(dataloader):
            elapsed = time.time() - start_time

            current_loss = running_loss / total_samples
            current_accuracy = 100.0 * running_correct / total_samples

            print(
                f"  Batch {batch_index:>3}/{len(dataloader)} | "
                f"Loss: {current_loss:.4f} | "
                f"Accuracy: {current_accuracy:.2f}% | "
                f"Time: {elapsed / 60:.1f} min"
            )

    epoch_loss = running_loss / total_samples
    epoch_accuracy = 100.0 * running_correct / total_samples

    return epoch_loss, epoch_accuracy


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        running_correct += calculate_accuracy(logits, labels)
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_accuracy = 100.0 * running_correct / total_samples

    return epoch_loss, epoch_accuracy


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_accuracy,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_accuracy": best_val_accuracy,
        "model_depth": MODEL_DEPTH,
        "num_classes": NUM_CLASSES,
    }

    torch.save(checkpoint, path)


def save_history(history):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "learning_rate",
        "epoch_minutes",
    ]

    with HISTORY_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)


def save_training_config(device):
    config = {
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_workers": NUM_WORKERS,
        "model_depth": MODEL_DEPTH,
        "num_classes": NUM_CLASSES,
        "image_size": 128,
        "patch_size": 16,
        "embed_dim": 192,
        "num_heads": 3,
        "mlp_dim": 384,
        "dropout": 0.1,
        "device": str(device),
    }

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=4,
            ensure_ascii=False,
        )


def main():
    torch.manual_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("Vision Transformer 학습")
    print("=" * 70)
    print(f"사용 장치: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epoch 수: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Transformer Block 수: {MODEL_DEPTH}")

    save_training_config(device)

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = build_vit(
        num_classes=NUM_CLASSES,
        depth=MODEL_DEPTH,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"전체 파라미터 수: {total_parameters:,}")
    print(f"Train 배치 수: {len(train_loader)}")
    print(f"Validation 배치 수: {len(val_loader)}")

    history = []
    best_val_accuracy = 0.0

    total_start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        print("\n" + "=" * 70)
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print("=" * 70)

        current_learning_rate = optimizer.param_groups[0]["lr"]

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_minutes = (time.time() - epoch_start_time) / 60

        print(
            f"\nTrain Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}%"
        )
        print(
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )
        print(
            f"Learning Rate: {current_learning_rate:.8f} | "
            f"Epoch Time: {epoch_minutes:.1f} min"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_learning_rate,
                "epoch_minutes": epoch_minutes,
            }
        )

        save_history(history)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            save_checkpoint(
                path=BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
            )

            print(
                f"최고 검증 정확도 갱신: "
                f"{best_val_accuracy:.2f}%"
            )
            print(f"저장: {BEST_MODEL_PATH}")

        save_checkpoint(
            path=LAST_MODEL_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_accuracy=best_val_accuracy,
        )

        scheduler.step()

    total_minutes = (time.time() - total_start_time) / 60

    print("\n" + "=" * 70)
    print("학습 완료")
    print("=" * 70)
    print(f"최고 Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"전체 학습 시간: {total_minutes:.1f}분")
    print(f"최고 모델: {BEST_MODEL_PATH}")
    print(f"학습 기록: {HISTORY_PATH}")


if __name__ == "__main__":
    main()