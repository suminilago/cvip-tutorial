import csv
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dataset import (
    ISBI2012Dataset,
    TRAIN_IMG_DIR,
    TRAIN_LABEL_DIR,
)
from unet import UNet


SEED = 42
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 2
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
VALIDATION_RATIO = 0.2

BASE_DIR = Path(__file__).resolve().parent
WEIGHT_PATH = BASE_DIR / "weights" / "unet_bce_dice_best.pth"
HISTORY_PATH = BASE_DIR / "result" / "unet_bce_dice_history.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    prediction_sum = predictions.sum(dim=(1, 2, 3))
    target_sum = targets.sum(dim=(1, 2, 3))

    dice = (
        2.0 * intersection + epsilon
    ) / (
        prediction_sum + target_sum + epsilon
    )

    return dice.mean()


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = (
        predictions.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
        - intersection
    )

    iou = (intersection + epsilon) / (union + epsilon)

    return iou.mean()

class DiceLoss(nn.Module):
    """
    sigmoid를 적용한 예측 확률과 정답 마스크 사이의
    Dice Loss를 계산합니다.
    """

    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection + self.epsilon
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.epsilon
        )

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    BCE Loss와 Dice Loss를 같은 비율로 결합합니다.

    total_loss = 0.5 * BCE + 0.5 * Dice
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return (
            self.bce_weight * bce
            + self.dice_weight * dice
        )

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(logits.detach(), masks).item()
        total_iou += iou_score(logits.detach(), masks).item()

    num_batches = len(dataloader)

    return (
        total_loss / num_batches,
        total_dice / num_batches,
        total_iou / num_batches,
    )


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            total_loss += loss.item()
            total_dice += dice_score(logits, masks).item()
            total_iou += iou_score(logits, masks).item()

    num_batches = len(dataloader)

    return (
        total_loss / num_batches,
        total_dice / num_batches,
        total_iou / num_batches,
    )


def save_history(history: list[dict]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_dice",
                "train_iou",
                "val_loss",
                "val_dice",
                "val_iou",
            ],
        )

        writer.writeheader()
        writer.writerows(history)


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"사용 장치: {device}")

    full_dataset = ISBI2012Dataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_LABEL_DIR,
        image_size=IMAGE_SIZE,
    )

    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    validation_size = int(
        len(full_dataset) * VALIDATION_RATIO
    )

    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    train_dataset = Subset(
        full_dataset,
        train_indices,
    )

    validation_dataset = Subset(
        full_dataset,
        validation_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"전체 데이터: {len(full_dataset)}")
    print(f"학습 데이터: {len(train_dataset)}")
    print(f"검증 데이터: {len(validation_dataset)}")

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ).to(device)

    criterion = BCEDiceLoss(
    bce_weight=0.5,
    dice_weight=0.5,
)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    WEIGHT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_dice = 0.0
    history = []

    print("=" * 90)
    print("U-Net 학습 시작")
    print("=" * 90)

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_dice, train_iou = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_dice, val_iou = validate(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        elapsed_time = time.time() - start_time

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
            }
        )

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Dice: {train_dice:.4f} "
            f"| Train IoU: {train_iou:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Dice: {val_dice:.4f} "
            f"| Val IoU: {val_iou:.4f} "
            f"| Time: {elapsed_time:.1f}s"
        )

        if val_dice > best_validation_dice:
            best_validation_dice = val_dice

            torch.save(
                model.state_dict(),
                WEIGHT_PATH,
            )

            print(
                f"  → 최고 모델 저장: "
                f"Val Dice {best_validation_dice:.4f}"
            )

        save_history(history)

    print("=" * 90)
    print("학습 완료")
    print(f"최고 Validation Dice: {best_validation_dice:.4f}")
    print(f"모델 저장 경로: {WEIGHT_PATH}")
    print(f"학습 기록 경로: {HISTORY_PATH}")


if __name__ == "__main__":
    main()