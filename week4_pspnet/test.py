import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from pspnet import PSPNet
from train import dice_score, iou_score


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
WEEK3_DIR = PROJECT_DIR / "week3_unet"

sys.path.append(str(WEEK3_DIR))

from dataset import (
    ISBI2012Dataset,
    TRAIN_IMG_DIR,
    TRAIN_LABEL_DIR,
)


SEED = 42
IMAGE_SIZE = (256, 256)
VALIDATION_RATIO = 0.2

WEIGHT_PATH = BASE_DIR / "weights" / "pspnet_best.pth"
RESULT_DIR = BASE_DIR / "result" / "predictions"


def get_validation_dataset():
    full_dataset = ISBI2012Dataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_LABEL_DIR,
        image_size=IMAGE_SIZE,
    )

    indices = list(range(len(full_dataset)))

    random.seed(SEED)
    random.shuffle(indices)

    validation_size = int(
        len(full_dataset) * VALIDATION_RATIO
    )

    validation_indices = indices[:validation_size]

    return Subset(
        full_dataset,
        validation_indices,
    )


def save_prediction(
    image: torch.Tensor,
    mask: torch.Tensor,
    prediction: torch.Tensor,
    save_path: Path,
) -> None:
    image_np = image.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    prediction_np = prediction.squeeze().cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np, cmap="gray")
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction_np, cmap="gray")
    plt.title("PSPNet Prediction")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"사용 장치: {device}")

    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {WEIGHT_PATH}"
        )

    validation_dataset = get_validation_dataset()

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = PSPNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
    ).to(device)

    model.load_state_dict(
        torch.load(
            WEIGHT_PATH,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for index, (images, masks) in enumerate(
            validation_loader,
            start=1,
        ):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits)
            predictions = (
                probabilities >= 0.5
            ).float()

            dice = dice_score(
                logits,
                masks,
            ).item()

            iou = iou_score(
                logits,
                masks,
            ).item()

            total_dice += dice
            total_iou += iou

            save_path = (
                RESULT_DIR
                / f"validation_{index:02d}.png"
            )

            save_prediction(
                image=images[0],
                mask=masks[0],
                prediction=predictions[0],
                save_path=save_path,
            )

            print(
                f"Sample {index:02d} "
                f"| Dice: {dice:.4f} "
                f"| IoU: {iou:.4f} "
                f"| 저장: {save_path.name}"
            )

    num_samples = len(validation_loader)

    print("=" * 70)
    print("PSPNet 검증 결과")
    print("=" * 70)
    print(
        f"평균 Dice: "
        f"{total_dice / num_samples:.4f}"
    )
    print(
        f"평균 IoU: "
        f"{total_iou / num_samples:.4f}"
    )
    print(f"예측 이미지 저장 위치: {RESULT_DIR}")


if __name__ == "__main__":
    main()