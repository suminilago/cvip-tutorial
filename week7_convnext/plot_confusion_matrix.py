from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import get_dataloaders
from model import ConvNeXt


# ============================================================
# 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
PLOT_DIR = BASE_DIR / "result" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CHECKPOINT = (
    CHECKPOINT_DIR / "convnext_baseline_best.pth"
)

KERNEL3_CHECKPOINT = (
    CHECKPOINT_DIR / "convnext_kernel3_best.pth"
)

BATCH_SIZE = 32
NUM_CLASSES = 30

DEPTHS = (2, 2, 3, 2)
DIMS = (64, 128, 256, 512)


# ============================================================
# 예측값 생성
# ============================================================

def get_predictions(
    checkpoint_path,
    kernel_size,
    test_loader,
    device,
):
    model = ConvNeXt(
        num_classes=NUM_CLASSES,
        depths=DEPTHS,
        dims=DIMS,
        kernel_size=kernel_size,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(
                predictions.cpu().numpy()
            )

    return (
        np.array(all_labels),
        np.array(all_predictions),
    )


# ============================================================
# Confusion Matrix 저장
# ============================================================

def plot_confusion(
    labels,
    predictions,
    class_names,
    title,
    save_path,
):
    cm = confusion_matrix(
        labels,
        predictions,
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(18, 18))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
        xticks_rotation=90,
        colorbar=True,
    )

    ax.set_title(
        title,
        fontsize=18,
        pad=20,
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:", save_path)


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("ConvNeXt Confusion Matrix Analysis")
    print("=" * 70)
    print("Device:", device)

    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    class_names = test_loader.dataset.classes

    print("Number of test samples:", len(test_loader.dataset))
    print("Number of classes:", len(class_names))

    # --------------------------------------------------------
    # Baseline 7x7
    # --------------------------------------------------------

    print()
    print("Evaluating ConvNeXt 7x7...")

    labels_7, predictions_7 = get_predictions(
        BASELINE_CHECKPOINT,
        kernel_size=7,
        test_loader=test_loader,
        device=device,
    )

    plot_confusion(
        labels_7,
        predictions_7,
        class_names,
        title="ConvNeXt 7x7 Normalized Confusion Matrix",
        save_path=(
            PLOT_DIR
            / "convnext_kernel7_confusion_matrix.png"
        ),
    )

    # --------------------------------------------------------
    # Kernel 3x3
    # --------------------------------------------------------

    print()
    print("Evaluating ConvNeXt 3x3...")

    labels_3, predictions_3 = get_predictions(
        KERNEL3_CHECKPOINT,
        kernel_size=3,
        test_loader=test_loader,
        device=device,
    )

    plot_confusion(
        labels_3,
        predictions_3,
        class_names,
        title="ConvNeXt 3x3 Normalized Confusion Matrix",
        save_path=(
            PLOT_DIR
            / "convnext_kernel3_confusion_matrix.png"
        ),
    )

    print()
    print("=" * 70)
    print("Completed")
    print("=" * 70)


if __name__ == "__main__":
    main()