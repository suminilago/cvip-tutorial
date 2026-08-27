from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from dataset import get_dataloaders
from model import ConvNeXt


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
PLOT_DIR = BASE_DIR / "result" / "plots"
REPORT_DIR = BASE_DIR / "result" / "reports"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CHECKPOINT = (
    CHECKPOINT_DIR / "convnext_baseline_best.pth"
)

KERNEL3_CHECKPOINT = (
    CHECKPOINT_DIR / "convnext_kernel3_best.pth"
)

BATCH_SIZE = 32


def get_predictions(
    checkpoint_path,
    test_loader,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = ConvNeXt(
        num_classes=checkpoint["num_classes"],
        depths=checkpoint["depths"],
        dims=checkpoint["dims"],
        kernel_size=checkpoint["kernel_size"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

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


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("ConvNeXt Class-wise F1 Comparison")
    print("=" * 70)
    print("Device:", device)

    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    class_names = test_loader.dataset.classes

    labels_7, predictions_7 = get_predictions(
        BASELINE_CHECKPOINT,
        test_loader,
        device,
    )

    labels_3, predictions_3 = get_predictions(
        KERNEL3_CHECKPOINT,
        test_loader,
        device,
    )

    _, _, f1_7, _ = precision_recall_fscore_support(
        labels_7,
        predictions_7,
        labels=range(len(class_names)),
        zero_division=0,
    )

    _, _, f1_3, _ = precision_recall_fscore_support(
        labels_3,
        predictions_3,
        labels=range(len(class_names)),
        zero_division=0,
    )

    difference = f1_3 - f1_7

    print()
    print(
        f"{'Class':15s}"
        f"{'7x7 F1':>10s}"
        f"{'3x3 F1':>10s}"
        f"{'Delta':>10s}"
    )

    print("-" * 45)

    for name, score7, score3, diff in zip(
        class_names,
        f1_7,
        f1_3,
        difference,
    ):
        print(
            f"{name:15s}"
            f"{score7:10.4f}"
            f"{score3:10.4f}"
            f"{diff:+10.4f}"
        )

    # 가장 많이 개선된 클래스
    best_indices = np.argsort(difference)[-5:][::-1]

    # 가장 많이 악화된 클래스
    worst_indices = np.argsort(difference)[:5]

    print()
    print("Top 5 improved classes")
    print("-" * 45)

    for i in best_indices:
        print(
            f"{class_names[i]:15s} "
            f"{f1_7[i]:.4f} -> "
            f"{f1_3[i]:.4f} "
            f"({difference[i]:+.4f})"
        )

    print()
    print("Top 5 decreased classes")
    print("-" * 45)

    for i in worst_indices:
        print(
            f"{class_names[i]:15s} "
            f"{f1_7[i]:.4f} -> "
            f"{f1_3[i]:.4f} "
            f"({difference[i]:+.4f})"
        )

    # 전체 클래스 F1 비교
    x = np.arange(len(class_names))
    width = 0.38

    plt.figure(figsize=(18, 8))

    plt.bar(
        x - width / 2,
        f1_7,
        width,
        label="7x7",
    )

    plt.bar(
        x + width / 2,
        f1_3,
        width,
        label="3x3",
    )

    plt.xticks(
        x,
        class_names,
        rotation=90,
    )

    plt.xlabel("Class")
    plt.ylabel("F1-score")
    plt.title(
        "ConvNeXt Class-wise F1-score: 7x7 vs 3x3"
    )

    plt.legend()
    plt.tight_layout()

    save_path = (
        PLOT_DIR
        / "convnext_class_f1_comparison.png"
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Saved:", save_path)


if __name__ == "__main__":
    main()