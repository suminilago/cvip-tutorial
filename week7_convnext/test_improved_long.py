from pathlib import Path

import torch
from sklearn.metrics import classification_report

from dataset import get_dataloaders
from model import ConvNeXt


BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    BASE_DIR
    / "checkpoints"
    / "convnext_kernel3_ls_30ep_best.pth"
)

REPORT_DIR = BASE_DIR / "result" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = (
    REPORT_DIR
    / "convnext_kernel3_ls_30ep_classification_report.txt"
)

BATCH_SIZE = 32


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print(
        "ConvNeXt Kernel 3x3 "
        "+ Label Smoothing "
        "+ 30 Epoch Test"
    )
    print("=" * 70)
    print("Device:", device)
    print("Checkpoint:", CHECKPOINT_PATH)
    print()

    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    class_names = test_loader.dataset.classes

    checkpoint = torch.load(
        CHECKPOINT_PATH,
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

    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += (
                predicted.eq(labels).sum().item()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predicted.cpu().numpy()
            )

    test_acc = 100.0 * correct / total

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
    )

    print(
        f"Best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation accuracy: "
        f"{checkpoint['val_acc']:.2f}%"
    )

    if "label_smoothing" in checkpoint:
        print(
            f"Label smoothing: "
            f"{checkpoint['label_smoothing']}"
        )

    if "num_epochs" in checkpoint:
        print(
            f"Training epochs: "
            f"{checkpoint['num_epochs']}"
        )

    print(
        f"Test accuracy: "
        f"{test_acc:.2f}%"
    )

    print()
    print("=" * 70)
    print("Classification Report")
    print("=" * 70)
    print(report)

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "ConvNeXt Kernel 3x3 "
            "+ Label Smoothing "
            "+ 30 Epoch Test Result\n"
        )

        f.write("=" * 70 + "\n")

        f.write(
            f"Best epoch: "
            f"{checkpoint['epoch']}\n"
        )

        f.write(
            f"Best validation accuracy: "
            f"{checkpoint['val_acc']:.2f}%\n"
        )

        if "label_smoothing" in checkpoint:
            f.write(
                f"Label smoothing: "
                f"{checkpoint['label_smoothing']}\n"
            )

        if "num_epochs" in checkpoint:
            f.write(
                f"Training epochs: "
                f"{checkpoint['num_epochs']}\n"
            )

        f.write(
            f"Test accuracy: "
            f"{test_acc:.2f}%\n\n"
        )

        f.write(report)

    print()
    print("Report saved:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()