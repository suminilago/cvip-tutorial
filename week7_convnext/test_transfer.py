from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torchvision.models import convnext_tiny

from dataset import get_dataloaders


BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    BASE_DIR
    / "checkpoints"
    / "convnext_tiny_transfer_best.pth"
)

REPORT_PATH = (
    BASE_DIR
    / "result"
    / "reports"
    / "convnext_tiny_transfer_classification_report.txt"
)

NUM_CLASSES = 30
BATCH_SIZE = 32


def get_class_names(dataset):
    if hasattr(dataset, "classes"):
        return dataset.classes

    if hasattr(dataset, "dataset"):
        if hasattr(dataset.dataset, "classes"):
            return dataset.dataset.classes

    return [
        f"class_{i}"
        for i in range(NUM_CLASSES)
    ]


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("Pretrained ConvNeXt-Tiny Transfer Learning Test")
    print("=" * 72)

    print("Device:", device)
    print("Checkpoint:", CHECKPOINT_PATH)
    print()

    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model = convnext_tiny(
        weights=None
    )

    in_features = model.classifier[2].in_features

    model.classifier[2] = nn.Linear(
        in_features,
        NUM_CLASSES,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

    accuracy = 100.0 * correct / total

    class_names = get_class_names(
        test_loader.dataset
    )

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

    print(
        f"Training phase: "
        f"{checkpoint['phase']}"
    )

    print(
        f"Test accuracy: "
        f"{accuracy:.2f}%"
    )

    print()
    print("=" * 72)
    print("Classification Report")
    print("=" * 72)
    print(report)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "ConvNeXt-Tiny Transfer Learning\n"
        )

        f.write(
            f"Best epoch: "
            f"{checkpoint['epoch']}\n"
        )

        f.write(
            f"Best validation accuracy: "
            f"{checkpoint['val_acc']:.2f}%\n"
        )

        f.write(
            f"Test accuracy: "
            f"{accuracy:.2f}%\n\n"
        )

        f.write(report)

    print()
    print("Report saved:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()