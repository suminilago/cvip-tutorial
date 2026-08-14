from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from dataset import get_dataloaders, get_datasets
from model import build_vit


BASE_DIR = Path(__file__).resolve().parent

# "baseline" 또는 "label_smoothing"
EXPERIMENT = "label_smoothing"

CHECKPOINTS = {
    "baseline": BASE_DIR / "checkpoints" / "best_vit.pth",
    "label_smoothing": BASE_DIR / "checkpoints" / "best_vit_ls.pth",
    "depth2": BASE_DIR / "checkpoints" / "best_vit_depth2.pth",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@torch.no_grad()
def evaluate(model, loader):
    criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_predictions = []

    top3_correct = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

        top3_indices = outputs.topk(
            k=3,
            dim=1,
        ).indices

        top3_correct += (
            top3_indices == labels.unsqueeze(1)
        ).any(dim=1).sum().item()

    test_loss = total_loss / total_samples

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    top3_accuracy = top3_correct / total_samples

    return {
        "loss": test_loss,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "top3_accuracy": top3_accuracy,
        "labels": all_labels,
        "predictions": all_predictions,
    }


def main():
    checkpoint_path = CHECKPOINTS[EXPERIMENT]

    print("=" * 70)
    print("Vision Transformer Test")
    print("=" * 70)
    print(f"실험: {EXPERIMENT}")
    print(f"체크포인트: {checkpoint_path}")
    print(f"사용 장치: {DEVICE}")

    _, _, test_loader = get_dataloaders(
        batch_size=32,
        num_workers=0,
    )

    train_dataset, _, _ = get_datasets()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model = build_vit(
        num_classes=30,
        depth=checkpoint["model_depth"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    result = evaluate(
        model,
        test_loader,
    )

    print("\n" + "=" * 70)
    print("Test 결과")
    print("=" * 70)
    print(f"Test Loss      : {result['loss']:.4f}")
    print(
        f"Accuracy       : "
        f"{result['accuracy'] * 100:.2f}%"
    )
    print(
        f"Macro Precision: "
        f"{result['macro_precision']:.4f}"
    )
    print(
        f"Macro Recall   : "
        f"{result['macro_recall']:.4f}"
    )
    print(
        f"Macro F1-score : "
        f"{result['macro_f1']:.4f}"
    )
    print(
        f"Top-3 Accuracy : "
        f"{result['top3_accuracy'] * 100:.2f}%"
    )

    report = classification_report(
        result["labels"],
        result["predictions"],
        target_names=train_dataset.classes,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        result["labels"],
        result["predictions"],
        target_names=train_dataset.classes,
        output_dict=True,
        zero_division=0,
    )

    class_rows = []

    for class_name in train_dataset.classes:
        class_result = report_dict[class_name]

        class_rows.append(
            {
                "class_name": class_name,
                "precision": class_result["precision"],
                "recall": class_result["recall"],
                "f1_score": class_result["f1-score"],
                "support": int(class_result["support"]),
            }
        )

    import csv

    csv_path = (
        BASE_DIR
        / "result"
        / f"classification_report_{EXPERIMENT}.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_name",
                "precision",
                "recall",
                "f1_score",
                "support",
            ],
        )

        writer.writeheader()
        writer.writerows(class_rows)

    print(f"클래스별 CSV 저장: {csv_path}")

    import pandas as pd

    prediction_rows = []

    for true_label, predicted_label in zip(
        result["labels"],
        result["predictions"],
    ):
        prediction_rows.append(
            {
                "label": true_label,
                "prediction": predicted_label,
                "class_name": train_dataset.classes[true_label],
                "predicted_class": train_dataset.classes[predicted_label],
            }
        )

    prediction_path = (
        BASE_DIR
        / "result"
        / f"predictions_{EXPERIMENT}.csv"
    )

    pd.DataFrame(prediction_rows).to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"예측 결과 저장: {prediction_path}")

    report_path = (
        BASE_DIR
        / "result"
        / f"classification_report_{EXPERIMENT}.txt"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(f"\n클래스별 보고서 저장: {report_path}")


if __name__ == "__main__":
    main()