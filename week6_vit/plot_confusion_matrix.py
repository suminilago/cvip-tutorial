from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
PLOT_DIR = RESULT_DIR / "plots"

EXPERIMENT = "label_smoothing"

PREDICTION_PATH = (
    RESULT_DIR
    / f"predictions_{EXPERIMENT}.csv"
)


def main():
    data = pd.read_csv(PREDICTION_PATH)

    labels = data["label"]
    predictions = data["prediction"]
    class_names = sorted(data["class_name"].unique())

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=range(len(class_names)),
        normalize="true",
    )

    figure, axis = plt.subplots(figsize=(14, 14))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=axis,
        xticks_rotation=90,
        values_format=".2f",
        colorbar=True,
    )

    plt.title(
        "Normalized Confusion Matrix\n"
        "ViT-4 + Label Smoothing"
    )
    plt.tight_layout()

    output_path = (
        PLOT_DIR
        / f"confusion_matrix_{EXPERIMENT}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()

    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()