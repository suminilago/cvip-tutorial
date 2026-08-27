from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR / "result" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    metrics = [
        "Test Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
    ]

    kernel7 = [
        72.60,
        72.39,
        72.60,
        72.00,
    ]

    kernel3 = [
        75.77,
        75.64,
        75.77,
        75.01,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))

    bars7 = plt.bar(
        x - width / 2,
        kernel7,
        width,
        label="7x7",
    )

    bars3 = plt.bar(
        x + width / 2,
        kernel3,
        width,
        label="3x3",
    )

    plt.ylabel("Score (%)")
    plt.xlabel("Metric")

    plt.title(
        "ConvNeXt Performance Comparison: 7x7 vs 3x3"
    )

    plt.xticks(
        x,
        metrics,
    )

    plt.ylim(65, 80)

    plt.legend()

    for bars in [bars7, bars3]:
        for bar in bars:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.15,
                f"{height:.2f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()

    save_path = (
        PLOT_DIR
        / "convnext_metric_comparison.png"
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print("Saved:", save_path)


if __name__ == "__main__":
    main()