from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
PLOT_DIR = RESULT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

baseline = pd.read_csv(RESULT_DIR / "history.csv")
label_smoothing = pd.read_csv(RESULT_DIR / "history_ls.csv")
depth2 = pd.read_csv(RESULT_DIR / "history_depth2.csv")


def save_validation_accuracy_comparison():
    plt.figure(figsize=(8, 5))

    plt.plot(
        baseline["epoch"],
        baseline["val_accuracy"],
        label="ViT-4 Baseline",
    )
    plt.plot(
        label_smoothing["epoch"],
        label_smoothing["val_accuracy"],
        label="ViT-4 + Label Smoothing",
    )
    plt.plot(
        depth2["epoch"],
        depth2["val_accuracy"],
        label="ViT-2",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = PLOT_DIR / "vit_val_accuracy_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"저장: {output_path}")


def save_validation_accuracy_zoom():
    baseline_zoom = baseline[baseline["epoch"] >= 14]
    label_smoothing_zoom = label_smoothing[
        label_smoothing["epoch"] >= 14
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        baseline_zoom["epoch"],
        baseline_zoom["val_accuracy"],
        marker="o",
        label="ViT-4 Baseline",
    )
    plt.plot(
        label_smoothing_zoom["epoch"],
        label_smoothing_zoom["val_accuracy"],
        marker="o",
        label="ViT-4 + Label Smoothing",
    )

    plt.xlim(14, 20)
    plt.ylim(55, 60)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy Zoom: Epoch 14-20")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = PLOT_DIR / "vit_val_accuracy_zoom.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"저장: {output_path}")


def save_validation_loss_comparison():
    plt.figure(figsize=(8, 5))

    plt.plot(
        baseline["epoch"],
        baseline["val_loss"],
        label="ViT-4 Baseline",
    )
    plt.plot(
        label_smoothing["epoch"],
        label_smoothing["val_loss"],
        label="ViT-4 + Label Smoothing",
    )
    plt.plot(
        depth2["epoch"],
        depth2["val_loss"],
        label="ViT-2",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = PLOT_DIR / "vit_val_loss_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"저장: {output_path}")


def save_test_accuracy_comparison():
    model_names = [
        "ViT-2",
        "ViT-4",
        "ViT-4 + LS",
    ]

    test_accuracies = [
        52.52,
        57.68,
        58.38,
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, test_accuracies)

    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Comparison")
    plt.ylim(50, 60)
    plt.grid(axis="y")

    for bar, value in zip(bars, test_accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            f"{value:.2f}%",
            ha="center",
        )

    plt.tight_layout()

    output_path = PLOT_DIR / "vit_test_accuracy_zoom.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"저장: {output_path}")


def save_macro_metrics_comparison():
    metrics = ["Precision", "Recall", "F1-score"]

    baseline_scores = [0.5776, 0.5768, 0.5716]
    label_smoothing_scores = [0.5766, 0.5838, 0.5755]
    depth2_scores = [0.5245, 0.5252, 0.5205]

    x = range(len(metrics))
    width = 0.25

    plt.figure(figsize=(9, 5))

    plt.bar(
        [value - width for value in x],
        depth2_scores,
        width=width,
        label="ViT-2",
    )
    plt.bar(
        x,
        baseline_scores,
        width=width,
        label="ViT-4",
    )
    plt.bar(
        [value + width for value in x],
        label_smoothing_scores,
        width=width,
        label="ViT-4 + LS",
    )

    plt.xticks(list(x), metrics)
    plt.ylabel("Score")
    plt.title("Macro Metrics Comparison")
    plt.ylim(0.50, 0.60)
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()

    output_path = PLOT_DIR / "vit_macro_metrics_zoom.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"저장: {output_path}")


def main():
    print("=" * 70)
    print("ViT 비교 그래프 생성")
    print("=" * 70)

    save_validation_accuracy_comparison()
    save_validation_accuracy_zoom()
    save_validation_loss_comparison()
    save_test_accuracy_comparison()
    save_macro_metrics_comparison()

    print("=" * 70)
    print("비교 그래프 생성 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()