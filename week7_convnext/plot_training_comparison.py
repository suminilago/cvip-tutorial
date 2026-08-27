from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

REPORT_DIR = BASE_DIR / "result" / "reports"
PLOT_DIR = BASE_DIR / "result" / "plots"

BASELINE_HISTORY = (
    REPORT_DIR / "convnext_baseline_history.csv"
)

KERNEL3_HISTORY = (
    REPORT_DIR / "convnext_kernel3_history.csv"
)


def plot_accuracy(df7, df3):
    plt.figure(figsize=(10, 6))

    plt.plot(
        df7["epoch"],
        df7["val_acc"],
        marker="o",
        label="7x7 Validation Accuracy",
    )

    plt.plot(
        df3["epoch"],
        df3["val_acc"],
        marker="o",
        label="3x3 Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title(
        "ConvNeXt Validation Accuracy: 7x7 vs 3x3"
    )

    plt.legend()
    plt.grid()

    save_path = (
        PLOT_DIR
        / "convnext_validation_accuracy_comparison.png"
    )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved:", save_path)


def plot_loss(df7, df3):
    plt.figure(figsize=(10, 6))

    plt.plot(
        df7["epoch"],
        df7["val_loss"],
        marker="o",
        label="7x7 Validation Loss",
    )

    plt.plot(
        df3["epoch"],
        df3["val_loss"],
        marker="o",
        label="3x3 Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(
        "ConvNeXt Validation Loss: 7x7 vs 3x3"
    )

    plt.legend()
    plt.grid()

    save_path = (
        PLOT_DIR
        / "convnext_validation_loss_comparison.png"
    )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("Saved:", save_path)


def main():
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df7 = pd.read_csv(BASELINE_HISTORY)
    df3 = pd.read_csv(KERNEL3_HISTORY)

    print("=" * 70)
    print("ConvNeXt Training Comparison")
    print("=" * 70)

    print()
    print(
        "7x7 Best Validation Accuracy:",
        f"{df7['val_acc'].max():.2f}%",
    )

    print(
        "3x3 Best Validation Accuracy:",
        f"{df3['val_acc'].max():.2f}%",
    )

    print()

    plot_accuracy(df7, df3)
    plot_loss(df7, df3)

    print()
    print("=" * 70)
    print("Completed")
    print("=" * 70)


if __name__ == "__main__":
    main()