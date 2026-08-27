from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

HISTORY_PATH = (
    BASE_DIR
    / "result"
    / "reports"
    / "convnext_baseline_history.csv"
)

PLOT_DIR = BASE_DIR / "result" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_loss(df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        df["epoch"],
        df["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ConvNeXt Baseline Loss")
    plt.legend()
    plt.grid()

    save_path = (
        PLOT_DIR
        / "convnext_baseline_loss.png"
    )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200,
    )

    plt.close()

    print("Saved:", save_path)


def plot_accuracy(df):
    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["train_acc"],
        label="Train Accuracy",
    )

    plt.plot(
        df["epoch"],
        df["val_acc"],
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("ConvNeXt Baseline Accuracy")
    plt.legend()
    plt.grid()

    save_path = (
        PLOT_DIR
        / "convnext_baseline_accuracy.png"
    )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200,
    )

    plt.close()

    print("Saved:", save_path)


def main():
    df = pd.read_csv(HISTORY_PATH)

    print(df)

    plot_loss(df)
    plot_accuracy(df)


if __name__ == "__main__":
    main()