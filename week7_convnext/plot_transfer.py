import csv
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

HISTORY_PATH = (
    BASE_DIR
    / "result"
    / "reports"
    / "convnext_tiny_transfer_history.csv"
)

OUTPUT_DIR = BASE_DIR / "result" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


epochs = []
train_acc = []
val_acc = []
train_loss = []
val_loss = []


with open(HISTORY_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        epochs.append(int(row["epoch"]))
        train_acc.append(float(row["train_acc"]))
        val_acc.append(float(row["val_acc"]))
        train_loss.append(float(row["train_loss"]))
        val_loss.append(float(row["val_loss"]))


# ============================================================
# Accuracy Curve
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    train_acc,
    marker="o",
    label="Train Accuracy",
)

plt.plot(
    epochs,
    val_acc,
    marker="o",
    label="Validation Accuracy",
)

# Epoch 1~3 classifier / Epoch 4~10 fine-tuning
plt.axvline(
    x=3.5,
    linestyle="--",
    label="Start Fine-tuning",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Pretrained ConvNeXt-Tiny Training Accuracy")

plt.xticks(epochs)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

accuracy_path = (
    OUTPUT_DIR
    / "convnext_transfer_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
)

plt.close()


# ============================================================
# Loss Curve
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Train Loss",
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss",
)

plt.axvline(
    x=3.5,
    linestyle="--",
    label="Start Fine-tuning",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Pretrained ConvNeXt-Tiny Training Loss")

plt.xticks(epochs)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

loss_path = (
    OUTPUT_DIR
    / "convnext_transfer_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
)

plt.close()


# ============================================================
# Test Accuracy Comparison
# ============================================================

models = [
    "7x7\nBaseline",
    "3x3\nBaseline",
    "Label\nSmoothing",
    "Longer\nTraining",
    "Pretrained\nConvNeXt-Tiny",
]

test_accuracy = [
    72.60,
    75.77,
    76.60,
    78.23,
    88.38,
]


plt.figure(figsize=(10, 5))

bars = plt.bar(
    models,
    test_accuracy,
)

plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy Comparison")
plt.ylim(65, 92)

for bar, value in zip(bars, test_accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.4,
        f"{value:.2f}%",
        ha="center",
    )

plt.tight_layout()

comparison_path = (
    OUTPUT_DIR
    / "convnext_final_accuracy_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=300,
)

plt.close()


print("=" * 70)
print("Transfer Learning plots saved")
print("=" * 70)
print(accuracy_path)
print(loss_path)
print(comparison_path)