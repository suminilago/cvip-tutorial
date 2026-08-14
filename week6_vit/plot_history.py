from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

history_path = BASE_DIR / "result" / "history.csv"
plot_dir = BASE_DIR / "result" / "plots"

plot_dir.mkdir(exist_ok=True)

history = pd.read_csv(history_path)

# Loss 그래프
plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Vision Transformer Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "vit_loss.png")
plt.close()

# Accuracy 그래프
plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_accuracy"], label="Train Accuracy")
plt.plot(history["epoch"], history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Vision Transformer Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_dir / "vit_accuracy.png")
plt.close()

print("=" * 60)
print("그래프 저장 완료")
print("=" * 60)
print(f"Loss 그래프 : {plot_dir / 'vit_loss.png'}")
print(f"Accuracy 그래프 : {plot_dir / 'vit_accuracy.png'}")