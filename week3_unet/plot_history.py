from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

BASELINE_HISTORY = (
    BASE_DIR / "result" / "unet_history.csv"
)

BCE_DICE_HISTORY = (
    BASE_DIR / "result" / "unet_bce_dice_history.csv"
)

OUTPUT_DIR = BASE_DIR / "result" / "plots"


def main():
    baseline = pd.read_csv(BASELINE_HISTORY)
    bce_dice = pd.read_csv(BCE_DICE_HISTORY)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Loss 비교
    plt.figure(figsize=(9, 6))

    plt.plot(
        baseline["epoch"],
        baseline["val_loss"],
        label="Baseline Val Loss",
    )

    plt.plot(
        bce_dice["epoch"],
        bce_dice["val_loss"],
        label="BCE + Dice Val Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison")
    plt.legend()
    plt.grid(True)

    loss_path = OUTPUT_DIR / "validation_loss_comparison.png"

    plt.savefig(
        loss_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # 2. Dice 비교
    plt.figure(figsize=(9, 6))

    plt.plot(
        baseline["epoch"],
        baseline["val_dice"],
        label="Baseline Val Dice",
    )

    plt.plot(
        bce_dice["epoch"],
        bce_dice["val_dice"],
        label="BCE + Dice Val Dice",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice")
    plt.title("Validation Dice Comparison")
    plt.legend()
    plt.grid(True)

    dice_path = OUTPUT_DIR / "validation_dice_comparison.png"

    plt.savefig(
        dice_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # 3. IoU 비교
    plt.figure(figsize=(9, 6))

    plt.plot(
        baseline["epoch"],
        baseline["val_iou"],
        label="Baseline Val IoU",
    )

    plt.plot(
        bce_dice["epoch"],
        bce_dice["val_iou"],
        label="BCE + Dice Val IoU",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Validation IoU")
    plt.title("Validation IoU Comparison")
    plt.legend()
    plt.grid(True)

    iou_path = OUTPUT_DIR / "validation_iou_comparison.png"

    plt.savefig(
        iou_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    baseline_best = baseline.loc[
        baseline["val_dice"].idxmax()
    ]

    bce_dice_best = bce_dice.loc[
        bce_dice["val_dice"].idxmax()
    ]

    print("=" * 70)
    print("실험 비교")
    print("=" * 70)

    print(
        "Baseline "
        f"| Best Epoch: {int(baseline_best['epoch'])} "
        f"| Val Dice: {baseline_best['val_dice']:.4f} "
        f"| Val IoU: {baseline_best['val_iou']:.4f}"
    )

    print(
        "BCE + Dice "
        f"| Best Epoch: {int(bce_dice_best['epoch'])} "
        f"| Val Dice: {bce_dice_best['val_dice']:.4f} "
        f"| Val IoU: {bce_dice_best['val_iou']:.4f}"
    )

    print(f"\n그래프 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()