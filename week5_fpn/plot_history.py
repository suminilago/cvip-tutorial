from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

FPN_HISTORY = BASE_DIR / "result" / "fpn_history.csv"
OUTPUT_DIR = BASE_DIR / "result" / "plots"


def save_curve(
    history: pd.DataFrame,
    train_column: str,
    val_column: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    plt.figure(figsize=(9, 6))

    plt.plot(
        history["epoch"],
        history[train_column],
        label=f"Train {ylabel}",
    )

    plt.plot(
        history["epoch"],
        history[val_column],
        label=f"Validation {ylabel}",
    )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    save_path = OUTPUT_DIR / filename

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"저장: {save_path.name}")


def main():
    if not FPN_HISTORY.exists():
        raise FileNotFoundError(
            f"학습 기록을 찾을 수 없습니다: {FPN_HISTORY}"
        )

    history = pd.read_csv(FPN_HISTORY)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_curve(
        history=history,
        train_column="train_loss",
        val_column="val_loss",
        ylabel="Loss",
        title="FPN Training and Validation Loss",
        filename="fpn_loss.png",
    )

    save_curve(
        history=history,
        train_column="train_dice",
        val_column="val_dice",
        ylabel="Dice",
        title="FPN Training and Validation Dice",
        filename="fpn_dice.png",
    )

    save_curve(
        history=history,
        train_column="train_iou",
        val_column="val_iou",
        ylabel="IoU",
        title="FPN Training and Validation IoU",
        filename="fpn_iou.png",
    )

    best_row = history.loc[
        history["val_dice"].idxmax()
    ]

    print("=" * 70)
    print("FPN 최고 성능")
    print("=" * 70)
    print(f"Best Epoch: {int(best_row['epoch'])}")
    print(
        f"Validation Dice: "
        f"{best_row['val_dice']:.4f}"
    )
    print(
        f"Validation IoU: "
        f"{best_row['val_iou']:.4f}"
    )
    print(f"그래프 저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()