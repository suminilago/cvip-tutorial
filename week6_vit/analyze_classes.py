from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
PLOT_DIR = RESULT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "label_smoothing"

REPORT_PATH = (
    RESULT_DIR
    / f"classification_report_{EXPERIMENT}.csv"
)


def main():
    report = pd.read_csv(REPORT_PATH)

    report = report.sort_values(
        by="f1_score",
        ascending=False,
    )

    best_classes = report.head(5)
    worst_classes = report.tail(5).sort_values(
        by="f1_score",
        ascending=True,
    )

    print("=" * 70)
    print(f"실험: {EXPERIMENT}")
    print("=" * 70)

    print("\nF1-score 상위 5개 클래스")
    print(
        best_classes[
            [
                "class_name",
                "precision",
                "recall",
                "f1_score",
            ]
        ].to_string(index=False)
    )

    print("\nF1-score 하위 5개 클래스")
    print(
        worst_classes[
            [
                "class_name",
                "precision",
                "recall",
                "f1_score",
            ]
        ].to_string(index=False)
    )

    output_table = pd.concat(
        [
            best_classes.assign(group="Top 5"),
            worst_classes.assign(group="Bottom 5"),
        ]
    )

    output_path = (
        RESULT_DIR
        / f"class_analysis_{EXPERIMENT}.csv"
    )

    output_table.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n분석 표 저장: {output_path}")

    plot_data = pd.concat(
        [
            worst_classes,
            best_classes.sort_values(
                by="f1_score",
                ascending=True,
            ),
        ]
    )

    plt.figure(figsize=(9, 6))
    bars = plt.barh(
        plot_data["class_name"],
        plot_data["f1_score"],
    )

    plt.xlabel("F1-score")
    plt.ylabel("Class")
    plt.title(
        f"Best and Worst Classes: {EXPERIMENT}"
    )
    plt.xlim(0, 1)
    plt.grid(axis="x")

    for bar, value in zip(
        bars,
        plot_data["f1_score"],
    ):
        plt.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
        )

    plt.tight_layout()

    graph_path = (
        PLOT_DIR
        / f"class_f1_{EXPERIMENT}.png"
    )

    plt.savefig(
        graph_path,
        dpi=200,
    )
    plt.close()

    print(f"그래프 저장: {graph_path}")


if __name__ == "__main__":
    main()