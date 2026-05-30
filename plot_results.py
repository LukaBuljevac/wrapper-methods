from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "summary_results.csv"


def method_label(row):
    if row["method"] == "baseline":
        return "baseline"
    return f"{row['method']}_{row['selection_metric']}"


def plot_bar(df: pd.DataFrame, dataset: str, classifier: str, column: str, ylabel: str, filename_prefix: str):
    subset = df[(df["dataset"] == dataset) & (df["classifier"] == classifier)].copy()
    subset["label"] = subset.apply(method_label, axis=1)

    plt.figure(figsize=(11, 5))
    plt.bar(subset["label"], subset[column])
    plt.title(f"{ylabel} - {dataset} - {classifier}")
    plt.xlabel("Metoda")
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    out = RESULTS_DIR / f"{filename_prefix}_{dataset}_{classifier}.png"
    plt.savefig(out, dpi=150)
    plt.close()


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("Nema results/summary_results.csv. Prvo pokreni run_experiments.py")

    df = pd.read_csv(SUMMARY_PATH)
    RESULTS_DIR.mkdir(exist_ok=True)

    for dataset in df["dataset"].unique():
        for classifier in df["classifier"].unique():
            if df[(df["dataset"] == dataset) & (df["classifier"] == classifier)].empty:
                continue
            plot_bar(df, dataset, classifier, "accuracy_mean", "Accuracy mean", "accuracy")
            plot_bar(df, dataset, classifier, "macro_f1_mean", "Macro-F1 mean", "macro_f1")
            plot_bar(df, dataset, classifier, "n_selected_mean", "Average number of selected features", "features")
            plot_bar(df, dataset, classifier, "time_mean_sec", "Average time [s]", "time")

    print("[INFO] Grafovi su spremljeni u results/.")


if __name__ == "__main__":
    main()
