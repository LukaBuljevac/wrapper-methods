from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "summary_results.csv"


def method_label(row):
    if row["method"] == "baseline":
        return "baseline"
    return f"{row['method']}_{row['selection_metric']}"


def plot_metric_for_classifier(
    df: pd.DataFrame,
    classifier: str,
    metric_column: str,
    selection_metric: str,
    title: str,
    output_name: str,
):
    subset = df[df["classifier"] == classifier].copy()
    subset["method_full"] = subset.apply(method_label, axis=1)

    wanted_methods = [
        "baseline",
        f"sfs_{selection_metric}",
        f"sbs_{selection_metric}",
        f"random_{selection_metric}",
    ]

    markers = {
        "baseline": "x",
        f"sfs_{selection_metric}": "^",
        f"sbs_{selection_metric}": "s",
        f"random_{selection_metric}": "o",
    }

    labels = {
        "baseline": "Baseline",
        f"sfs_{selection_metric}": "SFS",
        f"sbs_{selection_metric}": "SBS",
        f"random_{selection_metric}": "Random",
    }

    datasets = sorted(subset["dataset"].unique())

    plt.figure(figsize=(12, 6))

    for method in wanted_methods:
        method_df = subset[subset["method_full"] == method]

        values = []
        for dataset in datasets:
            row = method_df[method_df["dataset"] == dataset]

            if row.empty:
                values.append(None)
            else:
                values.append(row.iloc[0][metric_column])

        plt.plot(
            datasets,
            values,
            marker=markers[method],
            label=labels[method],
        )

    plt.title(title)
    plt.xlabel("Dataset")
    plt.ylabel(metric_column)
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    out = RESULTS_DIR / output_name
    plt.savefig(out, dpi=150)
    plt.close()


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Nema results/summary_results.csv. Prvo pokreni run_experiments.py"
        )

    df = pd.read_csv(SUMMARY_PATH)

    for classifier in sorted(df["classifier"].unique()):
        plot_metric_for_classifier(
            df=df,
            classifier=classifier,
            metric_column="accuracy_mean",
            selection_metric="accuracy",
            title=f"Accuracy po datasetima - {classifier}",
            output_name=f"line_accuracy_{classifier}.png",
        )

        plot_metric_for_classifier(
            df=df,
            classifier=classifier,
            metric_column="macro_f1_mean",
            selection_metric="macro_f1",
            title=f"Macro-F1 po datasetima - {classifier}",
            output_name=f"line_macro_f1_{classifier}.png",
        )

    print("[INFO] Sažeti linijski grafovi spremljeni su u results/.")


if __name__ == "__main__":
    main()