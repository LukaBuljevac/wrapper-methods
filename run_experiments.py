from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from feature_selectors import random_select, sbs_select, sfs_select


RESULTS_DIR = Path("results")
DATASETS_DIR = Path("datasets")


def build_classifiers() -> Dict[str, object]:
    return {
        "knn5": KNeighborsClassifier(n_neighbors=5),
        "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale"),
    }


def load_csv_dataset(path: Path, target_column: str | None = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)

    if target_column is None:
        target_column = df.columns[-1]

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' ne postoji u {path.name}")

    y_raw = df[target_column]
    X_df = df.drop(columns=[target_column])

    # Remove obvious ID/name columns if they are non-numeric and too unique.
    for col in list(X_df.columns):
        if X_df[col].dtype == object and X_df[col].nunique() > 0.8 * len(X_df):
            X_df = X_df.drop(columns=[col])

    # Convert categorical feature columns into numeric dummy variables if needed.
    X_df = pd.get_dummies(X_df, drop_first=False)

    # Convert missing/infinite values into column medians.
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    # Missing vrijednosti se zasad popunjavaju medijanom.
    # Ako bude potrebno, ovo se lako može zamijeniti s dropna().

    X = X_df.to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(y_raw.astype(str))

    if len(np.unique(y)) != 2:
        raise ValueError(
            f"Dataset {path.name} nije binaran. Broj klasa: {len(np.unique(y))}. "
            "Za ovaj eksperiment koristi strogo binarne skupove."
        )

    return X, y, list(X_df.columns)


def load_all_datasets(datasets_dir: Path, dataset_names: List[str], target_column: str | None) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    if not datasets_dir.exists():
        raise FileNotFoundError(
            f"Mapa {datasets_dir} ne postoji. Prvo pokreni prepare_datasets.py ili ručno dodaj CSV datoteke."
        )

    csv_files = sorted(datasets_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"U mapi {datasets_dir} nema CSV datoteka.")

    selected_names = set(dataset_names) if dataset_names else None
    datasets = {}

    for csv_path in csv_files:
        name = csv_path.stem
        if selected_names is not None and name not in selected_names:
            continue
        X, y, feature_names = load_csv_dataset(csv_path, target_column=target_column)
        datasets[name] = (X, y, feature_names)

    if not datasets:
        raise ValueError("Nijedan dataset nije učitan. Provjeri --datasets i nazive CSV datoteka.")

    return datasets

def create_dataset_summary(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    output_dir: Path,
):
    rows = []

    for dataset_name, (X, y, feature_names) in datasets.items():
        unique_classes, class_counts = np.unique(y, return_counts=True)

        if len(class_counts) == 2:
            ir = max(class_counts) / min(class_counts)
        else:
            ir = np.nan

        rows.append({
            "dataset": dataset_name,
            "n_instances": X.shape[0],
            "n_features": X.shape[1],
            "class_0_count": class_counts[0],
            "class_1_count": class_counts[1],
            "IR": ir,
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "dataset_summary.csv", index=False)

    return summary_df

def split_train_val_test(X: np.ndarray, y: np.ndarray, random_state: int):
    # First split: 75% train_val, 25% test.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )

    # Second split: from 75% train_val, take 1/3 as validation -> 25% total.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=1 / 3,
        stratify=y_train_val,
        random_state=random_state + 10000,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_final_model(classifier):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clone(classifier)),
    ])


def evaluate_on_test(classifier, X_train_final, y_train_final, X_test, y_test, selected_features: List[int]):
    model = make_final_model(classifier)
    model.fit(X_train_final[:, selected_features], y_train_final)
    y_pred = model.predict(X_test[:, selected_features])
    return {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def resolve_random_trials(value: str, n_features: int) -> int:
    if value == "n2":
        return n_features * n_features
    return int(value)


def run_one_split(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    classifier_name: str,
    classifier,
    split_id: int,
    random_trials_arg: str,
    random_min_features: int,
    max_steps: int | None,
    base_seed: int,
):
    random_state = base_seed + split_id
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, random_state)
    X_train_final = np.vstack([X_train, X_val])
    y_train_final = np.concatenate([y_train, y_val])
    n_features = X.shape[1]

    rows = []

    # Baseline: no validation selection needed. Train on train+validation, test on test.
    start_time = time.perf_counter()
    all_features = list(range(n_features))
    test_scores = evaluate_on_test(classifier, X_train_final, y_train_final, X_test, y_test, all_features)
    rows.append({
        "dataset": dataset_name,
        "classifier": classifier_name,
        "split": split_id,
        "random_state": random_state,
        "method": "baseline",
        "selection_metric": "none",
        "validation_score": np.nan,
        "test_accuracy": test_scores["test_accuracy"],
        "test_macro_f1": test_scores["test_macro_f1"],
        "n_selected": n_features,
        "selected_features": ",".join(feature_names),
        "n_evaluated_subsets": 0,
        "time_sec": time.perf_counter() - start_time,
    })

    selection_jobs = [
        ("sfs", "accuracy"),
        ("sfs", "macro_f1"),
        ("sbs", "accuracy"),
        ("sbs", "macro_f1"),
        ("random", "accuracy"),
        ("random", "macro_f1"),
    ]

    for method, metric in selection_jobs:
        start_time = time.perf_counter()
        if method == "sfs":
            result = sfs_select(classifier, X_train, y_train, X_val, y_val, metric=metric, max_steps=max_steps)
        elif method == "sbs":
            result = sbs_select(classifier, X_train, y_train, X_val, y_val, metric=metric, max_steps=max_steps)
        elif method == "random":
            n_trials = resolve_random_trials(random_trials_arg, n_features)
            result = random_select(
                classifier,
                X_train,
                y_train,
                X_val,
                y_val,
                metric=metric,
                n_trials=n_trials,
                random_state=random_state + 20000,
                min_features=random_min_features,
                max_features=n_features,
            )
        else:
            raise ValueError(method)

        selected = result.selected_features
        if not selected:
            selected = all_features

        test_scores = evaluate_on_test(classifier, X_train_final, y_train_final, X_test, y_test, selected)
        selected_names = [feature_names[i] for i in selected]

        rows.append({
            "dataset": dataset_name,
            "classifier": classifier_name,
            "split": split_id,
            "method": method,
            "selection_metric": metric,
            "validation_score": result.validation_score,
            "test_accuracy": test_scores["test_accuracy"],
            "test_macro_f1": test_scores["test_macro_f1"],
            "n_selected": len(selected),
            "selected_features": ",".join(selected_names),
            "n_evaluated_subsets": result.n_evaluated_subsets,
            "time_sec": time.perf_counter() - start_time,
        })

    return rows


def summarize_results(all_results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "classifier", "method", "selection_metric"]
    return (
        all_results
        .groupby(group_cols, as_index=False)
        .agg(
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_std=("test_accuracy", "std"),
            accuracy_min=("test_accuracy", "min"),
            accuracy_max=("test_accuracy", "max"),
            macro_f1_mean=("test_macro_f1", "mean"),
            macro_f1_std=("test_macro_f1", "std"),
            macro_f1_min=("test_macro_f1", "min"),
            macro_f1_max=("test_macro_f1", "max"),
            n_selected_mean=("n_selected", "mean"),
            n_selected_std=("n_selected", "std"),
            n_selected_min=("n_selected", "min"),
            n_selected_max=("n_selected", "max"),
            time_mean_sec=("time_sec", "mean"),
            time_std_sec=("time_sec", "std"),
            subsets_mean=("n_evaluated_subsets", "mean"),
        )
        .sort_values(group_cols)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Wrapper feature selection experiments: SFS, SBS, Random.")
    parser.add_argument("--datasets-dir", default="datasets", help="Mapa s CSV datasetovima.")
    parser.add_argument("--datasets", default="", help="CSV nazivi bez .csv odvojeni zarezom. Prazno = svi CSV datasetovi.")
    parser.add_argument("--target-column", default=None, help="Naziv target stupca. Ako nije zadan, uzima se zadnji stupac.")
    parser.add_argument("--classifiers", default="knn5,svm_rbf", help="knn5,svm_rbf ili oba odvojena zarezom.")
    parser.add_argument("--splits", type=int, default=30, help="Broj različitih stratificiranih train/val/test podjela.")
    parser.add_argument("--random-trials", default="n2", help="Broj random podskupova. 'n2' znači n_features*n_features.")
    parser.add_argument("--random-min-features", type=int, default=1, help="Minimalan broj featurea u random podskupu.")
    parser.add_argument("--max-steps", type=int, default=None, help="Opcionalni limit SFS/SBS koraka. Default None = bez limita.")
    parser.add_argument("--seed", type=int, default=42, help="Bazni random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    datasets = load_all_datasets(Path(args.datasets_dir), dataset_names, args.target_column)

    dataset_summary = create_dataset_summary(datasets, RESULTS_DIR)

    print("\n[INFO] Sažetak datasetova:")
    print(dataset_summary.round(4).to_string(index=False))

    available_classifiers = build_classifiers()
    requested_classifiers = [x.strip() for x in args.classifiers.split(",") if x.strip()]
    classifiers = {name: available_classifiers[name] for name in requested_classifiers}

    print("[INFO] Učitani datasetovi:")
    for name, (X, y, _) in datasets.items():
        unique, counts = np.unique(y, return_counts=True)
        print(f"  - {name}: n={X.shape[0]}, d={X.shape[1]}, klase={dict(zip(unique, counts))}")

    print("[INFO] Klasifikatori:", ", ".join(classifiers.keys()))
    print(f"[INFO] Splitovi: {args.splits} | Podjela: 50/25/25 | Stratify: DA")
    print(f"[INFO] Random trials: {args.random_trials}")

    all_rows = []

    for dataset_name, (X, y, feature_names) in datasets.items():
        for classifier_name, classifier in classifiers.items():
            for split_id in range(args.splits):
                print(f"[INFO] dataset={dataset_name} | classifier={classifier_name} | split={split_id + 1}/{args.splits}")
                split_rows = run_one_split(
                    dataset_name=dataset_name,
                    X=X,
                    y=y,
                    feature_names=feature_names,
                    classifier_name=classifier_name,
                    classifier=classifier,
                    split_id=split_id,
                    random_trials_arg=args.random_trials,
                    random_min_features=args.random_min_features,
                    max_steps=args.max_steps,
                    base_seed=args.seed,
                )
                all_rows.extend(split_rows)

                # Save partial results after every split, useful for long runs.
                pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "all_results_partial.csv", index=False)

    all_results = pd.DataFrame(all_rows)
    all_results.to_csv(RESULTS_DIR / "all_results.csv", index=False)

    summary = summarize_results(all_results)
    summary.to_csv(RESULTS_DIR / "summary_results.csv", index=False)

    print("\n=== SUMMARY ===")
    print(summary.round(4).to_string(index=False))
    print("\n[INFO] Spremljeno u mapu results/.")


if __name__ == "__main__":
    main()
