from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SelectionResult:
    selected_features: List[int]
    validation_score: float
    n_evaluated_subsets: int
    selection_time_sec: float


def _make_model(classifier):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clone(classifier)),
    ])


def _score_model(y_true, y_pred, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    raise ValueError("metric mora biti 'accuracy' ili 'macro_f1'")


def evaluate_subset(
    classifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    subset: List[int],
    metric: str,
) -> float:
    """Train on X_train[:, subset], evaluate on X_val[:, subset]."""
    model = _make_model(classifier)
    model.fit(X_train[:, subset], y_train)
    y_pred = model.predict(X_val[:, subset])
    return _score_model(y_val, y_pred, metric)


def sfs_select(
    classifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str,
    tol: float = 0.0,
    max_steps: Optional[int] = None,
) -> SelectionResult:
    """
    Sequential Forward Selection.

    Start with no features. In each step, try adding every remaining feature.
    Add the feature that gives the best validation score. Stop when no candidate
    improves the current best score.
    """
    start = time.perf_counter()
    n_features = X_train.shape[1]
    remaining = list(range(n_features))
    current_subset: List[int] = []
    best_score = -np.inf
    best_subset: List[int] = []
    evaluated = 0
    steps = 0

    while remaining:
        candidate_results: List[Tuple[float, List[int], int]] = []

        for feat in remaining:
            candidate_subset = current_subset + [feat]
            score = evaluate_subset(classifier, X_train, y_train, X_val, y_val, candidate_subset, metric)
            evaluated += 1
            candidate_results.append((score, candidate_subset, feat))

        step_best_score, step_best_subset, step_best_feat = max(candidate_results, key=lambda item: item[0])

        if step_best_score > best_score + tol:
            current_subset = step_best_subset
            remaining.remove(step_best_feat)
            best_score = step_best_score
            best_subset = current_subset.copy()
            steps += 1
        else:
            break

        if max_steps is not None and steps >= max_steps:
            break

    return SelectionResult(
        selected_features=best_subset,
        validation_score=float(best_score),
        n_evaluated_subsets=evaluated,
        selection_time_sec=time.perf_counter() - start,
    )


def sbs_select(
    classifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str,
    tol: float = 0.0,
    max_steps: Optional[int] = None,
) -> SelectionResult:
    """
    Sequential Backward Selection.

    Start with all features. In each step, try removing every current feature.
    Remove the feature whose removal gives the best validation score. Stop when
    no removal improves the current best score.
    """
    start = time.perf_counter()
    n_features = X_train.shape[1]
    current_subset = list(range(n_features))
    best_score = evaluate_subset(classifier, X_train, y_train, X_val, y_val, current_subset, metric)
    best_subset = current_subset.copy()
    evaluated = 1
    steps = 0

    while len(current_subset) > 1:
        candidate_results: List[Tuple[float, List[int], int]] = []

        for feat in current_subset:
            candidate_subset = [f for f in current_subset if f != feat]
            score = evaluate_subset(classifier, X_train, y_train, X_val, y_val, candidate_subset, metric)
            evaluated += 1
            candidate_results.append((score, candidate_subset, feat))

        step_best_score, step_best_subset, removed_feat = max(candidate_results, key=lambda item: item[0])

        if step_best_score > best_score + tol:
            current_subset = step_best_subset
            best_score = step_best_score
            best_subset = current_subset.copy()
            steps += 1
        else:
            break

        if max_steps is not None and steps >= max_steps:
            break

    return SelectionResult(
        selected_features=best_subset,
        validation_score=float(best_score),
        n_evaluated_subsets=evaluated,
        selection_time_sec=time.perf_counter() - start,
    )


def random_select(
    classifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str,
    n_trials: int,
    random_state: int,
    min_features: int = 1,
    max_features: Optional[int] = None,
) -> SelectionResult:
    """
    Random wrapper selection.

    Generate n_trials random feature subsets. Each subset is evaluated by
    training the classifier on train and measuring validation performance.
    The best validation subset is returned.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(random_state)
    n_features = X_train.shape[1]

    if max_features is None:
        max_features = n_features

    min_features = max(1, min(min_features, n_features))
    max_features = max(min_features, min(max_features, n_features))

    best_score = -np.inf
    best_subset: List[int] = []
    evaluated = 0
    seen = set()
    max_unique_attempts = max(n_trials * 10, n_trials + 100)
    attempts = 0

    while evaluated < n_trials and attempts < max_unique_attempts:
        attempts += 1
        k = int(rng.integers(min_features, max_features + 1))
        subset = sorted(rng.choice(n_features, size=k, replace=False).tolist())
        subset_key = tuple(subset)

        if subset_key in seen:
            continue

        seen.add(subset_key)
        score = evaluate_subset(classifier, X_train, y_train, X_val, y_val, subset, metric)
        evaluated += 1

        if score > best_score:
            best_score = score
            best_subset = subset

    return SelectionResult(
        selected_features=best_subset,
        validation_score=float(best_score),
        n_evaluated_subsets=evaluated,
        selection_time_sec=time.perf_counter() - start,
    )
