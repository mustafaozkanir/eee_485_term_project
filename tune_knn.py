"""
tune_knn.py
-----------
k-fold stratified cross-validation to tune k for the custom kNN classifier.
"""

import numpy as np
import matplotlib.pyplot as plt

from knn import KNearestNeighbors
from data_prep import prepare_data

# ---------------------------------------------------------------------------
# 1. Cross-Validation Splitter (stratified)
# ---------------------------------------------------------------------------

def get_cv_splits(X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_seed: int = 42):
    """
    Stratified k-fold cross-validation splitter implemented from scratch.

    Stratification ensures each fold mirrors the class distribution of the
    full dataset, which matters here because classes are imbalanced (~78/22).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    n_splits : int
        Number of folds.
    random_seed : int

    Yields
    ------
    (train_indices, val_indices) : tuple of 1-D integer arrays
        One pair per fold.
    """
    rng = np.random.default_rng(random_seed)
    n_samples = len(y)
    classes = np.unique(y)

    fold_assignment = np.empty(n_samples, dtype=np.int32)

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        fold_ids = np.arange(len(cls_idx)) % n_splits
        fold_assignment[cls_idx] = fold_ids

    for fold in range(n_splits):
        val_idx   = np.where(fold_assignment == fold)[0]
        train_idx = np.where(fold_assignment != fold)[0]
        yield train_idx, val_idx


# ---------------------------------------------------------------------------
# 2. Evaluation Metrics
# ---------------------------------------------------------------------------

def calculate_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification error = 1 - accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(1.0 - np.mean(y_true == y_pred))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute precision, recall, and F1 from scratch.
    Returns a dict with keys: precision, recall, f1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)

    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# 3. Cross-Validation Loop
# ---------------------------------------------------------------------------

def evaluate_knn_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_values: list,
    n_splits: int = 5,
    train_subsample: int = 2_000,
    random_seed: int = 42,
) -> tuple[dict, dict, dict, dict]:
    """
    Evaluate kNN for each k using n_splits-fold stratified cross-validation.

    Returns
    -------
    avg_train_errors : dict  {k: mean_train_error}
    avg_val_errors   : dict  {k: mean_val_error}
    avg_train_metrics: dict  {k: {precision, recall, f1}}
    avg_val_metrics  : dict  {k: {precision, recall, f1}}
    """
    rng = np.random.default_rng(random_seed)

    splits = list(get_cv_splits(X_train, y_train, n_splits=n_splits, random_seed=random_seed))

    avg_train_errors  = {}
    avg_val_errors    = {}
    avg_train_metrics = {}
    avg_val_metrics   = {}

    for k in k_values:
        print(f"\n  k = {k}")
        fold_train_errors  = []
        fold_val_errors    = []
        fold_train_metrics = {"precision": [], "recall": [], "f1": []}
        fold_val_metrics   = {"precision": [], "recall": [], "f1": []}

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            X_cv_train, y_cv_train = X_train[tr_idx], y_train[tr_idx]
            X_cv_val,   y_cv_val   = X_train[val_idx], y_train[val_idx]

            model = KNearestNeighbors(k=k)
            model.fit(X_cv_train, y_cv_train)

            # --- validation ---
            val_preds = model.predict(X_cv_val)
            val_err   = calculate_error(y_cv_val, val_preds)
            val_mets  = calculate_metrics(y_cv_val, val_preds)
            fold_val_errors.append(val_err)
            for m in fold_val_metrics:
                fold_val_metrics[m].append(val_mets[m])

            # --- training (stratified subsample) ---
            if train_subsample is not None and train_subsample < len(y_cv_train):
                sub_idx = _stratified_subsample(y_cv_train, train_subsample, rng)
            else:
                sub_idx = np.arange(len(y_cv_train))

            train_preds = model.predict(X_cv_train[sub_idx])
            train_err   = calculate_error(y_cv_train[sub_idx], train_preds)
            train_mets  = calculate_metrics(y_cv_train[sub_idx], train_preds)
            fold_train_errors.append(train_err)
            for m in fold_train_metrics:
                fold_train_metrics[m].append(train_mets[m])

            print(
                f"    Fold {fold_idx + 1}/{n_splits} — "
                f"train_err={train_err:.4f}  val_err={val_err:.4f}  "
                f"val_prec={val_mets['precision']:.4f}  "
                f"val_rec={val_mets['recall']:.4f}  "
                f"val_f1={val_mets['f1']:.4f}"
            )

        avg_train_errors[k]  = float(np.mean(fold_train_errors))
        avg_val_errors[k]    = float(np.mean(fold_val_errors))
        avg_train_metrics[k] = {m: float(np.mean(fold_train_metrics[m])) for m in fold_train_metrics}
        avg_val_metrics[k]   = {m: float(np.mean(fold_val_metrics[m]))   for m in fold_val_metrics}

        print(
            f"    → avg val_err={avg_val_errors[k]:.4f}  "
            f"avg val_prec={avg_val_metrics[k]['precision']:.4f}  "
            f"avg val_rec={avg_val_metrics[k]['recall']:.4f}  "
            f"avg val_f1={avg_val_metrics[k]['f1']:.4f}"
        )

    return avg_train_errors, avg_val_errors, avg_train_metrics, avg_val_metrics


def _stratified_subsample(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return indices for a stratified subsample of size n from label array y."""
    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    indices = []
    for cls, prop in zip(classes, proportions):
        cls_idx = np.where(y == cls)[0]
        n_cls   = min(max(1, round(n * prop)), len(cls_idx))
        chosen  = rng.choice(cls_idx, size=n_cls, replace=False)
        indices.extend(chosen.tolist())
    return np.array(indices)


# ---------------------------------------------------------------------------
# Main — CV, print, plot, final model
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")

    k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    n_folds  = 5

    # ── n_folds-fold Cross-Validation ─────────────────────────────────────
    print(f"\nRunning {n_folds}-fold stratified CV over k = {k_values} ...")
    avg_train_errors, avg_val_errors, avg_train_metrics, avg_val_metrics = evaluate_knn_cv(
        X_train, y_train,
        k_values=k_values,
        n_splits=n_folds,
        train_subsample=2_000,
    )

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'k':>6}  {'TrainErr':>9}  {'ValErr':>7}  {'ValPrec':>8}  {'ValRec':>7}  {'ValF1':>7}")
    print("-" * 78)
    for k in k_values:
        vm = avg_val_metrics[k]
        print(
            f"{k:>6}  {avg_train_errors[k]:>9.4f}  {avg_val_errors[k]:>7.4f}  "
            f"{vm['precision']:>8.4f}  {vm['recall']:>7.4f}  {vm['f1']:>7.4f}"
        )
    print("=" * 78)

    # ── Best k by validation F1 ────────────────────────────────────────────
    best_k = max(avg_val_metrics, key=lambda k: avg_val_metrics[k]["f1"])
    print(f"\nBest k by validation F1: k = {best_k}  "
          f"(val_f1 = {avg_val_metrics[best_k]['f1']:.4f}  "
          f"val_err = {avg_val_errors[best_k]:.4f})")

    # ── Plot — two side-by-side subplots ───────────────────────────────────
    train_err_vals = [avg_train_errors[k]           for k in k_values]
    val_err_vals   = [avg_val_errors[k]             for k in k_values]
    train_f1_vals  = [avg_train_metrics[k]["f1"]    for k in k_values]
    val_f1_vals    = [avg_val_metrics[k]["f1"]      for k in k_values]
    val_prec_vals  = [avg_val_metrics[k]["precision"] for k in k_values]
    val_rec_vals   = [avg_val_metrics[k]["recall"]  for k in k_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # --- Left: Error curves ---
    ax1.plot(k_values, train_err_vals, marker="o", linewidth=2,
             label="Training Error (CV subsample)")
    ax1.plot(k_values, val_err_vals,   marker="s", linewidth=2,
             label="Validation Error (CV)")
    ax1.axvline(x=best_k, color="gray", linestyle="--", linewidth=1,
                label=f"Best k = {best_k}")
    ax1.scatter([best_k], [avg_val_errors[best_k]], zorder=5, s=80, color="red")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(k_values)
    ax1.set_xticklabels([str(k) for k in k_values])
    ax1.set_xlabel("k  (log₂ scale)", fontsize=12)
    ax1.set_ylabel("Error Rate", fontsize=12)
    ax1.set_title(f"kNN {n_folds}-Fold CV: Error", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)

    # --- Right: F1 / Precision / Recall curves ---
    ax2.plot(k_values, train_f1_vals, marker="o", linewidth=2,
             label="Training F1 (CV subsample)")
    ax2.plot(k_values, val_f1_vals,   marker="s", linewidth=2,
             label="Validation F1 (CV)")
    ax2.plot(k_values, val_prec_vals, marker="^", linewidth=2,
             linestyle="--", label="Validation Precision")
    ax2.plot(k_values, val_rec_vals,  marker="v", linewidth=2,
             linestyle="--", label="Validation Recall")
    ax2.axvline(x=best_k, color="gray", linestyle="--", linewidth=1,
                label=f"Best k = {best_k}")
    ax2.scatter([best_k], [avg_val_metrics[best_k]["f1"]], zorder=5, s=80, color="red")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(k_values)
    ax2.set_xticklabels([str(k) for k in k_values])
    ax2.set_xlabel("k  (log₂ scale)", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title(f"kNN {n_folds}-Fold CV: F1 / Precision / Recall", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.suptitle("kNN Hyperparameter Tuning — 5-Fold Stratified CV", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("knn_cv_results.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to knn_cv_results.png")
    plt.show()

    # ── Final model on full X_train, evaluate on held-out X_test ──────────
    print(f"\nTraining final model with k = {best_k} on full X_train ...")
    final_model = KNearestNeighbors(k=best_k)
    final_model.fit(X_train, y_train)

    test_preds = final_model.predict(X_test)
    test_error = calculate_error(y_test, test_preds)
    test_mets  = calculate_metrics(y_test, test_preds)

    print(f"\nFinal Test Results  (best k = {best_k})")
    print(f"  Accuracy : {1.0 - test_error:.4f}")
    print(f"  Precision: {test_mets['precision']:.4f}")
    print(f"  Recall   : {test_mets['recall']:.4f}")
    print(f"  F1-score : {test_mets['f1']:.4f}")
