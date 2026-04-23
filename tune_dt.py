"""
tune_dt.py
-----------
k-fold stratified cross-validation to tune hyperparameters for the custom 
Decision Tree classifier.
"""

import numpy as np
import matplotlib.pyplot as plt

from decision_tree import DecisionTree
from data_prep import prepare_data

# ---------------------------------------------------------------------------
# 1. Cross-Validation Splitter (stratified)
# ---------------------------------------------------------------------------

def get_cv_splits(X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_seed: int = 42):
    """
    Stratified k-fold cross-validation splitter implemented from scratch.
    Ensures each fold mirrors the class distribution of the full dataset.
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
    """Compute precision, recall, and F1 from scratch."""
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

def evaluate_dt_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    depth_values: list,
    n_splits: int = 5,
    random_seed: int = 42,
) -> tuple[dict, dict, dict, dict]:
    """
    Evaluate Decision Tree for each max_depth using n_splits-fold stratified CV.
    """
    splits = list(get_cv_splits(X_train, y_train, n_splits=n_splits, random_seed=random_seed))

    avg_train_errors  = {}
    avg_val_errors    = {}
    avg_train_metrics = {}
    avg_val_metrics   = {}

    for depth in depth_values:
        print(f"\n  max_depth = {depth}")
        fold_train_errors  = []
        fold_val_errors    = []
        fold_train_metrics = {"precision": [], "recall": [], "f1": []}
        fold_val_metrics   = {"precision": [], "recall": [], "f1": []}

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            X_cv_train, y_cv_train = X_train[tr_idx], y_train[tr_idx]
            X_cv_val,   y_cv_val   = X_train[val_idx], y_train[val_idx]

            # Initialize and fit
            model = DecisionTree(max_depth=depth, random_seed=random_seed)
            model.fit(X_cv_train, y_cv_train)

            # --- validation ---
            val_preds = model.predict(X_cv_val)
            val_err   = calculate_error(y_cv_val, val_preds)
            val_mets  = calculate_metrics(y_cv_val, val_preds)
            fold_val_errors.append(val_err)
            for m in fold_val_metrics:
                fold_val_metrics[m].append(val_mets[m])

            # --- training ---
            train_preds = model.predict(X_cv_train)
            train_err   = calculate_error(y_cv_train, train_preds)
            train_mets  = calculate_metrics(y_cv_train, train_preds)
            fold_train_errors.append(train_err)
            for m in fold_train_metrics:
                fold_train_metrics[m].append(train_mets[m])

            print(
                f"    Fold {fold_idx + 1}/{n_splits} — "
                f"train_err={train_err:.4f}  val_err={val_err:.4f}  "
                f"val_f1={val_mets['f1']:.4f}"
            )

        avg_train_errors[depth]  = float(np.mean(fold_train_errors))
        avg_val_errors[depth]    = float(np.mean(fold_val_errors))
        avg_train_metrics[depth] = {m: float(np.mean(fold_train_metrics[m])) for m in fold_train_metrics}
        avg_val_metrics[depth]   = {m: float(np.mean(fold_val_metrics[m]))   for m in fold_val_metrics}

    return avg_train_errors, avg_val_errors, avg_train_metrics, avg_val_metrics


# ---------------------------------------------------------------------------
# Main — CV, print, plot, final model
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Range of depths to test
    depth_values = [2, 4, 6, 8, 10, 12, 15, 20]
    n_folds = 5

    print(f"\nRunning {n_folds}-fold stratified CV over max_depth = {depth_values} ...")
    avg_train_errors, avg_val_errors, avg_train_metrics, avg_val_metrics = evaluate_dt_cv(
        X_train, y_train,
        depth_values=depth_values,
        n_splits=n_folds
    )

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'Depth':>6}  {'TrainErr':>9}  {'ValErr':>7}  {'ValPrec':>8}  {'ValRec':>7}  {'ValF1':>7}")
    print("-" * 78)
    for d in depth_values:
        vm = avg_val_metrics[d]
        print(
            f"{d:>6}  {avg_train_errors[d]:>9.4f}  {avg_val_errors[d]:>7.4f}  "
            f"{vm['precision']:>8.4f}  {vm['recall']:>7.4f}  {vm['f1']:>7.4f}"
        )
    print("=" * 78)

    # ── Best depth by validation F1 ────────────────────────────────────────
    best_depth = max(avg_val_metrics, key=lambda d: avg_val_metrics[d]["f1"])
    print(f"\nBest max_depth by validation F1: {best_depth} "
          f"(val_f1 = {avg_val_metrics[best_depth]['f1']:.4f})")

    # ── Plot ──────────────────────────────────────────────────────────────
    train_err_vals = [avg_train_errors[d] for d in depth_values]
    val_err_vals   = [avg_val_errors[d]   for d in depth_values]
    val_f1_vals    = [avg_val_metrics[d]["f1"] for d in depth_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Error Plot
    ax1.plot(depth_values, train_err_vals, marker="o", label="Training Error")
    ax1.plot(depth_values, val_err_vals,   marker="s", label="Validation Error")
    ax1.set_xlabel("max_depth")
    ax1.set_ylabel("Error Rate")
    ax1.set_title("DT Complexity Curve: Error")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # F1 Plot
    ax2.plot(depth_values, val_f1_vals, marker="D", color="green", label="Validation F1")
    ax2.set_xlabel("max_depth")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("DT Complexity Curve: F1 Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dt_cv_results.png")
    print("\nPlot saved to dt_cv_results.png")

    # ── Final model ────────────────────────────────────────────────────────
    print(f"\nTraining final model with max_depth = {best_depth} on full X_train ...")
    final_model = DecisionTree(max_depth=best_depth)
    final_model.fit(X_train, y_train)

    test_preds = final_model.predict(X_test)
    test_mets  = calculate_metrics(y_test, test_preds)

    print(f"\nFinal Test Results (max_depth = {best_depth})")
    print(f"  Accuracy : {1.0 - calculate_error(y_test, test_preds):.4f}")
    print(f"  F1-score : {test_mets['f1']:.4f}")