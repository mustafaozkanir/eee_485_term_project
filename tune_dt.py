"""
tune_dt.py
-----------
k-fold stratified cross-validation to tune hyperparameters for the custom
Decision Tree classifier.
"""

import numpy as np
import matplotlib.pyplot as plt

from dt import DecisionTree
from data_prep import prepare_data
from tune_knn import get_cv_splits
from utils import (accuracy_score, precision_score, recall_score,
                   f1_score, roc_auc_score_manual, plot_confusion_matrix)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Cross-Validation Loop
# ---------------------------------------------------------------------------

def evaluate_dt_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    depth_values: list,
    n_splits: int = 5,
    random_seed: int = 42,
    threshold: float = 0.5,
) -> tuple[dict, dict, dict, dict]:
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

            model = DecisionTree(max_depth=depth, random_seed=random_seed)
            model.fit(X_cv_train, y_cv_train)

            # Validation
            val_preds = model.predict(X_cv_val, threshold=threshold)
            val_err   = 1.0 - accuracy_score(y_cv_val, val_preds)
            fold_val_errors.append(val_err)
            fold_val_metrics["precision"].append(precision_score(y_cv_val, val_preds))
            fold_val_metrics["recall"].append(recall_score(y_cv_val, val_preds))
            fold_val_metrics["f1"].append(f1_score(y_cv_val, val_preds))

            # Training
            train_preds = model.predict(X_cv_train, threshold=threshold)
            train_err   = 1.0 - accuracy_score(y_cv_train, train_preds)
            fold_train_errors.append(train_err)
            fold_train_metrics["precision"].append(precision_score(y_cv_train, train_preds))
            fold_train_metrics["recall"].append(recall_score(y_cv_train, train_preds))
            fold_train_metrics["f1"].append(f1_score(y_cv_train, train_preds))

            print(
                f"    Fold {fold_idx + 1}/{n_splits} — "
                f"train_err={train_err:.4f}  val_err={val_err:.4f}  "
                f"val_f1={fold_val_metrics['f1'][-1]:.4f}"
            )

        avg_train_errors[depth]  = float(np.mean(fold_train_errors))
        avg_val_errors[depth]    = float(np.mean(fold_val_errors))
        avg_train_metrics[depth] = {m: float(np.mean(fold_train_metrics[m])) for m in fold_train_metrics}
        avg_val_metrics[depth]   = {m: float(np.mean(fold_val_metrics[m]))   for m in fold_val_metrics}

    return avg_train_errors, avg_val_errors, avg_train_metrics, avg_val_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()

    depth_values = [2, 4, 6, 8, 10, 12, 15, 20]
    n_folds = 5

    print(f"\nRunning {n_folds}-fold stratified CV over max_depth = {depth_values} ...")
    avg_train_errors, avg_val_errors, avg_train_metrics, avg_val_metrics = evaluate_dt_cv(
        X_train, y_train,
        depth_values = depth_values,
        n_splits     = n_folds,
        threshold    = THRESHOLD,
    )

    # Console summary
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

    best_depth = max(avg_val_metrics, key=lambda d: avg_val_metrics[d]["f1"])
    print(f"\nBest max_depth by validation F1: {best_depth} "
          f"(val_f1 = {avg_val_metrics[best_depth]['f1']:.4f})")

    # Plot
    train_err_vals = [avg_train_errors[d] for d in depth_values]
    val_err_vals   = [avg_val_errors[d]   for d in depth_values]
    val_f1_vals    = [avg_val_metrics[d]["f1"] for d in depth_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    ax1.plot(depth_values, train_err_vals, marker="o", label="Training Error")
    ax1.plot(depth_values, val_err_vals,   marker="s", label="Validation Error")
    ax1.set_xlabel("max_depth")
    ax1.set_ylabel("Error Rate")
    ax1.set_title("DT Complexity Curve: Error")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(depth_values, val_f1_vals, marker="D", color="green", label="Validation F1")
    ax2.set_xlabel("max_depth")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("DT Complexity Curve: F1 Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dt_cv_results.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to dt_cv_results.png")
    plt.show()

    # Final model
    print(f"\nTraining final model with max_depth = {best_depth} on full X_train ...")
    final_model = DecisionTree(max_depth=best_depth)
    final_model.fit(X_train, y_train)

    y_prob = final_model.predict_proba(X_test)
    y_pred = final_model.predict(X_test, threshold=THRESHOLD)

    print(f"\nFinal Test Results (max_depth={best_depth}, threshold={THRESHOLD})")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-score  : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score_manual(y_test, y_prob):.4f}")

    plot_confusion_matrix(
        y_test, y_pred,
        class_names=["No Default", "Default"],
        title=f"Decision Tree (depth={best_depth}) — Confusion Matrix",
        save_path="dt_confusion_matrix.png",
    )
