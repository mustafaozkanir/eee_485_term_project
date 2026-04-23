"""
tune_dt_threshold.py
--------------------
Cross-validation to find the optimal classification threshold for a
pre-tuned Decision Tree, optimising for F1.
"""

import numpy as np
import matplotlib.pyplot as plt

from dt import DecisionTree
from data_prep import prepare_data
from tune_knn import get_cv_splits
from utils import (accuracy_score, precision_score, recall_score,
                   f1_score, roc_auc_score_manual, plot_confusion_matrix)


# ============================================================
# CONFIGURATION
# ============================================================

BEST_DEPTH     = 8
N_FOLDS        = 5
THRESHOLD_MIN  = 0.10
THRESHOLD_MAX  = 0.90
THRESHOLD_STEP = 0.05
RANDOM_SEED    = 42


# ============================================================
# CV THRESHOLD SEARCH
# ============================================================

def evaluate_threshold_cv(X, y, best_depth, thresholds, n_splits=5, random_seed=42):
    splits = list(get_cv_splits(X, y, n_splits=n_splits, random_seed=random_seed))

    # Train once per fold, store probabilities — sweep thresholds without retraining
    print(f"  Training {n_splits} CV folds...")
    fold_probs  = []
    fold_labels = []

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        X_cv_train, y_cv_train = X[tr_idx], y[tr_idx]
        X_cv_val,   y_cv_val   = X[val_idx], y[val_idx]

        model = DecisionTree(max_depth=best_depth, random_seed=random_seed)
        model.fit(X_cv_train, y_cv_train)

        fold_probs.append(model.predict_proba(X_cv_val))
        fold_labels.append(y_cv_val)
        print(f"    Fold {fold_idx + 1}/{n_splits} done.")

    print(f"\n  Sweeping {len(thresholds)} thresholds...")
    results = {}

    for t in thresholds:
        fold_f1s        = []
        fold_precisions = []
        fold_recalls    = []

        for probs, labels in zip(fold_probs, fold_labels):
            preds = (probs >= t).astype(int)
            fold_f1s.append(f1_score(labels, preds))
            fold_precisions.append(precision_score(labels, preds))
            fold_recalls.append(recall_score(labels, preds))

        results[round(float(t), 4)] = {
            "f1"       : float(np.mean(fold_f1s)),
            "precision": float(np.mean(fold_precisions)),
            "recall"   : float(np.mean(fold_recalls)),
        }

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")

    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)

    print(f"\nRunning {N_FOLDS}-fold CV threshold search (max_depth={BEST_DEPTH})...")
    cv_results = evaluate_threshold_cv(
        X_train, y_train,
        best_depth  = BEST_DEPTH,
        thresholds  = thresholds,
        n_splits    = N_FOLDS,
        random_seed = RANDOM_SEED,
    )

    # Best threshold by F1
    best_threshold = max(cv_results, key=lambda t: cv_results[t]["f1"])
    best_f1        = cv_results[best_threshold]["f1"]
    best_precision = cv_results[best_threshold]["precision"]
    best_recall    = cv_results[best_threshold]["recall"]

    # Console summary
    print("\n" + "=" * 56)
    print(f"{'Threshold':>10}  {'CV F1':>8}  {'CV Prec':>9}  {'CV Rec':>8}")
    print("-" * 56)
    for t, v in cv_results.items():
        marker = " ← best" if t == best_threshold else ""
        print(f"{t:>10.2f}  {v['f1']:>8.4f}  {v['precision']:>9.4f}  {v['recall']:>8.4f}{marker}")
    print("=" * 56)
    print(f"\nBest threshold (max CV F1): {best_threshold:.2f}")
    print(f"  CV F1        : {best_f1:.4f}")
    print(f"  CV Precision : {best_precision:.4f}")
    print(f"  CV Recall    : {best_recall:.4f}")

    # Final model on full X_train
    print(f"\nTraining final model (max_depth={BEST_DEPTH}) on full X_train...")
    final_model = DecisionTree(max_depth=BEST_DEPTH, random_seed=RANDOM_SEED)
    final_model.fit(X_train, y_train)

    y_prob = final_model.predict_proba(X_test)
    y_pred = (y_prob >= best_threshold).astype(int)

    print(f"\nFinal Test Results  (threshold = {best_threshold:.2f})")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-score  : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score_manual(y_test, y_prob):.4f}")

    # Plot
    thresh_list = list(cv_results.keys())
    f1_vals     = [cv_results[t]["f1"]        for t in thresh_list]
    prec_vals   = [cv_results[t]["precision"] for t in thresh_list]
    rec_vals    = [cv_results[t]["recall"]    for t in thresh_list]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresh_list, f1_vals,   linewidth=2, label="CV F1",        color="#43A047", linestyle="--")
    ax.plot(thresh_list, prec_vals, linewidth=2, label="CV Precision", color="#1E88E5")
    ax.plot(thresh_list, rec_vals,  linewidth=2, label="CV Recall",    color="#E53935")
    ax.axvline(best_threshold, color="gray", linestyle="--", linewidth=1.2,
               label=f"Best threshold = {best_threshold:.2f}")
    ax.scatter([best_threshold], [best_f1], color="#43A047", zorder=5, s=80)
    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"DT Threshold Tuning — {N_FOLDS}-Fold CV  (max_depth={BEST_DEPTH}, optimising F1)",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("dt_threshold_tuning.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to dt_threshold_tuning.png")
    plt.show()

    plot_confusion_matrix(
        y_test, y_pred,
        class_names=["No Default", "Default"],
        title=f"Decision Tree (depth={BEST_DEPTH}, threshold={best_threshold:.2f}) — Confusion Matrix",
        save_path="dt_threshold_confusion_matrix.png",
    )
