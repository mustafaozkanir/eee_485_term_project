"""
tune_lr_threshold.py
--------------------
k-Fold cross-validation to find the optimal classification threshold
for Logistic Regression, optimising for F1.

Pipeline:
    1. Load prepared data (data_prep.py)
    2. For each threshold candidate:
         - Run k-fold CV on X_train
         - Train LR on each CV-train fold
         - Predict probabilities on each CV-val fold
         - Compute F1 at this threshold on each fold
         - Average F1 across k folds
    3. Select threshold with highest average CV F1
    4. Train final LR on full X_train with best threshold
    5. Evaluate on held-out X_test and print all metrics
    6. Plot average CV F1 (and precision/F1 for context)
       vs threshold
"""

import numpy as np
import matplotlib.pyplot as plt

from data_prep import prepare_data, add_bias


from lr import fit_logistic_regression, predict_proba
from tune_knn import get_cv_splits
from utils import precision_score, recall_score, f1_score, accuracy_score


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
N_FOLDS        = 5
LEARNING_RATE  = 0.05
ITERATIONS     = 5000
REG_LAMBDA     = 0.1
THRESHOLD_MIN  = 0.10
THRESHOLD_MAX  = 0.90
THRESHOLD_STEP = 0.01


# ─────────────────────────────────────────────────────────────
# CROSS-VALIDATION THRESHOLD SEARCH
# ─────────────────────────────────────────────────────────────

def cv_threshold_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    thresholds: np.ndarray,
    n_folds: int = 5,
    learning_rate: float = 0.05,
    iterations: int = 5000,
    reg_lambda: float = 0.1,
    random_seed: int = 42,
) -> dict:
    """
    For every threshold, run k-fold CV and compute average recall,
    precision, and F1 across folds.

    Returns
    -------
    results : dict  {threshold: {"recall": float, "precision": float, "f1": float}}
    """
    # Pre-compute splits once — same folds for every threshold
    splits = list(get_cv_splits(X_train, y_train,
                                n_splits=n_folds,
                                random_seed=random_seed))

    # Collect predicted probabilities for every fold first,
    # then evaluate all thresholds without retraining.
    print(f"  Training {n_folds} CV folds...")
    fold_val_probs  = []   # predicted probabilities on val set per fold
    fold_val_labels = []   # true labels on val set per fold

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        X_cv_train = X_train[tr_idx]
        y_cv_train = y_train[tr_idx].reshape(-1, 1).astype(float)
        X_cv_val   = X_train[val_idx]
        y_cv_val   = y_train[val_idx]

        # Class weight from CV-train fold only
        n_neg      = np.sum(y_cv_train == 0)
        n_pos      = np.sum(y_cv_train == 1)
        pos_weight = n_neg / (n_pos + 1e-12)

        # Add bias and train
        X_cv_train_b = add_bias(X_cv_train)
        X_cv_val_b   = add_bias(X_cv_val)

        weights, _ = fit_logistic_regression(
            X_cv_train_b, y_cv_train,
            learning_rate=learning_rate,
            iterations=iterations,
            reg_lambda=reg_lambda,
            pos_weight=pos_weight,
        )

        probs = predict_proba(X_cv_val_b, weights).flatten()
        fold_val_probs.append(probs)
        fold_val_labels.append(y_cv_val.flatten())
        print(f"    Fold {fold_idx + 1}/{n_folds} done.")

    print(f"\n  Sweeping {len(thresholds)} thresholds...")
    results = {}

    for thresh in thresholds:
        fold_recalls    = []
        fold_precisions = []
        fold_f1s        = []

        for probs, labels in zip(fold_val_probs, fold_val_labels):
            preds = (probs >= thresh).astype(int)
            fold_recalls.append(recall_score(labels, preds))
            fold_precisions.append(precision_score(labels, preds))
            fold_f1s.append(f1_score(labels, preds))

        results[round(float(thresh), 4)] = {
            "recall"    : float(np.mean(fold_recalls)),
            "precision" : float(np.mean(fold_precisions)),
            "f1"        : float(np.mean(fold_f1s)),
        }

    return results


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Load data ───────────────────────────────────────────
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")

    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP,
                           THRESHOLD_STEP)

    # ── 2. CV threshold search ─────────────────────────────────
    print(f"\nRunning {N_FOLDS}-fold CV threshold search (optimising for F1)...")
    cv_results = cv_threshold_search(
        X_train, y_train,
        thresholds=thresholds,
        n_folds=N_FOLDS,
        learning_rate=LEARNING_RATE,
        iterations=ITERATIONS,
        reg_lambda=REG_LAMBDA,
    )

    # ── 3. Select best threshold by recall ────────────────────
    best_threshold = max(cv_results, key=lambda t: cv_results[t]["f1"])
    best_recall    = cv_results[best_threshold]["recall"]
    best_precision = cv_results[best_threshold]["precision"]
    best_f1        = cv_results[best_threshold]["f1"]

    # ── 4. Console summary ─────────────────────────────────────
    print("\n" + "=" * 56)
    print(f"{'Threshold':>10}  {'CV Recall':>10}  {'CV Prec':>9}  {'CV F1':>8}")
    print("-" * 56)
    # Print every 5th threshold to keep output readable
    for t, v in cv_results.items():
        if round(t / THRESHOLD_STEP) % 5 == 0:
            marker = " ← best" if t == best_threshold else ""
            print(f"{t:>10.2f}  {v['recall']:>10.4f}  "
                  f"{v['precision']:>9.4f}  {v['f1']:>8.4f}{marker}")
    print("=" * 56)
    print(f"\nBest threshold (max CV F1): {best_threshold:.2f}")
    print(f"  CV Recall    : {best_recall:.4f}")
    print(f"  CV Precision : {best_precision:.4f}")
    print(f"  CV F1        : {best_f1:.4f}")

    # ── 5. Final model on full X_train ─────────────────────────
    print(f"\nTraining final model on full X_train...")
    n_neg      = np.sum(y_train == 0)
    n_pos      = np.sum(y_train == 1)
    pos_weight = n_neg / (n_pos + 1e-12)

    X_train_b = add_bias(X_train)
    X_test_b  = add_bias(X_test)

    final_weights, _ = fit_logistic_regression(
        X_train_b,
        y_train.reshape(-1, 1).astype(float),
        learning_rate=LEARNING_RATE,
        iterations=ITERATIONS,
        reg_lambda=REG_LAMBDA,
        pos_weight=pos_weight,
    )

    y_prob_test = predict_proba(X_test_b, final_weights).flatten()
    y_pred_test = (y_prob_test >= best_threshold).astype(int)

    print(f"\nFinal Test Results  (threshold = {best_threshold:.2f})")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_test):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred_test):.4f}")
    print(f"  F1-score : {f1_score(y_test, y_pred_test):.4f}")

    # ── 6. Plot ────────────────────────────────────────────────
    thresh_list = list(cv_results.keys())
    recall_vals = [cv_results[t]["recall"]    for t in thresh_list]
    prec_vals   = [cv_results[t]["precision"] for t in thresh_list]
    f1_vals     = [cv_results[t]["f1"]        for t in thresh_list]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(thresh_list, recall_vals,    linewidth=2, label="CV Recall",    color="#E53935")
    ax.plot(thresh_list, prec_vals,      linewidth=2, label="CV Precision", color="#1E88E5")
    ax.plot(thresh_list, f1_vals,        linewidth=2, label="CV F1",        color="#43A047",
            linestyle="--")

    ax.axvline(best_threshold, color="gray", linestyle="--", linewidth=1.2,
               label=f"Best threshold = {best_threshold:.2f}")
    ax.scatter([best_threshold], [best_f1], color="#43A047", zorder=5, s=80)

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"LR Threshold Tuning — {N_FOLDS}-Fold CV  (optimising F1)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("lr_threshold_tuning.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to lr_threshold_tuning.png")
    plt.show()