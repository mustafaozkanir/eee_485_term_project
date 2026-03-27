#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logistic Regression from scratch.
Only numpy and pandas are used.
Data is supplied by data_prep.prepare_data() to ensure consistent
cleaning, encoding, stratified splitting, and leak-free normalisation.
"""

import numpy as np
import pandas as pd


# ============================================================
# LOGISTIC REGRESSION FUNCTIONS
# ============================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)  # numerical stability
    return 1 / (1 + np.exp(-z))


def compute_loss(X, y, w, reg_lambda=0.0, pos_weight=1.0):
    m = X.shape[0]
    h = sigmoid(X @ w)
    eps = 1e-12
    # pos_weight upscales the minority-class (default=1) log-loss term
    loss = -(1 / m) * np.sum(
        pos_weight * y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps)
    )
    reg = (reg_lambda / (2 * m)) * np.sum(w[1:] ** 2)
    return loss + reg


def fit_logistic_regression(X, y, learning_rate=0.05, iterations=5000,
                             reg_lambda=0.1, pos_weight=1.0):
    m, n = X.shape
    w = np.zeros((n, 1))
    losses = []

    for i in range(iterations):
        h = sigmoid(X @ w)
        error = h - y

        # Scale the residual for positive-class samples so their gradient
        # contribution matches their weight in the loss
        weighted_error = error.copy()
        weighted_error[y == 1] *= pos_weight

        grad = (1 / m) * (X.T @ weighted_error)
        grad[1:] += (reg_lambda / m) * w[1:]  # do not regularize bias

        w = w - learning_rate * grad

        if i % 100 == 0:
            losses.append(compute_loss(X, y, w, reg_lambda, pos_weight))

    return w, losses


def predict_proba(X, w):
    return sigmoid(X @ w)


def predict(X, w, threshold=0.5):
    probs = predict_proba(X, w)
    return (probs >= threshold).astype(int)


# ============================================================
# METRICS
# ============================================================

def confusion_matrix_binary(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[tn, fp],
                     [fn, tp]])


def accuracy_score(y_true, y_pred):
    return np.mean(y_true.flatten() == y_pred.flatten())


def precision_score(y_true, y_pred):
    cm = confusion_matrix_binary(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    return tp / (tp + fp + 1e-12)


def recall_score(y_true, y_pred):
    cm = confusion_matrix_binary(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    return tp / (tp + fn + 1e-12)


def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-12)


def roc_auc_score_manual(y_true, y_prob):
    y_true = y_true.flatten()
    y_prob = y_prob.flatten()

    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    count = 0.0
    ties  = 0.0

    for p in pos:
        count += np.sum(p > neg)
        ties  += np.sum(p == neg)

    auc = (count + 0.5 * ties) / (len(pos) * len(neg))
    return auc


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    from data_prep import prepare_data, add_bias

    # ----------------------------------------------------------
    # 1. Load prepared data
    #    - Cleaned, one-hot encoded, stratified 80/20 split,
    #      z-score normalised (fit on train only — no leakage).
    # ----------------------------------------------------------
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")

    # Logistic regression expects y shaped (n, 1)
    y_train = y_train.reshape(-1, 1).astype(float)
    y_test  = y_test.reshape(-1, 1).astype(float)

    # Prepend bias column of 1s (intercept term)
    X_train_b = add_bias(X_train)
    X_test_b  = add_bias(X_test)

    # ----------------------------------------------------------
    # 2. Train
    # ----------------------------------------------------------
    # Class-weighted loss: penalise minority-class (default=1) errors
    # proportionally more so the gradient is balanced across classes.
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    pos_weight = n_neg / n_pos          # ≈ 3.52 for this dataset
    print(f"\nClass counts — 0: {int(n_neg)}  1: {int(n_pos)}")
    print(f"pos_weight = {pos_weight:.4f}")

    print("\nTraining logistic regression (5 000 iterations)...")
    weights, losses = fit_logistic_regression(
        X_train_b,
        y_train,
        learning_rate=0.05,
        iterations=5000,
        reg_lambda=0.1,
        pos_weight=pos_weight,
    )

    # ----------------------------------------------------------
    # 3. Evaluate
    # ----------------------------------------------------------
    # Threshold lowered to 0.35: with class weighting the model's predicted
    # probabilities shift upward for defaulters, so 0.35 captures more of
    # them without sacrificing too much precision.
    threshold = 0.60
    y_prob = predict_proba(X_test_b, weights)
    y_pred = predict(X_test_b, weights, threshold=threshold)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score_manual(y_test, y_prob)
    cm   = confusion_matrix_binary(y_test, y_pred)

    # ----------------------------------------------------------
    # 4. Print results
    # ----------------------------------------------------------
    print(f"\nLogistic Regression Results  (threshold={threshold})")
    print("Accuracy : {:.4f}".format(acc))
    print("Precision: {:.4f}".format(prec))
    print("Recall   : {:.4f}".format(rec))
    print("F1-score : {:.4f}".format(f1))
    print("ROC-AUC  : {:.4f}".format(auc))
    print("\nConfusion Matrix:")
    print(cm)

    # ----------------------------------------------------------
    # 5. Save results
    # ----------------------------------------------------------
    results = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        "Value":  [acc, prec, rec, f1, auc],
    })
    results.to_csv("logistic_regression_results_only_numpy_pandas.csv", index=False)

    # Feature names: bias + one feature label per column index
    n_features = X_train_b.shape[1]
    feature_names = ["Intercept"] + [f"feature_{i}" for i in range(n_features - 1)]
    coef_df = pd.DataFrame({
        "Feature":     feature_names,
        "Coefficient": weights.flatten(),
    })
    coef_df.to_csv("logistic_regression_coefficients_only_numpy_pandas.csv", index=False)

    print("\nResults saved to logistic_regression_results_only_numpy_pandas.csv")
    print("Coefficients saved to logistic_regression_coefficients_only_numpy_pandas.csv")
