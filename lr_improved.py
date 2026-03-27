#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:51:37 2026

@author: furkanbulut
"""

import numpy as np
import pandas as pd

# ============================================================
# IMPROVED LOGISTIC REGRESSION FROM SCRATCH
# Only numpy and pandas are used
# Goal: improve recall and F1-score
# ============================================================

# ----------------------------
# 1. LOAD DATA
# ----------------------------
DATA_PATH = "UCI_Credit_Card.csv"
df = pd.read_csv(DATA_PATH)

# Remove ID if present
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

target_col = "default.payment.next.month"

# ----------------------------
# 2. BASIC PREPROCESSING
# ----------------------------
# Treat these as categorical
categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]

# One-hot encode categoricals
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate X and y
X_df = df.drop(columns=[target_col])
y = df[target_col].values.reshape(-1, 1).astype(float)

# Fill missing values if any
X_df = X_df.fillna(X_df.median(numeric_only=True))

# Standardize features
X = X_df.values.astype(float)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1.0
X = (X - X_mean) / X_std

# Add intercept term
X = np.hstack([np.ones((X.shape[0], 1)), X])

# ----------------------------
# 3. TRAIN / VALIDATION / TEST SPLIT
# ----------------------------
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

n = len(indices)
test_size = int(0.20 * n)
val_size = int(0.20 * (n - test_size))   # 20% of remaining train part

test_idx = indices[:test_size]
remaining_idx = indices[test_size:]

val_idx = remaining_idx[:val_size]
train_idx = remaining_idx[val_size:]

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]

# ----------------------------
# 4. LOGISTIC REGRESSION FUNCTIONS
# ----------------------------
def sigmoid(z):
    z = np.clip(z, -500, 500)  # numerical stability
    return 1 / (1 + np.exp(-z))

def compute_weighted_loss(X, y, w, pos_weight=1.0, reg_lambda=0.0):
    m = X.shape[0]
    h = sigmoid(X @ w)
    eps = 1e-12

    # sample weights: higher weight for positive class
    sample_weights = np.where(y == 1, pos_weight, 1.0)

    loss_terms = -(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))
    weighted_loss = np.sum(sample_weights * loss_terms) / m

    # do not regularize intercept
    reg = (reg_lambda / (2 * m)) * np.sum(w[1:] ** 2)
    return weighted_loss + reg

def fit_logistic_regression_weighted(
    X,
    y,
    learning_rate=0.03,
    iterations=8000,
    reg_lambda=0.05,
    pos_weight=1.0
):
    m, n = X.shape
    w = np.zeros((n, 1))
    losses = []

    # sample weights for gradient
    sample_weights = np.where(y == 1, pos_weight, 1.0)

    for i in range(iterations):
        h = sigmoid(X @ w)
        error = h - y

        weighted_error = sample_weights * error
        grad = (X.T @ weighted_error) / m

        # regularize all except intercept
        grad[1:] += (reg_lambda / m) * w[1:]

        w = w - learning_rate * grad

        if i % 100 == 0:
            losses.append(compute_weighted_loss(X, y, w, pos_weight, reg_lambda))

    return w, losses

def predict_proba(X, w):
    return sigmoid(X @ w)

def predict(X, w, threshold=0.5):
    probs = predict_proba(X, w)
    return (probs >= threshold).astype(int)

# ----------------------------
# 5. METRICS
# ----------------------------
def confusion_matrix_binary(y_true, y_pred):
    y_true = y_true.flatten().astype(int)
    y_pred = y_pred.flatten().astype(int)

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
    ties = 0.0

    for p in pos:
        count += np.sum(p > neg)
        ties += np.sum(p == neg)

    auc = (count + 0.5 * ties) / (len(pos) * len(neg))
    return auc

# ----------------------------
# 6. CHOOSE POSITIVE CLASS WEIGHT
# ----------------------------
# A common good starting point:
# pos_weight = (# negatives) / (# positives)
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
pos_weight = n_neg / (n_pos + 1e-12)

# ----------------------------
# 7. TRAIN MODEL
# ----------------------------
weights, losses = fit_logistic_regression_weighted(
    X_train,
    y_train,
    learning_rate=0.03,
    iterations=8000,
    reg_lambda=0.05,
    pos_weight=pos_weight
)

# ----------------------------
# 8. THRESHOLD TUNING ON VALIDATION SET
# ----------------------------
val_prob = predict_proba(X_val, weights)

best_threshold = 0.50
best_f1 = -1

thresholds = np.arange(0.10, 0.90, 0.01)

for threshold in thresholds:
    val_pred = (val_prob >= threshold).astype(int)
    current_f1 = f1_score(y_val, val_pred)

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold

# ----------------------------
# 9. EVALUATE ON TEST SET
# ----------------------------
y_prob_test = predict_proba(X_test, weights)
y_pred_test = predict(X_test, weights, threshold=best_threshold)

acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test)
rec = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
auc = roc_auc_score_manual(y_test, y_prob_test)
cm = confusion_matrix_binary(y_test, y_pred_test)

# ----------------------------
# 10. PRINT RESULTS
# ----------------------------
print("Improved Logistic Regression Results")
print("Chosen threshold: {:.2f}".format(best_threshold))
print("Positive class weight: {:.4f}".format(pos_weight))
print("Accuracy : {:.4f}".format(acc))
print("Precision: {:.4f}".format(prec))
print("Recall   : {:.4f}".format(rec))
print("F1-score : {:.4f}".format(f1))
print("ROC-AUC  : {:.4f}".format(auc))
print("\nConfusion Matrix:")
print(cm)

# ----------------------------
# 11. SAVE RESULTS
# ----------------------------
results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "Chosen threshold",
        "Positive class weight"
    ],
    "Value": [acc, prec, rec, f1, auc, best_threshold, pos_weight]
})

results.to_csv("improved_logistic_regression_results_only_numpy_pandas.csv", index=False)

coef_df = pd.DataFrame({
    "Feature": ["Intercept"] + list(X_df.columns),
    "Coefficient": weights.flatten()
})
coef_df.to_csv("improved_logistic_regression_coefficients_only_numpy_pandas.csv", index=False)

print("\nResults saved to improved_logistic_regression_results_only_numpy_pandas.csv")
print("Coefficients saved to improved_logistic_regression_coefficients_only_numpy_pandas.csv")