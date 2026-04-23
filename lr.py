"""
Created on Sun Mar 22 12:58:34 2026

@author: furkanbulut
"""

import numpy as np
import pandas as pd
from data_prep import prepare_data, add_bias
from utils import (confusion_matrix_binary, accuracy_score, precision_score,
                   recall_score, f1_score, roc_auc_score_manual, plot_confusion_matrix)

# ============================================================
# LOGISTIC REGRESSION FROM SCRATCH
# ============================================================

# ----------------------------
# 1-3. LOAD, PREPROCESS, SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = prepare_data()

y_train = y_train.reshape(-1, 1).astype(float)
y_test  = y_test.reshape(-1, 1).astype(float)

X_train = add_bias(X_train)
X_test  = add_bias(X_test)

# ----------------------------
# 4. LOGISTIC REGRESSION FUNCTIONS
# ----------------------------
def sigmoid(z):
    z = np.clip(z, -500, 500)  # numerical stability
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, w, reg_lambda=0.0, pos_weight=1.0):
    m = X.shape[0]
    h = sigmoid(X @ w)
    eps = 1e-12
    sample_weights = np.where(y == 1, pos_weight, 1.0)
    loss_terms = -(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))
    loss = np.sum(sample_weights * loss_terms) / m
    reg = (reg_lambda / (2 * m)) * np.sum(w[1:] ** 2)
    return loss + reg

def fit_logistic_regression(X, y, learning_rate=0.05, iterations=5000, reg_lambda=0.1, pos_weight=1.0):
    m, n = X.shape
    w = np.zeros((n, 1))
    losses = []
    sample_weights = np.where(y == 1, pos_weight, 1.0)

    for i in range(iterations):
        h = sigmoid(X @ w)
        error = h - y

        weighted_error = sample_weights * error
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

# ----------------------------
# 5. TRAIN MODEL
# ----------------------------
n_neg = np.sum(y_train == 0)
n_pos = np.sum(y_train == 1)
pos_weight = n_neg / (n_pos + 1e-12)

weights, losses = fit_logistic_regression(
    X_train,
    y_train,
    learning_rate=0.05,
    iterations=5000,
    reg_lambda=0.1,
    pos_weight=pos_weight
)

# ----------------------------
# 6. EVALUATE MODEL
# ----------------------------
y_prob = predict_proba(X_test, weights)
y_pred = predict(X_test, weights, threshold=0.59)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score_manual(y_test, y_prob)
cm = confusion_matrix_binary(y_test, y_pred)

# ----------------------------
# 7. PRINT RESULTS
# ----------------------------
print("Logistic Regression Results")
print("Accuracy : {:.4f}".format(acc))
print("Precision: {:.4f}".format(prec))
print("Recall   : {:.4f}".format(rec))
print("F1-score : {:.4f}".format(f1))
print("ROC-AUC  : {:.4f}".format(auc))
print("\nConfusion Matrix:")
print(cm)

# ----------------------------
# 8. SAVE CONFUSION MATRIX PLOT
# ----------------------------
plot_confusion_matrix(
    y_test, y_pred,
    class_names=["No Default", "Default"],
    title="Logistic Regression — Confusion Matrix",
    save_path="lr_confusion_matrix.png",
)
print("\nPlot saved as 'lr_confusion_matrix.png'.")

# ----------------------------
# . SAVE RESULTS
# ----------------------------
results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
    "Value": [acc, prec, rec, f1, auc]
})

# results.to_csv("logistic_regression_results_only_numpy_pandas.csv", index=False)

# Optional: save coefficients
coef_df = pd.DataFrame({
    "Feature": ["Intercept"] + [f"feature_{i}" for i in range(len(weights) - 1)],
    "Coefficient": weights.flatten()
})