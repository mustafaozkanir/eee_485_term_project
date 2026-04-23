"""
data_prep_filtered.py
---------------------
Same pipeline as data_prep.py but drops low mutual-information features.

Removed features (NMI <= 0.001 vs target):
    AGE, SEX, MARRIAGE, BILL_AMT1–BILL_AMT6

All helper functions (add_bias, stratified_split, etc.) are re-exported
so any script can swap  `from data_prep import ...`
                    →   `from data_prep_filtered import ...`
with no other changes.
"""

import numpy as np
import pandas as pd
from data_prep import (load_data, clean_data, one_hot_encode,
                       stratified_split, fit_standardize,
                       apply_standardize, add_bias)

# Features dropped based on NMI <= 0.001 from mutual_info.py
DROP_COLS = ["AGE", "SEX", "MARRIAGE",
             "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
             "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]


def prepare_data(path: str = "dataset.csv", random_seed: int = 54):
    # --- Load & clean ---
    df = load_data(path)
    df = clean_data(df)

    target_col = "default.payment.next.month"

    # Drop low-MI features before any processing
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # MARRIAGE and SEX were dropped — only EDUCATION remains categorical
    categorical_cols = [c for c in ["EDUCATION"] if c in df.columns]

    numerical_cols = [c for c in [
        "LIMIT_BAL",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ] if c in df.columns]

    # --- One-hot encode remaining categorical features ---
    if categorical_cols:
        df = one_hot_encode(df, categorical_cols)

    # --- Separate features and target ---
    y    = df[target_col].values
    X_df = df.drop(columns=[target_col])

    feature_cols    = list(X_df.columns)
    num_col_indices = [feature_cols.index(c) for c in numerical_cols if c in feature_cols]

    X = X_df.values.astype(float)

    # --- Stratified split BEFORE scaling ---
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=0.20, random_seed=random_seed
    )

    # --- Standardize numerical columns only ---
    X_train_num = X_train[:, num_col_indices]
    X_test_num  = X_test[:, num_col_indices]

    mean, std = fit_standardize(X_train_num)

    X_train[:, num_col_indices] = apply_standardize(X_train_num, mean, std)
    X_test[:, num_col_indices]  = apply_standardize(X_test_num,  mean, std)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"X_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}   y_test  : {y_test.shape}")
    unique, counts = np.unique(y_train, return_counts=True)
    print("Train class distribution:", dict(zip(unique, counts / len(y_train))))
