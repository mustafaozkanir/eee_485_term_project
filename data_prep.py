"""
data_prep.py
------------
Data preparation pipeline for credit card default prediction.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data(path: str = "dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    return df


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Merge unknown / rare EDUCATION values → 0
    df["EDUCATION"] = df["EDUCATION"].replace({5: 0, 6: 0})
    # Merge unknown MARRIAGE value → 0
    df["MARRIAGE"] = df["MARRIAGE"].replace({3: 0})
    return df


# ---------------------------------------------------------------------------
# 3. One-Hot Encoding
# ---------------------------------------------------------------------------

def one_hot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        unique_vals = sorted(df[col].unique())
        for val in unique_vals:
            new_col = f"{col}_{val}"
            df[new_col] = (df[col] == val).astype(int)
        df = df.drop(columns=[col])
    return df


# ---------------------------------------------------------------------------
# 4. Stratified Train/Test Split ( 80/20)
# ---------------------------------------------------------------------------

def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    random_seed: int = 54,
) -> tuple:
    """
    Split data into train / test while preserving the class ratio.
    Returns (X_train, X_test, y_train, y_test) as numpy arrays.
    """
    rng = np.random.default_rng(random_seed)

    train_idx, test_idx = [], []
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        n_test = max(1, int(len(cls_indices) * test_size))
        test_idx.extend(cls_indices[:n_test].tolist())
        train_idx.extend(cls_indices[n_test:].tolist())

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)

    # Shuffle within each split so rows aren't grouped by class
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ---------------------------------------------------------------------------
# 5. Z-score Standardization
# ---------------------------------------------------------------------------

def fit_standardize(X_train: np.ndarray) -> tuple:
    """Compute mean and std from training data only."""
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    # Avoid division by zero for constant features
    std[std == 0] = 1.0
    return mean, std


def apply_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


# ---------------------------------------------------------------------------
# 6. Bias Term Utility
# ---------------------------------------------------------------------------

def add_bias(X: np.ndarray) -> np.ndarray:
    """Prepend a column of 1s to feature matrix X."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


# ---------------------------------------------------------------------------
# 7. Main Pipeline — returns X_train, X_test, y_train, y_test
# ---------------------------------------------------------------------------

def prepare_data(path: str = "dataset.csv", random_seed: int = 54):
    # --- Load & clean ---
    df = load_data(path)
    df = clean_data(df)

    target_col = "default.payment.next.month"
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]

    numerical_cols = [
        "LIMIT_BAL", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",  "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6",
    ]

    # --- One-hot encode categorical features ---
    df = one_hot_encode(df, categorical_cols)

    # --- Separate features and target ---
    y = df[target_col].values
    X_df = df.drop(columns=[target_col])

    # Identify column positions of numerical features after encoding
    feature_cols = list(X_df.columns)
    num_col_indices = [feature_cols.index(c) for c in numerical_cols if c in feature_cols]

    X = X_df.values.astype(float)

    # --- Stratified split BEFORE scaling (prevents data leakage) ---
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.20, random_seed=random_seed)

    # --- Standardize numerical columns only ---
    # Extract numerical sub-matrices, fit on train, apply to both
    X_train_num = X_train[:, num_col_indices]
    X_test_num  = X_test[:, num_col_indices]

    mean, std = fit_standardize(X_train_num)

    X_train[:, num_col_indices] = apply_standardize(X_train_num, mean, std)
    X_test[:, num_col_indices]  = apply_standardize(X_test_num,  mean, std)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"X_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}   y_test  : {y_test.shape}")
    unique, counts = np.unique(y_train, return_counts=True)
    print("Train class distribution:", dict(zip(unique, counts / len(y_train))))
    unique, counts = np.unique(y_test, return_counts=True)
    print("Test  class distribution:", dict(zip(unique, counts / len(y_test))))
