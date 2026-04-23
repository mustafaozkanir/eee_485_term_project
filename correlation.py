"""
correlation.py
--------------
Pearson correlation matrix heatmap from scratch (numpy + matplotlib only).
Uses raw features before one-hot encoding so column names stay readable.
Includes the target variable to show feature-target correlations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from data_prep import load_data, clean_data


# ============================================================
# PEARSON CORRELATION
# ============================================================

def pearson_correlation_matrix(X):
    """
    Compute the Pearson correlation matrix for a 2-D array X (m × n).

    r(i, j) = cov(Xi, Xj) / (std(Xi) * std(Xj))

    Returns an (n × n) matrix with values in [-1, 1].
    """
    X   = np.asarray(X, dtype=np.float64)
    mu  = X.mean(axis=0)
    X_c = X - mu                                  # centre columns

    std = X_c.std(axis=0)
    std[std == 0] = 1.0                            # avoid divide-by-zero for constant cols

    X_n = X_c / std                               # standardise
    R   = (X_n.T @ X_n) / (len(X) - 1)           # (n × n) correlation matrix

    # Clip to [-1, 1] to remove floating-point noise
    return np.clip(R, -1.0, 1.0)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # 1. Load raw data (before one-hot encoding — keeps column names readable)
    df = clean_data(load_data("dataset.csv"))

    # Put target at the end so it appears last in the heatmap
    target = "default.payment.next.month"
    feature_cols = [c for c in df.columns if c != target] + [target]
    df = df[feature_cols]

    col_names = list(df.columns)
    X         = df.values.astype(float)
    n         = len(col_names)

    # 2. Compute Pearson correlation matrix
    R = pearson_correlation_matrix(X)

    # 3. Plot heatmap
    fig, ax = plt.subplots(figsize=(13, 13))

    im = ax.imshow(R, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pearson", fontsize=11)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    # Axis ticks — top only, strictly vertical
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_xticklabels(col_names, rotation=90, ha="center", fontsize=8)
    ax.set_yticklabels(col_names, fontsize=8)

    # Mask upper triangle (keep diagonal + lower triangle only)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    R_masked = np.where(mask, np.nan, R)
    im.set_data(R_masked)

    # Annotate lower triangle + diagonal only
    for i in range(n):
        for j in range(i + 1):
            val   = R[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    # Highlight the target column/row with a border
    target_idx = col_names.index(target)
    for spine_pos in ["top", "bottom", "left", "right"]:
        ax.spines[spine_pos].set_visible(False)

    # Draw a rectangle around the target row and column
    ax.add_patch(plt.Rectangle(
        (target_idx - 0.5, -0.5), 1, n,
        fill=False, edgecolor="#E53935", linewidth=2, clip_on=False
    ))
    ax.add_patch(plt.Rectangle(
        (-0.5, target_idx - 0.5), n, 1,
        fill=False, edgecolor="#E53935", linewidth=2, clip_on=False
    ))

    ax.set_title("Pearson Correlation Matrix\n",
                 fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved to correlation_matrix.png")
    plt.show()
