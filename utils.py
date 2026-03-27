"""
utils.py
--------
Shared metrics and visualisation utilities.
Only numpy and matplotlib — no scikit-learn.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# METRICS
# ============================================================

def confusion_matrix_binary(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[tn, fp],
                     [fn, tp]])


def accuracy_score(y_true, y_pred):
    return float(np.mean(np.asarray(y_true).flatten() == np.asarray(y_pred).flatten()))


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
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    count = 0.0
    ties  = 0.0

    for p in pos:
        count += np.sum(p > neg)
        ties  += np.sum(p == neg)

    return (count + 0.5 * ties) / (len(pos) * len(neg))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> None:
    """
    Plot a styled confusion matrix.

    Diagonal cells (correct predictions) are shown in blue-green tones.
    Off-diagonal cells (errors) are shown in warm red tones.
    Each cell shows the raw count and the percentage of the true class total.

    Parameters
    ----------
    y_true      : ground-truth labels
    y_pred      : predicted labels
    class_names : list of strings for axis tick labels.
                  Defaults to the sorted unique values of y_true.
    title       : plot title
    save_path   : if provided, saves the figure to this path
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    classes = sorted(np.unique(y_true).tolist())
    n_classes = len(classes)

    if class_names is None:
        class_names = [str(c) for c in classes]

    # Build confusion matrix from scratch
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_idx, true_cls in enumerate(classes):
        for pred_idx, pred_cls in enumerate(classes):
            cm[true_idx, pred_idx] = int(
                np.sum((y_true == true_cls) & (y_pred == pred_cls))
            )

    # Row-normalised percentages (percentage of each true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct   = cm / (row_sums + 1e-12) * 100

    # ── Custom colourmaps ──────────────────────────────────────────────────
    # Diagonal  → blue-teal gradient (low = light, high = deep teal)
    correct_cmap = LinearSegmentedColormap.from_list(
        "correct", ["#dff2f0", "#0a7c6e"]
    )
    # Off-diagonal → white-to-crimson gradient
    error_cmap = LinearSegmentedColormap.from_list(
        "error", ["#fff5f5", "#c0392b"]
    )

    fig, ax = plt.subplots(figsize=(5 + n_classes, 4 + n_classes))

    # Normalise each cell independently against its own colour range
    # so both small and large matrices look good
    for true_idx in range(n_classes):
        for pred_idx in range(n_classes):
            value    = cm[true_idx, pred_idx]
            pct      = cm_pct[true_idx, pred_idx]
            is_diag  = (true_idx == pred_idx)

            # Colour intensity proportional to percentage within its cmap
            intensity = pct / 100.0
            colour = (correct_cmap(intensity) if is_diag
                      else error_cmap(intensity))

            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (pred_idx - 0.45, true_idx - 0.45),
                    0.90, 0.90,
                    boxstyle="round,pad=0.05",
                    facecolor=colour,
                    edgecolor="white",
                    linewidth=2,
                )
            )

            # Choose text colour for contrast
            text_colour = "white" if intensity > 0.55 else "#1a1a1a"

            ax.text(
                pred_idx, true_idx,
                f"{value}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=text_colour,
            )

    # ── Axes formatting ────────────────────────────────────────────────────
    ax.set_xlim(-0.5, n_classes - 0.5)
    ax.set_ylim(n_classes - 0.5, -0.5)      # invert y so row 0 is at top

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)

    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=13, labelpad=10)
    ax.set_title(title,              fontsize=14, fontweight="bold", pad=15)

    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position("top")

    ax.spines[:].set_visible(False)
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("#f8f8f8")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.show()
