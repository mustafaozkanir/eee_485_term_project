"""
results_table.py
----------------
Visual summary table comparing all models.
Fill in the numbers in RESULTS below, then run with: uv run results_table.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# FILL IN YOUR RESULTS HERE
# ============================================================

RESULTS = [
    # (Model name,             Accuracy, Precision, Recall,  F1)
    ("kNN",                    0.7821,     0.5068,  0.5614,  0.5327),
    ("Logistic Regression",    0.7951,      0.5395,  0.5041,  0.5212),
    ("Decision Tree",          0.7950,      0.5371,  0.5298,  0.5334),
    ("Neural Network",         0.8015,      0.5501,  0.5622,  0.5561),
]

METRICS   = ["Accuracy", "Precision", "Recall", "F1-Score"]
HIGHLIGHT = "F1-Score"   # metric used to highlight the best row


# ============================================================
# PLOT
# ============================================================

def render_table(results, metrics, highlight_metric):
    n_models  = len(results)
    n_metrics = len(metrics)

    model_names = [r[0] for r in results]
    values      = np.array([r[1:] for r in results], dtype=float)  # (n_models, n_metrics)

    # Find best value per column (for bold) and best row (for highlight)
    col_best  = np.argmax(values, axis=0)             # index of best model per metric
    hi_col    = metrics.index(highlight_metric)
    best_row  = int(np.argmax(values[:, hi_col]))

    fig, ax = plt.subplots(figsize=(13, 0.55 * n_models + 2.2))
    ax.set_xlim(0, n_metrics + 1)
    ax.set_ylim(-0.5, n_models + 0.6)
    ax.axis("off")

    # ── Title ─────────────────────────────────────────────────
    ax.text((n_metrics + 1) / 2, n_models + 0.45,
            "Model Comparison",
            fontsize=13, fontweight="bold", va="top", ha="center", color="#1a1a1a")

    # ── Column headers ────────────────────────────────────────
    header_y = n_models - 0.05
    ax.text(0.4, header_y, "Model", fontsize=11, fontweight="bold",
            va="bottom", ha="left", color="#1a1a1a")
    for j, metric in enumerate(metrics):
        color = "#1565C0" if metric == highlight_metric else "#1a1a1a"
        ax.text(j + 1.85, header_y, metric, fontsize=11, fontweight="bold",
                va="bottom", ha="center", color=color)

    # Header underline
    ax.axhline(n_models - 0.15, color="#333333", linewidth=1.2,
               xmin=0.02, xmax=0.98)

    # ── Rows ──────────────────────────────────────────────────
    for i, (row_name, *row_vals) in enumerate(results):
        row_y   = n_models - 1 - i
        is_best = (i == best_row)

        # Row background
        bg_color = "#E3F2FD" if is_best else ("#F9F9F9" if i % 2 == 0 else "white")
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, row_y - 0.42), n_metrics + 1.3, 0.84,
            boxstyle="round,pad=0.02",
            facecolor=bg_color, edgecolor="none", zorder=0,
        ))

        # Model name
        weight = "bold" if is_best else "normal"
        ax.text(0.4, row_y, row_name, fontsize=10, fontweight=weight,
                va="center", ha="left", color="#1a1a1a")

        # Metric values
        for j, val in enumerate(row_vals):
            is_col_best = (i == col_best[j])
            cell_color  = "#1565C0" if (is_col_best and j == hi_col) else \
                          "#2E7D32" if is_col_best else "#1a1a1a"
            fw = "bold" if is_col_best else "normal"
            marker = "★ " if (is_col_best and j == hi_col) else ""
            ax.text(j + 1.85, row_y, f"{marker}{val:.4f}",
                    fontsize=10, fontweight=fw, va="center", ha="center",
                    color=cell_color)

    # ── Row separator lines ───────────────────────────────────
    for i in range(n_models - 1):
        row_y = n_models - 1 - i - 0.5
        ax.axhline(row_y, color="#E0E0E0", linewidth=0.6, xmin=0.02, xmax=0.98)

    plt.tight_layout()
    plt.savefig("results_table.png", dpi=150, bbox_inches="tight")
    print("Saved to results_table.png")
    plt.show()


if __name__ == "__main__":
    render_table(RESULTS, METRICS, HIGHLIGHT)
