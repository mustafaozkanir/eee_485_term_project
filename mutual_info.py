"""
mutual_info.py
--------------
Mutual Information between features and the target.

Unlike Pearson correlation, MI captures non-linear relationships.
Continuous features are discretized via equal-width binning before computing MI.

MI(X, Y) = Σ_x Σ_y  p(x,y) · log( p(x,y) / (p(x)·p(y)) )

Normalized MI (NMI) scales MI to [0, 1]:
NMI(X, Y) = MI(X, Y) / sqrt(H(X) · H(Y))
"""

import numpy as np
import matplotlib.pyplot as plt
from data_prep import load_data, clean_data


# ============================================================
# CORE FUNCTIONS
# ============================================================

def _entropy(counts):
    """Shannon entropy from a frequency array."""
    probs = counts / (counts.sum() + 1e-12)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs + 1e-12))


def discretize(x, n_bins=10):
    """Bin a continuous 1-D array into n_bins equal-width integer bins."""
    x     = np.asarray(x, dtype=np.float64)
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return np.zeros(len(x), dtype=int)
    bins  = np.linspace(x_min, x_max, n_bins + 1)
    bins[-1] += 1e-9          # include the right edge
    return np.digitize(x, bins) - 1


def mutual_information(x, y, n_bins=10):
    """
    Compute MI between feature x and target y.
    x is discretized if it has more than n_bins unique values.
    y is assumed discrete (e.g. binary target).

    Returns (mi, nmi) — raw MI and normalized MI.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # Discretize x if continuous
    if len(np.unique(x)) > n_bins:
        x = discretize(x, n_bins)

    x_vals = np.unique(x)
    y_vals = np.unique(y)
    m      = len(x)

    # Joint and marginal counts
    joint  = np.zeros((len(x_vals), len(y_vals)), dtype=np.float64)
    for xi, xv in enumerate(x_vals):
        for yi, yv in enumerate(y_vals):
            joint[xi, yi] = np.sum((x == xv) & (y == yv))

    p_xy  = joint / m
    p_x   = joint.sum(axis=1) / m
    p_y   = joint.sum(axis=0) / m

    # MI = Σ p(x,y) · log(p(x,y) / p(x)·p(y))
    mi = 0.0
    for xi in range(len(x_vals)):
        for yi in range(len(y_vals)):
            if p_xy[xi, yi] > 0:
                mi += p_xy[xi, yi] * np.log(p_xy[xi, yi] / (p_x[xi] * p_y[yi] + 1e-12))

    # Normalized MI
    hx  = _entropy(joint.sum(axis=1))
    hy  = _entropy(joint.sum(axis=0))
    nmi = mi / (np.sqrt(hx * hy) + 1e-12)

    return mi, nmi


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    N_BINS = 10

    # 1. Load raw data
    df     = clean_data(load_data("dataset.csv"))
    target = "default.payment.next.month"
    y      = df[target].values
    features = [c for c in df.columns if c != target]

    # 2. Compute MI for each feature vs target
    print(f"{'Feature':<35} {'MI':>8}  {'NMI':>8}")
    print("-" * 55)

    mi_vals  = []
    nmi_vals = []
    for feat in features:
        mi, nmi = mutual_information(df[feat].values, y, n_bins=N_BINS)
        mi_vals.append(mi)
        nmi_vals.append(nmi)
        print(f"{feat:<35} {mi:>8.4f}  {nmi:>8.4f}")

    mi_vals  = np.array(mi_vals)
    nmi_vals = np.array(nmi_vals)

    # 3. Sort by NMI descending
    order    = np.argsort(nmi_vals)[::-1]
    features_sorted = [features[i] for i in order]
    nmi_sorted      = nmi_vals[order]
    mi_sorted       = mi_vals[order]

    # 4. Plot bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(
        range(len(features_sorted)), nmi_sorted[::-1],
        color=plt.cm.coolwarm(nmi_sorted[::-1] / (nmi_sorted.max() + 1e-12)),
        edgecolor="white", linewidth=0.5,
    )

    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted[::-1], fontsize=9)
    ax.set_xlabel("Normalized Mutual Information (NMI)", fontsize=11)
    ax.set_title(f"Feature → Target Mutual Information\n"
                 f"(target: {target})", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Annotate bars with NMI value
    for i, (nmi, mi) in enumerate(zip(nmi_sorted[::-1], mi_sorted[::-1])):
        ax.text(nmi + 0.002, i, f"{nmi:.3f}", va="center", fontsize=8)

    ax.set_xlim(0, nmi_sorted.max() * 1.15)
    plt.tight_layout()
    plt.savefig("mutual_info.png", dpi=150, bbox_inches="tight")
    print("\nSaved to mutual_info.png")
    plt.show()
