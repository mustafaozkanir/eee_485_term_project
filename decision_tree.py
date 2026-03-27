"""
decision_tree.py
----------------
Decision Tree classifier implemented from scratch using numpy and pandas.
Splitting criterion: Information Gain (Entropy).
No scikit-learn.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Impurity helpers
# ---------------------------------------------------------------------------

def _entropy(y: np.ndarray) -> float:
    """Shannon entropy of label array y."""
    n = len(y)
    if n == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    # 0 * log2(0) is treated as 0
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _information_gain(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    n  = len(y_parent)
    if n == 0:
        return 0.0
    nl, nr = len(y_left), len(y_right)
    return _entropy(y_parent) - (nl / n) * _entropy(y_left) - (nr / n) * _entropy(y_right)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = (
        "feature_index", "threshold",
        "left", "right",
        "is_leaf", "leaf_class",
    )

    def __init__(self):
        self.feature_index: int | None  = None
        self.threshold: float | None    = None
        self.left:  "_Node | None"      = None
        self.right: "_Node | None"      = None
        self.is_leaf: bool              = False
        self.leaf_class: int | None     = None

    @classmethod
    def leaf(cls, label: int) -> "_Node":
        node = cls()
        node.is_leaf    = True
        node.leaf_class = label
        return node


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

class DecisionTree:
    """
    Binary Decision Tree using Information Gain (Entropy) as the criterion.

    Parameters
    ----------
    max_depth : int | None
        Maximum tree depth. None → unlimited (may overfit).
    min_samples_split : int
        Minimum number of samples required to attempt a split.
    min_samples_leaf : int
        Minimum samples in each resulting child node.
    max_features : int | float | None
        Number of features to consider at each split:
        - None  → all features
        - int   → exactly that many features
        - float → fraction of total features
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | None = None,
        random_seed: int = 42,
    ):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self._rng              = np.random.default_rng(random_seed)
        self._root: _Node | None = None
        self._n_features: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _majority_class(self, y: np.ndarray) -> int:
        labels, counts = np.unique(y, return_counts=True)
        return int(labels[np.argmax(counts)])

    def _feature_subset(self, n_total: int) -> np.ndarray:
        if self.max_features is None:
            return np.arange(n_total)
        if isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_total))
        else:
            k = max(1, int(self.max_features))
        k = min(k, n_total)
        return self._rng.choice(n_total, size=k, replace=False)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Find the (feature_index, threshold) pair that maximises Information Gain.
        Returns (best_feat, best_thresh, best_gain) or (None, None, -inf) if no
        valid split exists.

        Vectorised via prefix sums — no Python loop over thresholds.
        Complexity: O(n_features * n_samples * log(n_samples)) instead of O(n²).
        """
        n = len(y)
        best_feat, best_thresh, best_gain = None, None, -np.inf
        feature_indices = self._feature_subset(X.shape[1])

        # Encode labels to integer indices once for this node
        classes = np.unique(y)
        n_classes = len(classes)
        class_map = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_map[yi] for yi in y], dtype=np.int32)

        # Parent entropy is constant across all features at this node
        parent_entropy = _entropy(y)

        def _vec_entropy(counts: np.ndarray) -> np.ndarray:
            """Entropy for each row of a (m, n_classes) count matrix."""
            totals = counts.sum(axis=1, keepdims=True)          # (m, 1)
            p = counts / (totals + 1e-12)
            return -np.sum(p * np.log2(p + 1e-12), axis=1)      # (m,)

        for feat in feature_indices:
            col   = X[:, feat]
            order = np.argsort(col, kind="stable")
            col_s = col[order]
            y_s   = y_idx[order]

            # Split positions: indices where the feature value changes
            split_pos = np.where(col_s[1:] != col_s[:-1])[0] + 1  # shape (n_splits,)
            if len(split_pos) == 0:
                continue

            # Build prefix class-count matrix via cumsum over one-hot rows
            onehot = np.zeros((n, n_classes), dtype=np.float64)
            onehot[np.arange(n), y_s] = 1.0
            prefix = np.cumsum(onehot, axis=0)                  # (n, n_classes)

            # Counts in left / right child for every candidate split
            left_counts  = prefix[split_pos - 1]                # (n_splits, n_classes)
            right_counts = prefix[-1] - left_counts

            nl = split_pos.astype(np.float64)                   # (n_splits,)
            nr = n - nl

            # Drop splits that violate min_samples_leaf
            valid = (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
            if not valid.any():
                continue

            lc = left_counts[valid]
            rc = right_counts[valid]
            nl_v = nl[valid]
            nr_v = nr[valid]
            sp_v = split_pos[valid]

            # Vectorised information gain for all valid splits at once
            gains = (parent_entropy
                     - (nl_v / n) * _vec_entropy(lc)
                     - (nr_v / n) * _vec_entropy(rc))

            i_best = int(np.argmax(gains))
            if gains[i_best] > best_gain:
                best_gain   = float(gains[i_best])
                best_feat   = feat
                pos         = sp_v[i_best]
                best_thresh = float((col_s[pos - 1] + col_s[pos]) / 2.0)

        return best_feat, best_thresh, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples = len(y)

        # --- Stopping criteria ---
        if (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(np.unique(y)) == 1
        ):
            return _Node.leaf(self._majority_class(y))

        feat, thresh, gain = self._best_split(X, y)

        # No valid split found
        if feat is None or gain <= 0.0:
            return _Node.leaf(self._majority_class(y))

        left_mask  = X[:, feat] <= thresh
        right_mask = ~left_mask

        node = _Node()
        node.feature_index = feat
        node.threshold     = thresh
        node.left  = self._build(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        X = np.array(X, dtype=float)
        y = np.array(y)
        self._n_features = X.shape[1]
        self._root = self._build(X, y, depth=0)
        return self

    def _predict_single(self, x: np.ndarray, node: _Node) -> int:
        if node.is_leaf:
            return node.leaf_class
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._root is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.array(X, dtype=float)
        return np.array([self._predict_single(row, self._root) for row in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_prep import prepare_data

    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    # Hyper-parameters — depth=10 balances accuracy vs training speed
    max_depth        = 10
    min_samples_split = 10
    min_samples_leaf  = 5

    print(
        f"\nTraining Decision Tree  "
        f"(max_depth={max_depth}, min_samples_split={min_samples_split}, "
        f"min_samples_leaf={min_samples_leaf})..."
    )
    model = DecisionTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X_train, y_train)

    print("Evaluating on test set...")
    accuracy = model.score(X_test, y_test)
    print(f"\nDecision Tree Test Accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")
