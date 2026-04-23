import numpy as np
from utils import (precision_score, recall_score, f1_score,
                   accuracy_score, roc_auc_score_manual, plot_confusion_matrix)
from data_prep import prepare_data

# ---------------------------------------------------------------------------
# Impurity helpers
# ---------------------------------------------------------------------------

def _entropy(y: np.ndarray) -> float:
    n = len(y)
    if n == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return float(-np.sum(probs * np.log2(probs + 1e-12)))

# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = (
        "feature_index", "threshold",
        "left", "right",
        "is_leaf", "leaf_class", "prob"
    )

    def __init__(self):
        self.feature_index: int | None = None
        self.threshold: float | None = None
        self.left: "_Node | None" = None
        self.right: "_Node | None" = None
        self.is_leaf: bool = False
        self.leaf_class: int | None = None
        self.prob: float | None = None # Probability of class 1

    @classmethod
    def leaf(cls, label: int, prob: float) -> "_Node":
        node = cls()
        node.is_leaf = True
        node.leaf_class = label
        node.prob = prob
        return node

# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | None = None,
        random_seed: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self._rng = np.random.default_rng(random_seed)
        self._root: _Node | None = None

    def _majority_class(self, y: np.ndarray) -> int:
        labels, counts = np.unique(y, return_counts=True)
        return int(labels[np.argmax(counts)])

    def _feature_subset(self, n_total: int) -> np.ndarray:
        if self.max_features is None:
            return np.arange(n_total)
        k = max(1, int(self.max_features * n_total)) if isinstance(self.max_features, float) else max(1, int(self.max_features))
        return self._rng.choice(n_total, size=min(k, n_total), replace=False)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n = len(y)
        best_feat, best_thresh, best_gain = None, None, -np.inf
        feature_indices = self._feature_subset(X.shape[1])
        classes = np.unique(y)
        n_classes = len(classes)
        class_map = {c: i for i, c in enumerate(classes)}
        y_idx = np.array([class_map[yi] for yi in y], dtype=np.int32)
        parent_entropy = _entropy(y)

        def _vec_entropy(counts: np.ndarray) -> np.ndarray:
            p = counts / (counts.sum(axis=1, keepdims=True) + 1e-12)
            return -np.sum(p * np.log2(p + 1e-12), axis=1)

        for feat in feature_indices:
            col = X[:, feat]
            order = np.argsort(col, kind="stable")
            col_s, y_s = col[order], y_idx[order]
            split_pos = np.where(col_s[1:] != col_s[:-1])[0] + 1
            if len(split_pos) == 0: continue

            onehot = np.zeros((n, n_classes))
            onehot[np.arange(n), y_s] = 1.0
            prefix = np.cumsum(onehot, axis=0)
            lc, rc = prefix[split_pos - 1], prefix[-1] - prefix[split_pos - 1]
            nl, nr = split_pos.astype(float), n - split_pos.astype(float)

            valid = (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
            if not valid.any(): continue

            gains = parent_entropy - (nl[valid]/n)*_vec_entropy(lc[valid]) - (nr[valid]/n)*_vec_entropy(rc[valid])
            idx = np.argmax(gains)
            if gains[idx] > best_gain:
                best_gain, best_feat = float(gains[idx]), feat
                best_thresh = float((col_s[split_pos[valid][idx]-1] + col_s[split_pos[valid][idx]]) / 2.0)
        
        return best_feat, best_thresh, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        prob_positive = np.sum(y == 1) / len(y) if len(y) > 0 else 0.0
        if (len(y) < self.min_samples_split or (self.max_depth and depth >= self.max_depth) or len(np.unique(y)) == 1):
            return _Node.leaf(self._majority_class(y), prob_positive)

        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 0.0:
            return _Node.leaf(self._majority_class(y), prob_positive)

        node = _Node()
        node.feature_index, node.threshold, node.prob = feat, thresh, prob_positive
        mask = X[:, feat] <= thresh
        node.left, node.right = self._build(X[mask], y[mask], depth + 1), self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self._root = self._build(np.array(X, dtype=float), np.array(y), 0)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        def _walk(x, node):
            if node.is_leaf: return node.prob
            return _walk(x, node.left) if x[node.feature_index] <= node.threshold else _walk(x, node.right)
        return np.array([_walk(row, self._root) for row in np.array(X, dtype=float)])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5):
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)
        return {
            "accuracy"  : accuracy_score(y, y_pred),
            "precision" : precision_score(y, y_pred),
            "recall"    : recall_score(y, y_pred),
            "f1_score"  : f1_score(y, y_pred),
            "roc_auc"   : roc_auc_score_manual(y, y_prob),
            "y_pred"    : y_pred,
        }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f" Train: {X_train.shape} Test: {X_test.shape}")

    # Hyper-parameters
    max_depth = 6
    min_samples_split = 10
    min_samples_leaf = 5

    

    print(
        f"\nTraining Decision Tree "
        f"(max_depth={max_depth}, min_samples_split={min_samples_split}, "
        f"min_samples_leaf={min_samples_leaf})..."
    )
    model = DecisionTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X_train, y_train)

    threshold = 0.3
    print("\nEvaluating Decision Tree on test set...")
    metrics = model.evaluate(X_test, y_test, threshold=threshold)

    y_pred    = metrics["y_pred"]
    accuracy  = metrics["accuracy"]
    precision = metrics["precision"]
    recall    = metrics["recall"]
    f1        = metrics["f1_score"]
    roc_auc   = metrics["roc_auc"]

    print(f"\nDecision Tree Results")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    plot_confusion_matrix(
        y_test, y_pred,
        class_names=["No Default", "Default"],
        title=f"Decision Tree (depth={max_depth}) — Confusion Matrix",
        save_path="dt_confusion_matrix.png",
    )
    print("\nPlot saved as 'dt_confusion_matrix.png'.")