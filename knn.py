"""
knn.py
------
k-Nearest Neighbors classifier implemented from scratch.
"""

import numpy as np
from data_prep import prepare_data
from utils import precision_score, recall_score, f1_score, plot_confusion_matrix


class KNearestNeighbors:
    """
    k-Nearest Neighbors classifier.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to consider.
    """

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Training — just memorise the data
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbors":
        self._X_train = np.array(X, dtype=float)
        self._y_train = np.array(y)

        # Compute per-class vote weights = n_majority / n_class so that
        # minority-class neighbours count proportionally more in the vote.
        classes, counts = np.unique(self._y_train, return_counts=True)
        max_count = counts.max()
        self._class_weights = {cls: max_count / cnt
                               for cls, cnt in zip(classes, counts)}
        return self

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _euclidean_distances(X: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between every test point and
        every training point without an explicit Python loop over rows.

        Uses the identity:
            ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b^T

        Returns shape (n_test, n_train).
        """
        # Squared norms
        sq_test  = np.sum(X ** 2,       axis=1, keepdims=True)  # (n_test,  1)
        sq_train = np.sum(X_train ** 2, axis=1, keepdims=True)  # (n_train, 1)
        dot      = X @ X_train.T                                 # (n_test, n_train)

        dist_sq = sq_test + sq_train.T - 2.0 * dot
        # Avoid numerical noise that can push tiny negatives slightly below 0
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.sqrt(dist_sq)

    def _majority_vote(self, neighbor_labels: np.ndarray) -> int:
        """
        Weighted vote: each neighbour's vote is scaled by its class weight.
        Class 1 (minority) votes count ~3.52× more than class 0 votes.
        """
        weighted_scores = {}
        for label in neighbor_labels:
            weighted_scores[label] = (weighted_scores.get(label, 0.0)
                                      + self._class_weights[label])
        return int(max(weighted_scores, key=weighted_scores.get))

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """
        Predict class labels for every row in X.
        Batching is used.
        """
        if self._X_train is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.array(X, dtype=float)
        n_test  = X.shape[0]
        n_train = self._X_train.shape[0]
        preds   = np.empty(n_test, dtype=self._y_train.dtype)

        for start in range(0, n_test, batch_size):
            end   = min(start + batch_size, n_test)
            batch = X[start:end]                                    # (B, d)

            dists = self._euclidean_distances(batch, self._X_train) # (B, n_train)

            # Indices of k nearest neighbours (partial sort)
            k_idx = np.argpartition(dists, self.k, axis=1)[:, :self.k]

            for i, idx in enumerate(k_idx):
                preds[start + i] = self._majority_vote(self._y_train[idx])

        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on (X, y)."""
        preds = self.predict(X)
        return float(np.mean(preds == y))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    k = 512
    model = KNearestNeighbors(k=k)
    model.fit(X_train, y_train)

    print(f"\nEvaluating kNN (k={k}) with weighted voting...")
    y_pred = model.predict(X_test)

    accuracy  = float(np.mean(y_pred == y_test))
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    print(f"\nkNN (k={k}) Results")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    plot_confusion_matrix(
        y_test, y_pred,
        class_names=["No Default", "Default"],
        title=f"kNN (k={k}) — Confusion Matrix",
        save_path="knn_confusion_matrix.png",
    )

