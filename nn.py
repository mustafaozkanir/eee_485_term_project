"""
nn.py
-----
Feed-forward Neural Network from scratch (numpy only).
Binary classification with ReLU hidden layers and sigmoid output.
Class-weighted loss to handle the 78/22 imbalance.
"""

import numpy as np
from data_prep import prepare_data
from utils import (accuracy_score, precision_score, recall_score,
                   f1_score, roc_auc_score_manual, plot_confusion_matrix)

# ============================================================
# CONFIGURATION
# ============================================================

HIDDEN_SIZES   = [16, 8, 4]      # hidden layer widths
LEARNING_RATE  = 0.001
ITERATIONS     = 1000
REG_LAMBDA     = 50
BATCH_SIZE     = 1024
THRESHOLD      = 0.52
POS_WEIGHT     = None          # None = compute from data (n_neg/n_pos); set a number to override
RANDOM_SEED    = 42


# ============================================================
# ACTIVATIONS
# ============================================================

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _relu(z):
    return np.maximum(0.0, z)


def _relu_deriv(z):
    return (z > 0.0).astype(np.float64)


# ============================================================
# NEURAL NETWORK CLASS
# ============================================================

class NeuralNetwork:
    """
    Fully-connected feed-forward network for binary classification.

    Parameters
    ----------
    hidden_sizes   : list of ints — widths of hidden layers
    learning_rate  : step size for gradient descent
    iterations     : number of full passes (epochs) over the data
    reg_lambda     : L2 regularisation strength
    batch_size     : mini-batch size (use len(X) for full-batch GD)
    pos_weight     : upweight for the positive (minority) class
    random_seed    : for reproducibility
    """

    def __init__(
        self,
        hidden_sizes  = None,
        learning_rate = 0.01,
        iterations    = 2000,
        reg_lambda    = 0.1,
        batch_size    = 256,
        pos_weight    = 1.0,
        random_seed   = 42,
    ):
        self.hidden_sizes   = hidden_sizes or [64, 32]
        self.learning_rate  = learning_rate
        self.iterations     = iterations
        self.reg_lambda     = reg_lambda
        self.batch_size     = batch_size
        self.pos_weight     = pos_weight
        self.random_seed    = random_seed
        self.weights        = []
        self.biases         = []
        self.losses         = []

    # ----------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------

    def _build_layer_sizes(self, n_features):
        return [n_features] + self.hidden_sizes + [1]

    def _init_weights(self, layer_sizes):
        rng = np.random.default_rng(self.random_seed)
        self.weights = []
        self.biases  = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialization for ReLU
            W = rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------

    def _forward(self, X):
        """Returns (list of activations, list of pre-activations)."""
        activations     = [X]
        pre_activations = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ W + b
            pre_activations.append(z)

            # ReLU for hidden layers, sigmoid for output
            if i < len(self.weights) - 1:
                a = _relu(z)
            else:
                a = _sigmoid(z)
            activations.append(a)

        return activations, pre_activations

    # ----------------------------------------------------------
    # Loss
    # ----------------------------------------------------------

    def _compute_loss(self, y_pred, y):
        m         = len(y)
        y_pred    = np.clip(y_pred.flatten(), 1e-12, 1 - 1e-12)
        y_flat    = y.flatten()
        # Per sample weight (to tackle class imbalance)
        sw        = np.where(y_flat == 1, self.pos_weight, 1.0)
        # Binary Cross Entropy Loss
        bce       = -np.mean(sw * (y_flat * np.log(y_pred) + (1 - y_flat) * np.log(1 - y_pred)))
        # Regularization Loss
        l2        = (self.reg_lambda / (2 * m)) * sum(np.sum(W ** 2) for W in self.weights)
        return bce + l2

    # ----------------------------------------------------------
    # Backward pass
    # ----------------------------------------------------------

    def _backward(self, y, activations, pre_activations):
        m  = len(y)
        sw = np.where(y.flatten() == 1, self.pos_weight, 1.0).reshape(-1, 1)

        # Output layer delta: d(BCE)/d(z_out) = (a_out - y) * sample_weight
        delta = (activations[-1] - y.reshape(-1, 1)) * sw

        dW_list = []
        db_list = []

        # Starting from the output layer go until 0th layer
        for i in range(len(self.weights) - 1, -1, -1):
            # Calculate the gradient for layer i
            dW = activations[i].T @ delta / m + (self.reg_lambda / m) * self.weights[i]
            db = delta.mean(axis=0, keepdims=True)
            dW_list.insert(0, dW)
            db_list.insert(0, db)

            if i > 0:
                # Propagate delta through ReLU of previous layer
                # Backpropagation formula we learned in the class
                delta = (delta @ self.weights[i].T) * _relu_deriv(pre_activations[i - 1])

        return dW_list, db_list

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        m = len(X)

        layer_sizes = self._build_layer_sizes(X.shape[1])
        self._init_weights(layer_sizes)

        rng         = np.random.default_rng(self.random_seed)
        self.losses = []

        for epoch in range(self.iterations):
            # Shuffle once per epoch
            perm = rng.permutation(m)
            X_s, y_s = X[perm], y[perm]

            for start in range(0, m, self.batch_size):
                X_b = X_s[start : start + self.batch_size]
                y_b = y_s[start : start + self.batch_size]

                activations, pre_activations = self._forward(X_b)
                dW_list, db_list = self._backward(y_b, activations, pre_activations)

                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * dW_list[i]
                    self.biases[i]  -= self.learning_rate * db_list[i]

            if epoch % 100 == 0:
                acts, _ = self._forward(X)
                loss = self._compute_loss(acts[-1], y)
                self.losses.append(loss)
                print(f"Epoch {epoch:4d}/{self.iterations}  loss={loss:.4f}")

        return self

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        activations, _ = self._forward(X)
        return activations[-1].flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # 1. Data
    X_train, X_test, y_train, y_test = prepare_data()

    n_neg      = np.sum(y_train == 0)
    n_pos      = np.sum(y_train == 1)
    pos_weight = POS_WEIGHT if POS_WEIGHT is not None else n_neg / (n_pos + 1e-12)
    print(f"Training samples : {len(X_train)}  |  pos_weight = {pos_weight:.4f}")

    # 2. Build and train
    model = NeuralNetwork(
        hidden_sizes   = HIDDEN_SIZES,
        learning_rate  = LEARNING_RATE,
        iterations     = ITERATIONS,
        reg_lambda     = REG_LAMBDA,
        batch_size     = BATCH_SIZE,
        pos_weight     = pos_weight,
        random_seed    = RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    # 3. Evaluate
    y_prob = model.predict_proba(X_test)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    print("\n── Test Results ──────────────────────────────")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score_manual(y_test, y_prob):.4f}")

    # 4. Confusion matrix
    plot_confusion_matrix(
        y_test, y_pred,
        class_names=["No Default", "Default"],
        title="Neural Network — Confusion Matrix",
        save_path="nn_confusion_matrix.png",
    )