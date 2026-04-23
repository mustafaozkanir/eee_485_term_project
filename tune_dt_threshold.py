"""
tune_threshold.py
-----------------
Cross-validation to find the optimal classification threshold for a 
pre-tuned Decision Tree.
"""

import numpy as np
import matplotlib.pyplot as plt
from dt3 import DecisionTree
from data_prep import prepare_data

def evaluate_threshold_cv(X, y, best_depth, thresholds, n_splits=5):
    from tune_dt import get_cv_splits, calculate_metrics
    
    splits = list(get_cv_splits(X, y, n_splits=n_splits))
    
    # Store results: {threshold: [fold_f1_scores]}
    threshold_f1s = {t: [] for t in thresholds}
    threshold_recalls = {t: [] for t in thresholds}

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        X_cv_train, y_cv_train = X[tr_idx], y[tr_idx]
        X_cv_val,   y_cv_val   = X[val_idx], y[val_idx]

        # Use the best depth found previously
        model = DecisionTree(max_depth=best_depth)
        model.fit(X_cv_train, y_cv_train)
        
        # Get probabilities once per fold
        probs = model.predict_proba(X_cv_val)

        for t in thresholds:
            preds = (probs >= t).astype(int)
            metrics = calculate_metrics(y_cv_val, preds)
            threshold_f1s[t].append(metrics['f1'])
            threshold_recalls[t].append(metrics['recall'])
            
        print(f"Fold {fold_idx + 1} processed.")

    # Calculate means
    avg_f1s = {t: np.mean(f1s) for t, f1s in threshold_f1s.items()}
    avg_recalls = {t: np.mean(recs) for t, recs in threshold_recalls.items()}
    
    return avg_f1s, avg_recalls

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Range of thresholds to test
    thresholds = np.linspace(0.1, 0.9, 17) 
    best_depth = 8 # From your previous CV results

    print(f"Tuning threshold for max_depth={best_depth}...")
    avg_f1s, avg_recs = evaluate_threshold_cv(X_train, y_train, best_depth, thresholds)

    # Find best threshold
    best_t = max(avg_f1s, key=avg_f1s.get)
    
    print("\n" + "="*30)
    print(f"Best Threshold: {best_t:.2f}")
    print(f"Max Val F1:    {avg_f1s[best_t]:.4f}")
    print(f"Avg Val Recall: {avg_recs[best_t]:.4f}")
    print("="*30)

    # Plotting the trade-off
    plt.figure(figsize=(10, 6))
    t_vals = sorted(avg_f1s.keys())
    plt.plot(t_vals, [avg_f1s[t] for t in t_vals], label='F1 Score', marker='o')
    plt.plot(t_vals, [avg_recs[t] for t in t_vals], label='Recall', linestyle='--')
    plt.axvline(best_t, color='red', linestyle=':', label=f'Best T={best_t:.2f}')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning: F1 vs Recall")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("threshold_tuning.png")
    plt.show()