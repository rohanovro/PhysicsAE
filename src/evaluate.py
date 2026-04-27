"""
Evaluation utilities for PhysicsAE.

Functions
---------
get_anomaly_scores   : compute anomaly scores with optional MC-Dropout
evaluate_model       : compute AUC, F1, precision, recall
evaluate_all_models  : compare PhysicsAE vs baselines
print_results_table  : pretty-print comparison table
save_results_csv     : save results to CSV in results/tables/
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def get_anomaly_scores(
    model,
    X: np.ndarray,
    config: dict,
    mc_dropout: bool = False,
) -> np.ndarray:
    """
    Compute anomaly scores for a batch of windows.

    Parameters
    ----------
    model      : PhysicsAE or DeepSVDD instance
    X          : (N, n_bins) numpy array
    config     : CONFIG dict (for device and n_mc_samples)
    mc_dropout : if True, enable dropout at inference for uncertainty estimation

    Returns
    -------
    scores : (N,) numpy array
    """
    device = config['device']
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    if mc_dropout:
        model.train()   # keep dropout active
        all_scores = []
        with torch.no_grad():
            for _ in range(config['n_mc_samples']):
                all_scores.append(model.anomaly_score(X_t).cpu().numpy())
        scores = np.stack(all_scores).mean(axis=0)
    else:
        model.eval()
        with torch.no_grad():
            scores = model.anomaly_score(X_t).cpu().numpy()

    return scores


def evaluate_model(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold_percentile: float = 95.0,
) -> dict:
    """
    Evaluate binary anomaly detection performance.

    Parameters
    ----------
    scores               : (N,) anomaly scores (higher = more anomalous)
    y_true               : (N,) binary labels  (1 = fault, 0 = healthy)
    threshold_percentile : percentile of scores used as decision threshold

    Returns
    -------
    dict with keys: AUC, F1, Precision, Recall, Threshold
    """
    threshold = np.percentile(scores, threshold_percentile)
    y_pred = (scores >= threshold).astype(int)

    return {
        'AUC':       round(roc_auc_score(y_true, scores), 4),
        'F1':        round(f1_score(y_true, y_pred, zero_division=0), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_true, y_pred, zero_division=0), 4),
        'Threshold': round(float(threshold), 6),
    }


def build_test_arrays(X_val_sc: np.ndarray, X_fault_sc: dict) -> tuple:
    """
    Combine validation (healthy) and fault windows into a single test array.

    Returns
    -------
    X_test : (N_test, n_bins)
    y_test : (N_test,) binary labels
    """
    fault_arrays = list(X_fault_sc.values())
    X_test = np.vstack([X_val_sc] + fault_arrays)
    y_test = np.concatenate([
        np.zeros(len(X_val_sc)),
        *[np.ones(len(Xf)) for Xf in fault_arrays],
    ])
    return X_test, y_test


def evaluate_all_models(models: dict, X_test: np.ndarray, y_test: np.ndarray, config: dict) -> pd.DataFrame:
    """
    Evaluate all models and return a comparison DataFrame.

    Parameters
    ----------
    models  : dict {model_name: model_instance}
    X_test  : (N, n_bins)
    y_test  : (N,)
    config  : CONFIG dict

    Returns
    -------
    pd.DataFrame with one row per model
    """
    rows = []
    for name, model in models.items():
        scores = get_anomaly_scores(model, X_test, config)
        metrics = evaluate_model(scores, y_test)
        metrics['Model'] = name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index('Model')[['AUC', 'F1', 'Precision', 'Recall']]
    return df


def print_results_table(df: pd.DataFrame):
    """Pretty-print the results DataFrame."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60)


def save_results_csv(df: pd.DataFrame, path: str = 'results/tables/results.csv'):
    """Save results DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"Results saved to {path}")
