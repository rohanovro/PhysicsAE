"""
SHAP-based frequency explainability for PhysicsAE.

Identifies which FFT frequency bins drive the anomaly score for each sample,
enabling interpretable fault diagnosis.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import shap
import torch


def compute_shap_values(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    config: dict,
    n_background: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute SHAP values for anomaly scores using DeepExplainer.

    Parameters
    ----------
    model        : trained PhysicsAE
    X_background : healthy training windows for SHAP background (N_bg, n_bins)
    X_explain    : windows to explain (N_exp, n_bins)
    config       : CONFIG dict
    n_background : number of background samples to subsample
    seed         : random seed

    Returns
    -------
    shap_values : (N_exp, n_bins) numpy array
    """
    np.random.seed(seed)
    device = config['device']
    model.eval()

    # Subsample background
    idx = np.random.choice(len(X_background), min(n_background, len(X_background)), replace=False)
    bg = torch.tensor(X_background[idx], dtype=torch.float32).to(device)
    X_exp_t = torch.tensor(X_explain, dtype=torch.float32).to(device)

    # Wrap model so SHAP sees the anomaly score as output
    class AnomalyWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.anomaly_score(x).unsqueeze(1)

    wrapper = AnomalyWrapper(model).to(device)

    explainer = shap.DeepExplainer(wrapper, bg)
    shap_values = explainer.shap_values(X_exp_t)

    # shap_values may be list-of-arrays; squeeze to (N, n_bins)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    return np.array(shap_values).squeeze()


def plot_mean_shap(
    shap_values: np.ndarray,
    fs: int,
    n_fft: int,
    fault_freqs: dict,
    title: str = 'Mean |SHAP| by Frequency Bin',
    save_path: str | None = None,
):
    """
    Plot mean absolute SHAP values across frequency bins,
    with fault frequency harmonics annotated.

    Parameters
    ----------
    shap_values  : (N, n_bins) array
    fs           : sampling rate (Hz)
    n_fft        : FFT size
    fault_freqs  : dict from config.FAULT_FREQUENCIES
    title        : figure title
    save_path    : if given, save figure to this path
    """
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mean_shap = np.abs(shap_values).mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(freq_axis, mean_shap, linewidth=0.8, color='steelblue', label='Mean |SHAP|')

    colors = ['red', 'green', 'orange', 'purple']
    for (name, ff), color in zip(fault_freqs.items(), colors):
        for k in range(1, 4):
            ax.axvline(k * ff, color=color, alpha=0.5, linewidth=0.8,
                       label=f'{name} h{k}' if k == 1 else None)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Mean |SHAP value|')
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=4)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"SHAP figure saved to {save_path}")

    plt.show()
    return fig
