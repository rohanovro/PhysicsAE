"""
Configuration parameters for PhysicsAE.
All hyperparameters, architecture settings, and physical constants live here.
"""

import torch

# ============================================================
# MAIN CONFIG DICTIONARY
# ============================================================

CONFIG = {
    # --- Signal Processing ---
    'fs_original': 64000,       # Original sampling rate (Hz)
    'fs_target': 12000,         # Resampled rate (Hz)
    'window_size': 1024,        # FFT window size (samples)
    'hop_size': 256,            # FFT hop size (samples)
    'n_fft': 1024,              # FFT size
    'n_bins': 513,              # FFT output bins (n_fft//2 + 1)

    # --- Physics Loss ---
    'bandwidth_hz': 15.0,       # ±bandwidth around each fault harmonic (Hz)
    'n_harmonics': 3,           # Number of harmonics to penalise
    'lambda_physics': 0.3,      # Physics loss weight

    # --- Model Architecture ---
    'latent_dim': 32,
    'dropout': 0.2,
    'encoder_layers': [256, 128, 64],
    'decoder_layers': [64, 128, 256],

    # --- Training ---
    'batch_size': 256,
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'grad_clip_norm': 1.0,

    # --- MC Dropout Uncertainty ---
    'n_mc_samples': 30,

    # --- Reproducibility ---
    'seeds': [42, 123, 456, 789, 2024],

    # --- Device ---
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# ============================================================
# BEARING SPECIFICATION  (SKF 6203 / Paderborn dataset)
# ============================================================

BEARING_SPEC = {
    'N': 8,           # Number of rolling elements
    'd_mm': 7.94,     # Ball diameter (mm)
    'D_mm': 46.0,     # Pitch diameter (mm)
    'theta_deg': 0.0, # Contact angle (degrees)
    'rpm': 1500,      # Operating speed (RPM)
}


def compute_fault_frequencies(spec: dict = BEARING_SPEC) -> dict:
    """
    Calculate bearing fault characteristic frequencies.

    Returns
    -------
    dict with keys: BPFO, BPFI, BSF, FTF  (all in Hz)
    """
    fr = spec['rpm'] / 60.0
    N  = spec['N']
    d_D = spec['d_mm'] / spec['D_mm']

    return {
        'BPFO': (N / 2) * (1 - d_D) * fr,
        'BPFI': (N / 2) * (1 + d_D) * fr,
        'BSF' : (spec['D_mm'] / (2 * spec['d_mm'])) * (1 - d_D ** 2) * fr,
        'FTF' : (1 / 2) * (1 - d_D) * fr,
    }


FAULT_FREQUENCIES = compute_fault_frequencies()

if __name__ == '__main__':
    print("Fault Frequencies:")
    for name, freq in FAULT_FREQUENCIES.items():
        print(f"  {name}: {freq:.2f} Hz")
    print(f"\nDevice: {CONFIG['device']}")
