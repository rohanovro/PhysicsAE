"""
Training routines for PhysicsAE and baselines.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import PhysicsAE, DeepSVDD


# ── Physics loss helpers ─────────────────────────────────────────────────────

def build_union_mask(
    fault_freqs_hz: dict,
    fs: int,
    n_fft: int,
    n_harmonics: int = 3,
    bandwidth_hz: float = 15.0,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Boolean mask of FFT bins that fall within ±bandwidth_hz of each
    fault frequency harmonic.

    Returns
    -------
    mask : (n_bins,) bool tensor
    """
    n_bins = n_fft // 2 + 1
    freq_bins = torch.fft.rfftfreq(n_fft, d=1.0 / fs).to(device)
    mask = torch.zeros(n_bins, dtype=torch.bool, device=device)

    for ff in fault_freqs_hz.values():
        for k in range(1, n_harmonics + 1):
            centre = k * ff
            mask |= (freq_bins >= centre - bandwidth_hz) & (freq_bins <= centre + bandwidth_hz)

    return mask


def physics_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    fault_freqs_hz: dict,
    fs: int,
    n_fft: int,
    n_harmonics: int = 3,
    bandwidth_hz: float = 15.0,
) -> torch.Tensor:
    """
    Fault-band reconstruction error ratio.

    Penalises poor reconstruction specifically in the frequency bands
    where fault signatures are expected to appear.

    Returns
    -------
    Scalar loss (fault-band MSE / total MSE).
    """
    device = x.device
    mask = build_union_mask(fault_freqs_hz, fs, n_fft, n_harmonics, bandwidth_hz, device)

    fault_band_err = ((x[:, mask] - x_hat[:, mask]) ** 2).mean()
    total_err = ((x - x_hat) ** 2).mean() + 1e-8

    return fault_band_err / total_err


# ── PhysicsAE trainer ────────────────────────────────────────────────────────

def train_physicsae(
    X_train: np.ndarray,
    X_val: np.ndarray,
    fault_freqs: dict,
    config: dict,
    use_physics: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> PhysicsAE:
    """
    Train a PhysicsAE (or plain AE baseline).

    Parameters
    ----------
    X_train     : (N_train, n_bins) scaled training windows
    X_val       : (N_val, n_bins) scaled validation windows
    fault_freqs : dict from config.FAULT_FREQUENCIES
    config      : CONFIG dict
    use_physics : if False, trains a plain reconstruction AE
    seed        : random seed for reproducibility
    verbose     : print progress every 20 epochs

    Returns
    -------
    Trained PhysicsAE with best validation checkpoint loaded.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = config['device']

    # DataLoader
    X_t = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, X_t), batch_size=config['batch_size'], shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    # Model + optimiser
    model = PhysicsAE(
        input_dim=X_train.shape[1],
        latent_dim=config['latent_dim'],
        dropout=config['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        epoch_loss = 0.0

        for x, _ in loader:
            x = x.to(device)
            x_hat, _ = model(x)

            recon = nn.functional.mse_loss(x_hat, x)

            if use_physics:
                phys = physics_loss(
                    x, x_hat, fault_freqs,
                    config['fs_target'], config['n_fft'],
                    config['n_harmonics'], config['bandwidth_hz'],
                )
                loss = recon + config['lambda_physics'] * phys
            else:
                loss = recon

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            x_hat_val, _ = model(X_val_t)
            val_loss = nn.functional.mse_loss(x_hat_val, X_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and epoch % 20 == 0:
            avg_loss = epoch_loss / len(loader)
            label = "PhysicsAE" if use_physics else "BaselineAE"
            print(f"[{label}] Epoch {epoch:3d}/{config['epochs']} | "
                  f"train={avg_loss:.5f} | val={val_loss:.5f}")

    model.load_state_dict(best_state)
    return model


# ── Deep SVDD trainer ────────────────────────────────────────────────────────

def train_deep_svdd(
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: dict,
    seed: int = 42,
    verbose: bool = True,
) -> DeepSVDD:
    """
    Train a Deep SVDD model.

    Returns
    -------
    Trained DeepSVDD with centre initialised.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = config['device']

    X_t = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(X_t, batch_size=config['batch_size'], shuffle=True)

    model = DeepSVDD(input_dim=X_train.shape[1], latent_dim=config['latent_dim']).to(device)

    # Initialise centre
    init_loader = DataLoader(X_t, batch_size=config['batch_size'], shuffle=False)
    model.init_centre(init_loader, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            batch = batch.to(device)
            z = model(batch)
            loss = ((z - model.centre) ** 2).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if verbose and epoch % 20 == 0:
            print(f"[DeepSVDD] Epoch {epoch:3d}/{config['epochs']} | "
                  f"train={epoch_loss/len(loader):.5f}")

    return model
