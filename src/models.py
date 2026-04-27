"""
Model definitions for PhysicsAE project.

Classes
-------
PhysicsAE   : Physics-guided self-supervised autoencoder
DeepSVDD    : Deep Support Vector Data Description baseline
"""

import torch
import torch.nn as nn


# ============================================================
# PhysicsAE
# ============================================================

class PhysicsAE(nn.Module):
    """
    Physics-Guided Self-Supervised Autoencoder.

    The encoder maps FFT magnitude spectra to a low-dimensional latent
    space; the decoder reconstructs the spectrum. Anomaly scores are
    the per-sample mean squared reconstruction error.

    Parameters
    ----------
    input_dim : int
        Number of FFT bins (default 513 for n_fft=1024).
    latent_dim : int
        Dimensionality of the bottleneck layer.
    dropout : float
        Dropout probability applied in the encoder.
    """

    def __init__(self, input_dim: int = 513, latent_dim: int = 32, dropout: float = 0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, input_dim) tensor

        Returns
        -------
        x_hat : (B, input_dim) reconstructed spectrum
        z     : (B, latent_dim) latent codes
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error."""
        x_hat, _ = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)


# ============================================================
# Deep SVDD
# ============================================================

class DeepSVDD(nn.Module):
    """
    Deep Support Vector Data Description.

    Maps inputs to a latent space; anomaly score is the squared
    Euclidean distance from a learned hypersphere centre.

    Parameters
    ----------
    input_dim : int
    latent_dim : int
    """

    def __init__(self, input_dim: int = 513, latent_dim: int = 32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim),
        )
        self.centre: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def init_centre(self, loader, device: str, eps: float = 0.1):
        """Initialise hypersphere centre from training data."""
        z_sum = 0
        n = 0
        self.eval()
        with torch.no_grad():
            for batch in loader:
                z = self.forward(batch.to(device))
                z_sum += z.sum(dim=0)
                n += len(batch)
        centre = z_sum / n
        # Avoid centre collapsing near zero
        centre[(centre.abs() < eps) & (centre < 0)] = -eps
        centre[(centre.abs() < eps) & (centre >= 0)] = eps
        self.centre = centre.to(device)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Squared distance from hypersphere centre."""
        if self.centre is None:
            raise RuntimeError("Call init_centre() before computing anomaly scores.")
        z = self.forward(x)
        return ((z - self.centre) ** 2).sum(dim=1)
