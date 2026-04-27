"""
Miscellaneous utility functions for PhysicsAE.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model: torch.nn.Module, path: str):
    """Save model state dict to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")


def load_model(model: torch.nn.Module, path: str, device: str = 'cpu') -> torch.nn.Module:
    """Load model state dict from *path*."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    print(f"Model loaded ← {path}")
    return model


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def moving_average(values: list, window: int = 5) -> np.ndarray:
    """Smooth a list of values with a simple moving average."""
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')
