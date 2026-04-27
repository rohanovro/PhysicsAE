"""
Data loader for the Paderborn University bearing dataset.

Reads .mat vibration files directly from the downloaded .zip archive,
resamples to a target frequency, computes FFT magnitude windows, and
returns train/val/test splits with no data leakage.
"""

import io
import zipfile
from math import gcd

import numpy as np
import scipy.io
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Bearing folder labels ────────────────────────────────────────────────────

HEALTHY_FOLDERS = ['K001', 'K002', 'K003', 'K004', 'K005', 'K006']

FAULT_FOLDERS = {
    'inner': ['KI04', 'KI14', 'KI16'],
    'outer': ['KA04', 'KA15', 'KA16'],
    'mixed': ['KB23'],
}


# ── Low-level helpers ────────────────────────────────────────────────────────

def _read_vibration_from_mat(zip_ref: zipfile.ZipFile, file_path: str) -> np.ndarray | None:
    """Extract the vibration channel from a single .mat file inside a zip."""
    with zip_ref.open(file_path) as f:
        mat = scipy.io.loadmat(io.BytesIO(f.read()), simplify_cells=True)

    for key, val in mat.items():
        if key.startswith('_'):
            continue
        if isinstance(val, dict) and 'Y' in val and len(val['Y']) > 6:
            try:
                return val['Y'][6]['Data'].flatten().astype(np.float32)
            except (KeyError, IndexError, AttributeError):
                continue
    return None


def _load_bearing_from_zip(zip_path: str, folder_name: str, n_files: int = 20) -> np.ndarray:
    """Concatenate vibration data from the first *n_files* .mat files of a folder."""
    segments = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        mat_files = sorted(
            f for f in zf.namelist()
            if f.startswith(folder_name + '/') and f.endswith('.mat')
        )[:n_files]

        for path in mat_files:
            sig = _read_vibration_from_mat(zf, path)
            if sig is not None:
                segments.append(sig)

    return np.concatenate(segments) if segments else np.array([], dtype=np.float32)


def _resample(signal: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    """Polyphase resample signal from fs_orig → fs_target."""
    g = gcd(fs_target, fs_orig)
    return resample_poly(signal, fs_target // g, fs_orig // g).astype(np.float32)


def _normalize(segment: np.ndarray) -> np.ndarray:
    """Unit-variance normalisation; safe against zero-std segments."""
    std = segment.std()
    return segment / std if std > 1e-8 else segment


def _to_fft_windows(signal: np.ndarray, window_size: int, hop_size: int) -> np.ndarray:
    """
    Sliding-window FFT magnitude spectrogram.

    Returns
    -------
    np.ndarray of shape (n_windows, n_fft//2 + 1)
    """
    hann = np.hanning(window_size)
    n_bins = window_size // 2 + 1
    n_windows = (len(signal) - window_size) // hop_size + 1
    out = np.zeros((n_windows, n_bins), dtype=np.float32)

    for i in range(n_windows):
        seg = signal[i * hop_size: i * hop_size + window_size]
        if len(seg) < window_size:
            break
        out[i] = np.abs(np.fft.rfft(_normalize(seg) * hann)) / window_size

    return out


# ── Main pipeline ────────────────────────────────────────────────────────────

def load_paderborn_data(zip_path: str, config: dict):
    """
    Full Paderborn loading pipeline.

    Parameters
    ----------
    zip_path : str
        Path to `paderborn-db.zip`.
    config : dict
        CONFIG dict from src/config.py.

    Returns
    -------
    X_train_sc : np.ndarray  (scaled, healthy train windows)
    X_val_sc   : np.ndarray  (scaled, healthy val windows)
    X_fault_sc : dict        {group: scaled fault windows}
    scaler     : StandardScaler  fitted on training data only
    """
    FS    = config['fs_original']
    FS_T  = config['fs_target']
    WS    = config['window_size']
    HS    = config['hop_size']

    print("=" * 60)
    print("Loading Paderborn dataset...")
    print("=" * 60)

    # ── Healthy bearings ──────────────────────────────────────
    healthy_windows = []
    for folder in HEALTHY_FOLDERS:
        sig = _load_bearing_from_zip(zip_path, folder, n_files=20)
        if len(sig) == 0:
            print(f"  ✗ {folder}: not found")
            continue
        wins = _to_fft_windows(_resample(sig, FS, FS_T), WS, HS)
        healthy_windows.append(wins)
        print(f"  ✓ {folder}: {len(wins):,} windows")

    X_healthy = np.vstack(healthy_windows)
    print(f"\nTotal healthy windows: {X_healthy.shape}")

    # ── Fault bearings ────────────────────────────────────────
    X_fault: dict[str, np.ndarray] = {}
    for group, folders in FAULT_FOLDERS.items():
        group_wins = []
        for folder in folders:
            sig = _load_bearing_from_zip(zip_path, folder, n_files=20)
            if len(sig) == 0:
                print(f"  ✗ {folder}: not found")
                continue
            group_wins.append(_to_fft_windows(_resample(sig, FS, FS_T), WS, HS))
            print(f"  ✓ {folder}")
        if group_wins:
            X_fault[group] = np.vstack(group_wins)
            print(f"  → {group}: {X_fault[group].shape}")

    # ── Train / val split + scaling (NO leakage) ──────────────
    X_train, X_val = train_test_split(X_healthy, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_val_sc   = scaler.transform(X_val).astype(np.float32)
    X_fault_sc = {g: scaler.transform(Xf).astype(np.float32) for g, Xf in X_fault.items()}

    print(f"\nTrain : {X_train_sc.shape}")
    print(f"Val   : {X_val_sc.shape}")
    for g, Xf in X_fault_sc.items():
        print(f"{g:6s}: {Xf.shape}")

    return X_train_sc, X_val_sc, X_fault_sc, scaler
