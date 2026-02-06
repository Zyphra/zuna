# ---------------------------------------------------------------
# eeg_feature_clf.py
# ---------------------------------------------------------------
"""
Quick EEG feature extraction + logistic-regression benchmark.

Public API
----------
run_feature_clf(model_input_all, model_output_all, data_dir, *,
                fs=512, bands=None, k_eig=10,
                test_size=0.2, random_state=42)
    → (accuracy_input, accuracy_output)
"""

from pathlib import Path
import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import entropy, skew, kurtosis

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# ------------------------------------------------------------------
# ░░ 1.  Low-level feature helpers ░░
# ------------------------------------------------------------------
def _band_power(epoch, fs, bands):
    feats = []
    for ch in range(epoch.shape[1]):
        f, Pxx = welch(epoch[:, ch], fs=fs, nperseg=fs * 2)
        for low, high in bands:
            idx = (f >= low) & (f <= high)  
            feats.append(np.trapz(Pxx[idx], f[idx]))
    return feats


def _spectral_entropy(epoch, fs):
    ent = []
    for ch in range(epoch.shape[1]):
        f, Pxx = welch(epoch[:, ch], fs=fs, nperseg=fs * 2)
        Pxx /= Pxx.sum()
        ent.append(entropy(Pxx))
    return ent


def _time_stats(epoch):
    return np.hstack(
        [
            epoch.mean(0),
            epoch.var(0),
            skew(epoch, axis=0),
            kurtosis(epoch, axis=0),
        ]
    )


def _cov_eigen_k(epoch, k=10):
    w, _ = np.linalg.eigh(np.cov(epoch.T))
    return np.log(w[-k:])


def _extract_features(epochs, fs, bands, k_eig):
    feats_all = []
    for ep in epochs:  # ep shape (T, C)
        feats = (
            _band_power(ep, fs, bands)
            + _spectral_entropy(ep, fs)
            + list(_time_stats(ep))
            + list(_cov_eigen_k(ep, k_eig))
        )
        feats_all.append(feats)
    return np.asarray(feats_all, dtype=np.float32)


# ------------------------------------------------------------------
# ░░ 2.  Main entry-point ░░
# ------------------------------------------------------------------
def run_feature_clf(
    model_input_all,
    model_output_all,
    *,
    data_dir,
    fs: int = 512,
    bands=None,
    k_eig: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Parameters
    ----------
    model_input_all, model_output_all : Sequence[np.ndarray]
        Lists of batched EEG tensors with shape (B, T, C).
    data_dir : str | Path
        Path to MOABB data directory (the script expects a sibling
        directory named 'moabb_meta' with .pt files containing labels).
    fs : int
        Sampling rate in Hz.
    bands : list[tuple[float,float]]
        Frequency bands.  If None, defaults to δ/θ/α/β/γ.
    k_eig : int
        Number of top covariance eigen-values to keep.
    Returns
    -------
    acc_in, acc_out : float
        Test-set accuracies for features from model input / output.
    """

    # 0.  Default bands
    if bands is None:
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 80)]

    # 1.  Stack → (N, T, C)
    model_input_arr = np.concatenate(model_input_all, axis=0)
    model_output_arr = np.concatenate(model_output_all, axis=0)

    # 2.  Load labels
    data_dir = Path(data_dir).expanduser().resolve()
    meta_dir = data_dir.with_name("moabb_meta")
    pt_files = sorted(meta_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {meta_dir}")
    labels = (
        torch.load(pt_files[0], map_location="cpu")["labels"][: len(model_output_arr)]
        .numpy()
    )

    # 3.  Remove flat channels
    channel_std = model_input_arr.std(axis=(0, 1))
    good_ch = channel_std > 1e-6
    model_input_arr = model_input_arr[:, :, good_ch]
    model_output_arr = model_output_arr[:, :, good_ch]

    # 4.  Feature matrices
    X_in_feat = _extract_features(model_input_arr, fs, bands, k_eig)
    X_out_feat = _extract_features(model_output_arr, fs, bands, k_eig)

    # 5.  Classifier template
    clf_template = lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=500,
            solver="saga",
            penalty="l2",
            C=0.1,
            n_jobs=-1,
            random_state=random_state,
        ),
    )

    # 6-A.  Train / test on INPUT features
    Xtr, Xte, ytr, yte = train_test_split(
        X_in_feat,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    clf_in = clf_template().fit(Xtr, ytr)
    acc_in = clf_in.score(Xte, yte)

    # 6-B.  Train / test on OUTPUT features
    Xtr, Xte, ytr, yte = train_test_split(
        X_out_feat,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    clf_out = clf_template().fit(Xtr, ytr)
    acc_out = clf_out.score(Xte, yte)
    return acc_in, acc_out