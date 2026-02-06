import numpy as np
from scipy.signal import welch
from scipy.stats import entropy, skew, kurtosis
from itertools import combinations

# ------------------------------------------------------------------
# model_input_arr : shape (N, T, C)   -- your existing array
fs = 256                              # sampling rate (Hz)

def band_power(epoch, fs, bands):
    """Return list of band powers for one epoch (T, C)."""
    bp = []
    for ch in range(epoch.shape[1]):
        f, Pxx = welch(epoch[:, ch], fs=fs, nperseg=fs*2)
        for low, high in bands:
            idx = np.logical_and(f >= low, f <= high)
            bp.append(np.trapz(Pxx[idx], f[idx]))
    return bp

def spectral_entropy(epoch, fs):
    """Shannon entropy of the normalised periodogram per channel."""
    ent = []
    for ch in range(epoch.shape[1]):
        f, Pxx = welch(epoch[:, ch], fs=fs, nperseg=fs*2)
        Pxx /= Pxx.sum()
        ent.append(entropy(Pxx))
    return ent

def time_stats(epoch):
    """Mean, var, skew, kurtosis per channel."""
    return np.hstack([epoch.mean(0),
                      epoch.var(0),
                      skew(epoch, axis=0),
                      kurtosis(epoch, axis=0)])

def upper_tri_feat(matrix):
    """Flatten upper-triangular (including diag)."""
    triu_idx = np.triu_indices_from(matrix)
    return matrix[triu_idx]

def spatial_cov(epoch):
    """(C, C) covariance for one epoch, flattened."""
    cov = np.cov(epoch.T)
    return upper_tri_feat(cov)

# ------------------------------------------------------------------
bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 80)]

feature_list = []
for ep in model_input_arr:                        # iterate over trials
    feats = ( band_power(ep, fs, bands) +
              spectral_entropy(ep, fs) +
              list(time_stats(ep)) +
              list(spatial_cov(ep)) )
    feature_list.append(feats)

X_feat = np.array(feature_list)                   # shape (N, n_features)