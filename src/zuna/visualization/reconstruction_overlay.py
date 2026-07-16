"""
Full-duration, multi-channel input-vs-reconstruction overlay for the direct-.fif (v4) path.

This is the primary evaluation figure: every channel is drawn over the whole recording,
with the original input and the model reconstruction on shared axes, and the regions the
model *inferred* (bad channels / BAD_ annotation spans / requested channels / upsampled
channels) shaded in the background.

Channels are de-meaned per trace for display so the input and reconstruction overlay
cleanly (the saved .fif keeps the true, inverse-z-scored volts; de-meaning is view-only).
"""
from pathlib import Path

import numpy as np
import mne


def _contiguous_runs(flags):
    """Yield (start, end) index pairs for each contiguous True run in a 1-D bool array."""
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return
    edges = np.diff(flags.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if flags[0]:
        starts = [0] + starts
    if flags[-1]:
        ends = ends + [len(flags)]
    yield from zip(starts, ends)


def plot_reconstruction_overlay(
    input_fif,
    recon_fif,
    out_path,
    mask_npz=None,
    title=None,
    max_channels=None,
    window_sec=None,
    demean=True,
    highlight=True,
    input_highpass_hz=None,
    input_lowpass_hz=None,
    input_notch_hz=None,
):
    """Save a stacked per-channel overlay of `input_fif` vs `recon_fif` over the full duration.

    Parameters
    ----------
    input_fif, recon_fif : path-like    original input and reconstructed .fif
    out_path             : path-like    where to write the .png
    mask_npz             : path-like    optional <name>_mask.npz (mask, ch_names) to shade inferred cells
    max_channels         : int|None     cap number of channels drawn (None = all)
    window_sec           : float|None    seconds to plot from the start (None = full recording)
    demean               : bool         subtract each trace's mean for visual alignment
    highlight            : bool         shade inferred (channel, time) regions
    input_highpass_hz,               apply this preprocessing to the plotted "input" so it is
    input_lowpass_hz,                compared in the SAME domain as the reconstruction (the model
    input_notch_hz                   sees highpass/resampled data). Resample-to-recon-rate is
                                     automatic; these mirror the model's filter settings.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_in = mne.io.read_raw_fif(str(input_fif), preload=True, verbose="ERROR")
    raw_rec = mne.io.read_raw_fif(str(recon_fif), preload=True, verbose="ERROR")
    for r in (raw_in, raw_rec):
        try:
            r.pick_types(eeg=True, exclude=[])
        except Exception:
            pass

    # Put the plotted "input" in the SAME domain as the reconstruction: resample to the recon's
    # rate (so the time axes align — inputs are often a different sfreq) and apply the model's
    # preprocessing (highpass/lowpass/notch). Mirrors EEGDataset_v4._prepare_raw (resample first,
    # then filter), so input-vs-reconstruction differences reflect the model, not preprocessing.
    rec_sfreq = raw_rec.info["sfreq"]
    if int(round(raw_in.info["sfreq"])) != int(round(rec_sfreq)):
        raw_in.resample(rec_sfreq, verbose="ERROR")
    if input_highpass_hz is not None or input_lowpass_hz is not None:
        raw_in.filter(l_freq=input_highpass_hz, h_freq=input_lowpass_hz, verbose="ERROR")
    if input_notch_hz:
        raw_in.notch_filter(input_notch_hz, verbose="ERROR")

    # Use the RECONSTRUCTION's channel list as the row set: it's what the model actually produced —
    # the kept channels PLUS any ADDED channels that weren't in the input. Added channels have no
    # input trace (drawn as NaN, so only the reconstruction line shows, fully shaded as inferred).
    ch_names = raw_rec.ch_names
    sfreq = raw_rec.info["sfreq"]
    drec_all = raw_rec.get_data() * 1e6      # µV, (C_rec, N)
    din_all = raw_in.get_data() * 1e6
    T = min(drec_all.shape[1], din_all.shape[1])
    if window_sec is not None:
        T = min(T, int(round(window_sec * sfreq)))   # cap plotted window; None = full recording
    drec = drec_all[:, :T]
    in_by = {c: i for i, c in enumerate(raw_in.ch_names)}
    din = np.full((len(ch_names), T), np.nan, dtype=float)   # NaN where a recon channel isn't in the input
    for i, c in enumerate(ch_names):
        if c in in_by:
            din[i] = din_all[in_by[c], :T]

    # Load the inferred-cell mask (per channel x sample), aligned to input channel order.
    mask = None
    if mask_npz is not None and Path(mask_npz).exists():
        z = np.load(str(mask_npz), allow_pickle=True)
        m, mnames = z["mask"], [str(x) for x in z["ch_names"]]
        mask = np.zeros((len(ch_names), T), dtype=bool)
        for i, c in enumerate(ch_names):
            if c in mnames:
                mask[i] = m[mnames.index(c)][:T]

    n = len(ch_names) if max_channels is None else min(max_channels, len(ch_names))
    t = np.arange(T) / sfreq

    fig, axes = plt.subplots(n, 1, figsize=(16, max(2, 1.4 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        a, b = din[i].copy(), drec[i].copy()
        if demean:
            a = a - np.nanmean(a)
            b = b - np.nanmean(b)

        if highlight and mask is not None:
            for s, e in _contiguous_runs(mask[i]):
                ax.axvspan(t[s], t[min(e, T - 1)], color="#ff9900", alpha=0.16, lw=0)

        ax.plot(t, a, color="#1f77b4", lw=0.7, alpha=0.9, label="input")
        ax.plot(t, b, color="#d62728", lw=0.7, alpha=0.8, label="reconstruction")

        ax.set_ylabel(ch_names[i], rotation=0, ha="right", va="center", fontsize=9)
        ax.set_yticks([])
        ax.margins(x=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        if mask is not None:
            frac = 100.0 * mask[i].mean()
            ax.text(0.997, 0.92, f"{frac:.0f}% inferred", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="#a15c00")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    if highlight and mask is not None:
        fig.text(0.997, 0.995, "shaded = inferred by model", ha="right", va="top",
                 fontsize=8, color="#a15c00")
    fig.suptitle(title or Path(input_fif).stem, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
